"""Agrandissement IA des cases personnages : tailles cibles, cache ``hd_*``, GPU/CPU, câblage pipeline."""

from __future__ import annotations

import json
import os
import types

import cv2
import numpy as np
import pytest

from src import pipeline as pipeline_mod
from src.models.panel import Panel
from src.modules import upscaler as up
from src.modules.figure_panels import FIGURES_DIRNAME
from src.modules.pacing import content_height
from src.modules.slicer import save_panels
from src.modules.timeline_builder import limit_panels_for_duration
from src.pipeline import PipelineOptions, PipelineResult

FRAME = (1920, 1080)


def nearest(rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(rgb, size, interpolation=cv2.INTER_NEAREST)


class Recorder:
    """Fausse fonction d'agrandissement qui compte ses appels."""

    method = "fake"

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, int], tuple[int, int]]] = []

    def __call__(self, rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        self.calls.append(((rgb.shape[1], rgb.shape[0]), size))
        return nearest(rgb, size)


def figure(index: int, width: int, height: int, y: int) -> Panel:
    rng = np.random.default_rng(index)
    image = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    return Panel(index=index, y_start=y, y_end=y + height, height=height, width=width, type="static", image=image)


@pytest.fixture
def chapter(tmp_path):
    """Chapitre avec deux personnages : un petit (à agrandir) et un grand (gardé tel quel)."""
    save_panels([figure(0, 400, 500, 0), figure(1, 700, 1200, 700)], tmp_path / FIGURES_DIRNAME)
    return tmp_path


# --- Tailles cibles ------------------------------------------------------------------------------
def test_target_size_fills_the_frame_within_limits() -> None:
    assert up.target_size(500, 600, FRAME) == (810, 972)  # hauteur limitante : 90 % de 1080
    assert up.target_size(800, 300, FRAME) == (1728, 648)  # largeur limitante : 90 % de 1920
    assert up.target_size(50, 50, FRAME) == (200, 200)  # plafonné à x4
    assert up.target_size(700, 1200, FRAME) is None  # déjà plus grand que le cadre
    assert up.target_size(600, 925, FRAME) is None  # x1,05 : pas la peine
    assert up.target_size(500, 600, (1080, 1920)) == (972, 1166)  # format court


# --- Dossier hd ---------------------------------------------------------------------------------
def test_ensure_upscaled_enlarges_small_figures_and_keeps_native_sizes(chapter) -> None:
    fake = Recorder()
    dst = up.ensure_upscaled(chapter, FRAME, upscale=fake)
    assert dst == chapter / FIGURES_DIRNAME / "hd_1920x1080"
    small, big = json.loads((dst / "panels.json").read_text(encoding="utf-8"))
    assert (small["width"], small["height"], small["native_width"], small["native_height"]) == (778, 972, 400, 500)
    assert small["upscale"] == pytest.approx(972 / 500, abs=1e-3)
    assert (small["y_start"], small["y_end"]) == (0, 500)  # position dans le strip inchangée
    assert (big["width"], big["height"], big["upscale"]) == (700, 1200, 1.0)
    assert fake.calls == [((400, 500), (778, 972))]
    assert cv2.imread(str(dst / "panel_000.png")).shape[:2] == (972, 778)
    assert (dst / "panel_001.png").read_bytes() == (chapter / FIGURES_DIRNAME / "panel_001.png").read_bytes()


def test_ensure_upscaled_caps_the_factor_at_four(tmp_path) -> None:
    save_panels([figure(0, 40, 30, 0)], tmp_path / FIGURES_DIRNAME)
    fake = Recorder()
    up.ensure_upscaled(tmp_path, FRAME, upscale=fake)
    assert fake.calls == [((40, 30), (160, 120))]


def test_ensure_upscaled_reuses_its_cache_until_something_changes(chapter) -> None:
    fake = Recorder()
    up.ensure_upscaled(chapter, FRAME, upscale=fake)
    up.ensure_upscaled(chapter, FRAME, upscale=fake)
    assert len(fake.calls) == 1  # relu du cache

    up.ensure_upscaled(chapter, FRAME, upscale=fake, force=True)
    assert len(fake.calls) == 2

    # Autre cadre : autre dossier. En 9:16 le grand personnage (700 px de large) est agrandi aussi.
    short = up.ensure_upscaled(chapter, (1080, 1920), upscale=fake)
    assert short.name == "hd_1080x1920" and len(fake.calls) == 4
    up.ensure_upscaled(chapter, FRAME, upscale=fake)
    assert len(fake.calls) == 4  # le dossier 1920x1080 est toujours valide

    save_panels([figure(0, 120, 90, 0)], chapter / FIGURES_DIRNAME)  # personnages recalculés
    dst = up.ensure_upscaled(chapter, FRAME, upscale=fake)
    assert len(fake.calls) == 5
    assert json.loads((dst / "panels.json").read_text(encoding="utf-8"))[0]["native_width"] == 120
    assert not (dst / "panel_001.png").exists()  # PNG obsolète supprimé


def test_ensure_upscaled_recomputes_when_a_file_is_missing(chapter) -> None:
    fake = Recorder()
    dst = up.ensure_upscaled(chapter, FRAME, upscale=fake)
    (dst / "panel_000.png").unlink()
    up.ensure_upscaled(chapter, FRAME, upscale=fake)
    assert len(fake.calls) == 2 and (dst / "panel_000.png").is_file()


def test_ensure_upscaled_needs_figures(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        up.ensure_upscaled(tmp_path, FRAME, upscale=Recorder())


def test_missing_model_falls_back_to_lanczos(chapter, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(self):
        raise up.UpscaleError("hors ligne")

    monkeypatch.setattr(up, "_DEFAULT", None)
    monkeypatch.setattr(up.Upscaler, "load", broken)
    dst = up.ensure_upscaled(chapter, FRAME)
    assert json.loads((dst / up.PARAMS_FILE).read_text(encoding="utf-8"))["method"] == "lanczos"
    assert cv2.imread(str(dst / "panel_000.png")).shape[:2] == (972, 778)


# --- Session ONNX : GPU d'abord, CPU en secours ---------------------------------------------------
class FakeSession:
    def __init__(self, provider: str, *, fail: bool = False) -> None:
        self.provider, self.fail, self.calls = provider, fail, 0

    def get_providers(self) -> list[str]:
        return [self.provider]

    def get_inputs(self):
        return [types.SimpleNamespace(name="input")]

    def run(self, _outputs, feed):
        self.calls += 1
        if self.fail:
            raise RuntimeError("pilote GPU en echec")
        x = feed["input"]
        return [x.repeat(4, axis=2).repeat(4, axis=3)]


def upscaler_with(monkeypatch: pytest.MonkeyPatch, gpu: FakeSession | None, cpu: FakeSession) -> up.Upscaler:
    import onnxruntime as ort

    providers = (["DmlExecutionProvider"] if gpu else []) + ["CPUExecutionProvider"]
    monkeypatch.setattr(ort, "get_available_providers", lambda: providers)
    upscaler = up.Upscaler(path="model.onnx")
    upscaler._open = lambda use_gpu: (gpu, "gpu") if use_gpu else (cpu, "cpu")  # type: ignore[method-assign]
    return upscaler


def test_upscaler_uses_the_gpu_and_resizes_to_the_target(monkeypatch: pytest.MonkeyPatch) -> None:
    gpu, cpu = FakeSession("DmlExecutionProvider"), FakeSession("CPUExecutionProvider")
    upscaler = upscaler_with(monkeypatch, gpu, cpu)
    rgb = np.full((10, 20, 3), 200, np.uint8)
    assert upscaler(rgb, (80, 40)).shape == (40, 80, 3)  # x4 exact : pas de redimensionnement
    out = upscaler(rgb, (50, 25))
    assert out.shape == (25, 50, 3) and out.dtype == np.uint8 and int(out.min()) == int(out.max()) == 200
    assert (gpu.calls, cpu.calls, upscaler.device) == (2, 0, "gpu")


def test_upscaler_falls_back_to_cpu_when_the_gpu_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    gpu, cpu = FakeSession("DmlExecutionProvider", fail=True), FakeSession("CPUExecutionProvider")
    upscaler = upscaler_with(monkeypatch, gpu, cpu)
    assert upscaler(np.zeros((4, 4, 3), np.uint8), (16, 16)).shape == (16, 16, 3)
    assert (gpu.calls, cpu.calls, upscaler.device) == (1, 1, "cpu")


def test_upscaler_without_gpu_runs_on_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    cpu = FakeSession("CPUExecutionProvider")
    upscaler = upscaler_with(monkeypatch, None, cpu)
    assert upscaler.load() == "cpu"


@pytest.mark.skipif(os.environ.get("TOONSPLIT_SLOW") != "1", reason="TOONSPLIT_SLOW=1 requis (modèle Hugging Face)")
def test_real_model_enlarges_four_times() -> None:
    upscaler = up.Upscaler()
    rgb = np.zeros((32, 48, 3), np.uint8)
    rgb[8:24, 12:36] = (230, 40, 40)
    out = upscaler.super_resolve(rgb)
    assert out.shape == (128, 192, 3)
    assert abs(int(out[64, 96, 0]) - 230) <= 6 and int(out[2, 2].max()) <= 6  # aplats conservés


# --- Rythme : réglé sur la taille d'origine --------------------------------------------------------
def test_pacing_weighs_panels_by_their_native_height() -> None:
    meta = {
        0: {"height": 972, "native_height": 200},
        1: {"height": 900, "native_height": 900},
        2: {"height": 500},
    }
    assert [content_height(meta[i]) for i in range(3)] == [200, 900, 500]
    # Deux cases pour 5 s à 2,5 s minimum : la plus petite *d'origine* est écartée.
    assert limit_panels_for_duration([0, 1, 2], 5.0, meta, 2.5) == [1, 2]


# --- Pipeline ------------------------------------------------------------------------------------
def test_montage_uses_the_upscaled_figures(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.models.scene import ChapterAnalysis, Scene

    figures_map = [{"index": 0, "x0": 0, "y0": 0, "x1": 400, "y1": 500, "reading_panels": [0]}]
    monkeypatch.setattr(pipeline_mod, "ensure_figures", lambda out_dir, options, force=False: figures_map)
    monkeypatch.setattr(pipeline_mod, "load_panels_meta", lambda panels_dir: [{"index": 0}])
    hd = tmp_path / FIGURES_DIRNAME / "hd_1920x1080"
    calls = []

    def fake_upscaled(out_dir, frame, force=False):
        calls.append(frame)
        return hd

    monkeypatch.setattr(pipeline_mod, "ensure_upscaled", fake_upscaled)
    analysis = ChapterAnalysis(model="m", language="en", n_panels=1, scenes=[
        Scene(index=0, panel_ids=[0], narration="Hello there.", emotion="calm")])
    result = PipelineResult(out_dir=tmp_path)
    panels_dir, display, _ = pipeline_mod._montage_panels(analysis, tmp_path, PipelineOptions(), result)
    assert panels_dir == hd and calls == [FRAME] and display.scenes[0].panel_ids == [0]

    panels_dir, _, _ = pipeline_mod._montage_panels(analysis, tmp_path, PipelineOptions(figure_upscale=False), result)
    assert panels_dir == tmp_path / FIGURES_DIRNAME and calls == [FRAME]


def test_each_format_upscales_for_its_own_frame() -> None:
    assert PipelineOptions(video_format="SHORT").frame() == (1080, 1920)
    assert PipelineOptions(width=1280, height=720).frame() == (1280, 720)
