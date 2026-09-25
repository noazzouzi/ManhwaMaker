"""toonsplit, étapes 3 et 5 : spec IA et juge (Gemini simulé, aucun appel réseau)."""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from src.modules.toonsplit import ai
from src.modules.toonsplit.ai import (
    AiError, AiJudge, AiSpecProvider, BlockSpec, ClaudeCliJson, GeminiJson, JudgeVerdict, ManualSpecProvider,
    fallback_spec, judge_images, make_client, normalize_spec, ruler_image,
)
from src.modules.toonsplit.geometry import HEAD, PERSON
from tests.toonsplit_helpers import FakeManager, art, box

GOOD = {
    "role": "key",
    "keep": [{"what": "hero", "y": [0.2, 0.9], "priority": 1}],
    "drop": [{"what": "caption", "y": [0.0, 0.15], "kind": "narration"}],
    "tts": ["My name is Suho Kim.", "  "],
    "speaker_hints": ["narrator"],
}


# --- Validation ----------------------------------------------------------------------------------
def test_normalize_spec_clamps_small_overshoots_and_drops_empty_text() -> None:
    spec = BlockSpec.model_validate({**GOOD, "keep": [{"what": "hero", "y": [-0.02, 1.03]}]})
    fixed, errors = normalize_spec(spec)
    assert errors == []
    assert fixed.keep[0].y == [0.0, 1.0]
    assert fixed.tts == ["My name is Suho Kim."]


@pytest.mark.parametrize("bad", [
    {"keep": [{"what": "hero", "y": [0.6, 0.2]}]},
    {"keep": [{"what": "hero", "y": [0.2]}]},
    {"keep": [{"what": "hero", "y": [0.2, 1.4]}]},
    {"keep": []},
])
def test_normalize_spec_reports_invalid_zones(bad: dict) -> None:
    _, errors = normalize_spec(BlockSpec.model_validate({**GOOD, **bad}))
    assert errors


def test_text_only_block_may_have_no_keep_zone() -> None:
    _, errors = normalize_spec(BlockSpec(role="text_only", tts=["Hi."]))
    assert errors == []


def test_fallback_spec_keeps_detected_subjects() -> None:
    spec = fallback_spec([box(100, 200, HEAD), box(150, 600, PERSON)], 1000)
    assert spec.role == "key" and spec.keep[0].y == [0.1, 0.6]
    assert fallback_spec([], 1000).keep[0].y == [0.0, 1.0]


# --- Images envoyées ----------------------------------------------------------------------------------
def test_ruler_image_adds_margins_and_caps_height() -> None:
    img = ruler_image(art(500, 400))
    assert img.shape[0] > 500 and img.shape[1] > 400
    assert (img[:, :20] != 255).any()  # graduations dans la marge gauche
    tall = ruler_image(art(5000, 400), max_height=1000)
    assert tall.shape[0] < 1100


def test_judge_images_give_overview_then_each_candidate() -> None:
    views = judge_images(art(2000, 400), [(0, 600), (300, 900)], max_side=1000)
    assert [caption.split(":")[0] for caption, _ in views] == ["Image 1", "Image 2", "Image 3"]
    overview, first = views[0][1], views[1][1]
    assert overview.shape[:2] == (1000, 200)  # réduit au plus grand côté permis
    assert first.shape[:2] == (600, 400)  # candidat en pleine résolution s'il tient


# --- Client JSON ------------------------------------------------------------------------------------
def test_ask_retries_with_the_error_then_succeeds() -> None:
    manager = FakeManager("not json", {**GOOD, "keep": []}, GOOD)
    client = GeminiJson(manager, retries=2)
    spec = client.ask("system", "prompt", [], BlockSpec, label="t", check=normalize_spec)
    assert spec.keep[0].what == "hero" and client.n_calls == 3
    retry_text = manager.calls[2][0][0].text
    assert retry_text.startswith("prompt")
    assert "needs at least one keep zone" in retry_text and "Previous answer" in retry_text
    config = manager.calls[0][1]
    assert config.temperature == 0.0 and config.response_mime_type == "application/json"
    assert config.response_schema is BlockSpec and config.seed == 7


def test_ask_gives_up_after_retries() -> None:
    client = GeminiJson(FakeManager("{}", "{}"), retries=1)
    with pytest.raises(AiError):
        client.ask("s", "p", [], BlockSpec, label="t", check=normalize_spec)


def test_ask_accepts_fenced_json() -> None:
    client = GeminiJson(FakeManager("```json\n" + json.dumps(GOOD) + "\n```"))
    assert client.ask("s", "p", [], BlockSpec, label="t").role == "key"


# --- Fournisseurs ----------------------------------------------------------------------------------
def test_spec_provider_caches_by_block_hash(tmp_path) -> None:
    manager = FakeManager(GOOD, {**GOOD, "role": "insert"})
    provider = AiSpecProvider(GeminiJson(manager), cache_dir=tmp_path)
    block = art(300, seed=1)
    first = provider(block, 0)
    again = provider(block.copy(), 5)  # même pixels, autre position : cache
    assert first == again and len(manager.calls) == 1 and provider.cache_hits == 1
    other = provider(art(300, seed=2), 1)
    assert other.role == "insert" and len(manager.calls) == 2
    assert len(list(tmp_path.glob("*.json"))) == 2
    # l'image envoyée porte la règle graduée (plus large que le bloc)
    image_part = manager.calls[0][0][1]
    assert image_part.inline_data.mime_type == "image/jpeg"


def test_judge_validates_the_candidate_number_and_describes_candidates(tmp_path) -> None:
    manager = FakeManager({"best": 4, "reason": "?"}, {"best": 2, "reason": "keeps the bubbles", "confident": True})
    judge = AiJudge(GeminiJson(manager), cache_dir=tmp_path)
    spec = BlockSpec.model_validate(GOOD)
    verdict = judge(art(900), [(0, 600), (0, 900), (100, 700)], spec, 3, ["follows", "keeps bubbles", ""])
    assert verdict == JudgeVerdict(best=2, reason="keeps the bubbles", confident=True)
    parts = manager.calls[0][0]
    prompt = parts[0].text
    assert "Candidate 2: from 0.00 to 1.00 of the block height (keeps bubbles)" in prompt
    assert "Keep: hero" in prompt
    # vue d'ensemble puis un recadrage par candidat, chacun précédé de sa légende
    assert [p.text for p in parts if p.text][1:] == [
        "Image 1: the whole block, candidate windows outlined and numbered.", "Image 2: candidate 1.",
        "Image 3: candidate 2.", "Image 4: candidate 3."]
    assert sum(p.inline_data is not None for p in parts) == 4
    assert judge(art(900), [(0, 600), (0, 900), (100, 700)], spec, 3, ["follows", "keeps bubbles", ""]) == verdict
    assert len(manager.calls) == 2  # 2e appel identique : cache


def test_manual_provider_reads_prototype_format() -> None:
    provider = ManualSpecProvider([{"block": 2, **GOOD}, {"block": 4, "role": "text_only", "tts": ["Hi."]}])
    assert provider(np.zeros((1, 1, 3), np.uint8), 2).keep[0].what == "hero"
    assert provider(np.zeros((1, 1, 3), np.uint8), 4).role == "text_only"
    with pytest.raises(KeyError):
        provider(np.zeros((1, 1, 3), np.uint8), 0)
    with pytest.raises(ValueError):
        ManualSpecProvider([{"block": 0, "role": "key", "keep": []}])


# --- Transport Claude CLI (sous-processus simulé) -------------------------------------------------
def cli_output(structured=None, *, result="", is_error=False, cost=0.02) -> str:
    events = [
        {"type": "system", "subtype": "init"},
        {"type": "result", "subtype": "success" if not is_error else "error", "is_error": is_error, "result": result,
         "structured_output": structured, "total_cost_usd": cost, "modelUsage": {"claude-opus-5": {}}},
    ]
    return "\n".join(json.dumps(e) for e in events) + "\n"


class FakeRun:
    def __init__(self, *outputs: str, stderr: str = "") -> None:
        self.outputs = list(outputs)
        self.stderr = stderr
        self.calls: list[dict] = []

    def __call__(self, cmd, **kwargs):
        self.calls.append({"cmd": cmd, **kwargs})
        return SimpleNamespace(stdout=self.outputs.pop(0) if self.outputs else "", stderr=self.stderr, returncode=0)


def test_claude_cli_call_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    run = FakeRun(cli_output(GOOD))
    monkeypatch.setattr(ai.subprocess, "run", run)
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "parent")
    monkeypatch.setenv("CLAUDECODE", "1")
    client = ClaudeCliJson(model="sonnet", effort="low", executable="claude.exe")
    spec = client.ask("SYSTEM\nline 2", "Analyse it.", [("", b"jpegbytes")], BlockSpec, label="t", check=normalize_spec)
    assert spec.keep[0].what == "hero" and client.cost_usd == 0.02 and client.models_used == {"claude-opus-5"}
    call = run.calls[0]
    cmd = call["cmd"]
    assert cmd[:2] == ["claude.exe", "-p"]
    assert cmd[cmd.index("--tools") + 1] == "" and "--no-session-persistence" in cmd
    assert cmd[cmd.index("--setting-sources") + 1] == ""  # pas de CLAUDE.md utilisateur dans la requête
    assert cmd[cmd.index("--system-prompt") + 1] == "SYSTEM\nline 2"
    assert cmd[cmd.index("--model") + 1] == "sonnet" and cmd[cmd.index("--effort") + 1] == "low"
    schema = cmd[cmd.index("--json-schema") + 1]
    assert "$ref" not in schema and "$defs" not in schema and '"enum"' in schema
    message = json.loads(call["input"])
    content = message["message"]["content"]
    assert content[0] == {"type": "text", "text": "Analyse it."}
    assert content[1]["type"] == "image" and content[1]["source"]["media_type"] == "image/jpeg"
    assert "CLAUDE_CODE_SESSION_ID" not in call["env"] and "CLAUDECODE" not in call["env"]
    assert client.cache_tag == "claude-cli:sonnet"


def test_claude_cli_retries_on_invalid_answer_then_accepts_text_json(monkeypatch: pytest.MonkeyPatch) -> None:
    run = FakeRun(cli_output({"role": "key", "keep": []}), cli_output(None, result=json.dumps(GOOD)))
    monkeypatch.setattr(ai.subprocess, "run", run)
    client = ClaudeCliJson(executable="claude")
    assert client.ask("s", "p", [], BlockSpec, label="t", check=normalize_spec).role == "key"
    retry = json.loads(run.calls[1]["input"])["message"]["content"][0]["text"]
    assert "needs at least one keep zone" in retry


def test_claude_cli_usage_limit_stops_further_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    run = FakeRun(cli_output(None, result="Claude AI usage limit reached", is_error=True))
    monkeypatch.setattr(ai.subprocess, "run", run)
    client = ClaudeCliJson(executable="claude")
    with pytest.raises(AiError, match="limite"):
        client.ask("s", "p", [], BlockSpec, label="t")
    with pytest.raises(AiError):
        client.ask("s", "p", [], BlockSpec, label="t")
    assert len(run.calls) == 1


def test_claude_cli_timeout_counts_as_invalid_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    def slow(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 1)

    monkeypatch.setattr(ai.subprocess, "run", slow)
    with pytest.raises(AiError, match="timed out"):
        ClaudeCliJson(executable="claude", retries=1).ask("s", "p", [], BlockSpec, label="t")


def test_find_claude_prefers_npm_exe_over_cmd_shim(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    shim = tmp_path / "claude.cmd"
    shim.write_text("@echo off")
    exe = tmp_path / "node_modules" / "@anthropic-ai" / "claude-code" / "bin" / "claude.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"")
    monkeypatch.delenv(ai.CLAUDE_BIN_ENV, raising=False)
    monkeypatch.setattr(ai.shutil, "which", lambda name: str(shim))
    monkeypatch.setattr(ai.os, "name", "nt")
    assert ai.find_claude() == str(exe)
    monkeypatch.setenv(ai.CLAUDE_BIN_ENV, "C:/bin/claude.exe")
    assert ai.find_claude() == "C:/bin/claude.exe"


def test_cache_is_separate_per_provider(tmp_path) -> None:
    gemini = AiSpecProvider(GeminiJson(FakeManager(GOOD)), cache_dir=tmp_path)
    block = art(200, seed=9)
    gemini(block, 0)

    class Canned(ClaudeCliJson):
        def _send(self, *args, **kwargs):
            return {**GOOD, "role": "insert"}

    claude = AiSpecProvider(Canned(executable="x"), cache_dir=tmp_path)
    assert claude(block, 0).role == "insert" and claude.cache_hits == 0 and claude.source == "claude"
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_make_client() -> None:
    assert isinstance(make_client("claude", model="opus", max_rpm=5), ClaudeCliJson)
    assert ClaudeCliJson(executable="x").cache_tag == "claude-cli:claude-opus-5"
    assert isinstance(make_client("gemini", model="opus", max_rpm=5), GeminiJson)
    with pytest.raises(ValueError):
        make_client("gpt")
