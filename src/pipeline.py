"""Pipeline de bout en bout : URL Webtoons → cases → scènes → voix off → timeline → CapCut + aperçu.

Le pipeline est découpé en **étapes** réutilisables par l'orchestrateur de lots
(:mod:`src.modules.batch_processor`), qui les répartit sous des sémaphores distincts
(réseau / Gemini d'un côté, charge locale TTS et rendu de l'autre) :

- :func:`stage_scrape_slice` — scraping + découpe (I/O réseau) ;
- :func:`stage_analyze` — analyse Gemini (réseau, quota) ;
- :func:`stage_tts` — voix off Kokoro (CPU) + rapport HTML ;
- :func:`stage_montage` — timeline, brouillon CapCut, aperçu ffmpeg (CPU).

:func:`run_pipeline` les enchaîne pour un seul chapitre. Chaque étape écrit ses
résultats dans le dossier du chapitre et est **réutilisée** si sa sortie existe déjà
(``force=True`` pour tout recalculer) :

```
<out_dir>/
├── chapter.json          métadonnées du chapitre (ChapterMeta)
├── panels.json + panel_NNN.png + debug_overlay.png   (Module 2 : cases de lecture, lues par l'IA)
├── figures/ panels.json + panel_NNN.png + figures_map.json   (cases personnages)
│   └── hd_1920x1080/ panels.json + panel_NNN.png   (personnages agrandis par IA, seuls montés)
├── scenes.json + scenes_report.html                  (Module 3)
├── audio/  scene_NNN.wav + voiceover.json + voiceover_full.wav/.mp3   (Module 4)
├── timeline.json                                     (Module 5)
├── capcut/<nom du projet>/draft_content.json         (Module 5)
└── preview_<N>s.mp4                                  (aperçu ffmpeg)
```
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel

from src.models.audio import VoiceoverManifest
from src.models.chapter import ChapterMeta
from src.models.scene import ChapterAnalysis, CharacterCard
from src.models.timeline import DEFAULT_BGM_GAIN_DB, Timeline
from src.modules.analyzer import (
    BATCH_DELAY_S,
    DEFAULT_KEYFRAME_WORKERS,
    GeminiAnalyzer,
    load_analysis,
    save_analysis,
)
from src.modules.figure_panels import FIGURES_DIRNAME, FigureOptions, ensure_figures, select_figures
from src.modules.upscaler import ensure_upscaled
from src.modules.capcut_builder import (
    DEFAULT_TRANSITION_COMPENSATION,
    CapCutError,
    build_capcut_draft,
    copy_draft,
    detect_capcut_drafts_dir,
)
from src.modules.preview_renderer import PreviewRenderer
from src.modules import asura
from src.modules.scraper import normalize_webtoon_url, parse_ids_from_url, scrape_chapter
from src.modules.script_writer import DEFAULT_MODEL as CLAUDE_SCRIPT_MODEL
from src.modules.script_writer import ClaudeScriptWriter
from src.modules.toonsplit.ai import AiError
from src.modules.slicer import load_panels, render_debug_overlay, save_panels, slice_panels
from src.modules.series_memory import archive_sheets, load_series_context, save_chapter_sheet, series_key
from src.modules.timeline_builder import (
    DEFAULT_CHAPTER_GAP_S,
    DEFAULT_DYNAMICS,
    build_timeline,
    concat_timelines,
    load_panels_meta,
    load_timeline,
    save_timeline,
)
from src.modules.tts_engine import (
    DEFAULT_PADDING_S,
    DEFAULT_SENTENCE_GAP_S,
    KokoroTTS,
    TTSError,
    export_mp3,
    load_manifest,
)
from src.utils.audio_assets import DEFAULT_BGM_DIR, DEFAULT_SFX_DIR, ensure_default_bgm, ensure_default_sfx
from src.utils.config import DEFAULT_NARRATION_LANGUAGE, DEFAULT_SITE_LANGUAGE, PROJECT_ROOT
from src.utils.gemini_manager import GeminiManager
from src.utils.report import build_html_report

logger = logging.getLogger(__name__)

DEFAULT_PREVIEW_SECONDS: float = 60.0
#: Extrait rendu pour une compilation : la vidéo complète se rend dans CapCut, bien plus vite
#: (une compilation de 20 chapitres, 1 h 44, demandait 57 min de rendu ffmpeg et 1,8 Go).
DEFAULT_COMPILATION_PREVIEW_S: float = 120.0
#: Sous-dossier d'une compilation où sont reliés les médias de ses chapitres.
MEDIA_DIRNAME: str = "media"
#: Étapes recalculables par ``--redo``, de la plus amont à la plus aval.
STAGE_RANKS: dict[str, int] = {"analyze": 1, "tts": 2, "montage": 3}


class PipelineOptions(BaseModel):
    """Réglages du pipeline (tous optionnels)."""

    language: str = DEFAULT_NARRATION_LANGUAGE
    site_language: str | None = DEFAULT_SITE_LANGUAGE
    voice: str | None = None
    speed: float = 1.0
    #: Silence entre deux phrases d'une même scène (secondes).
    sentence_gap_s: float = DEFAULT_SENTENCE_GAP_S
    #: Silence en fin de segment, entre deux scènes (secondes).
    padding_s: float = DEFAULT_PADDING_S
    #: IA qui écrit le script : ``claude`` (CLI local, abonnement ; Gemini en repli si Claude
    #: échoue) ou ``gemini`` (analyse Gemini seule).
    script_ai: Literal["claude", "gemini"] = "claude"
    #: Modèle Claude du script (ID explicite, voir :data:`src.modules.script_writer.DEFAULT_MODEL`).
    claude_model: str = CLAUDE_SCRIPT_MODEL
    #: Modèle Gemini (analyse Gemini ou repli).
    model: str | None = None
    #: Images par lot envoyées à Gemini (10 à 15). Plus le lot est gros, moins il y a
    #: d'appels (donc de quota et de temps), au prix d'un découpage en beats plus grossier.
    batch_size: int = 12
    #: Pause forcée entre deux envois d'un même chapitre ; 0 est sûr dès que
    #: ``--max-gemini-rpm`` correspond à la limite réelle du compte.
    gemini_batch_delay_s: float = BATCH_DELAY_S
    #: Appels « cases clés » menés en parallèle (groupes indépendants).
    keyframe_workers: int = DEFAULT_KEYFRAME_WORKERS
    #: Mode « une requête » : script et cases clés en un seul appel Gemini (18 fois moins
    #: d'appels). Bascule automatiquement sur le mode en deux étapes si les images du
    #: chapitre dépassent la limite de charge utile inline.
    single_call: bool = True
    thinking_budget: int | None = None
    width: int = 1920
    height: int = 1080
    fps: int = 60
    preview_seconds: float | None = DEFAULT_PREVIEW_SECONDS
    make_preview: bool = True
    make_capcut: bool = True
    capcut_dir: str | None = None
    project_name: str | None = None
    bgm_file: str | None = None
    #: Dossier des musiques par ambiance (``calm`` / ``tense`` / ``action`` / ``default``) ;
    #: vide = musiques de substitution synthetisees.
    bgm_dir: str | None = str(DEFAULT_BGM_DIR)
    use_bgm: bool = True
    bgm_gain_db: float = DEFAULT_BGM_GAIN_DB
    #: Bruitages aux transitions des scenes d'action (``swoosh`` / ``impact`` / ``roar``).
    use_sfx: bool = True
    sfx_dir: str | None = str(DEFAULT_SFX_DIR)
    sfx_gain_db: float = -12.0
    #: Format de sortie : ``LONG`` (16:9, comportement historique) ou ``SHORT`` (9:16
    #: TikTok / Reels : recadrage par saillance, cadence serree, voix acceleree, outro).
    #: Il pilote la resolution, le rythme, les mouvements, les sous-titres et la voix.
    video_format: str = "LONG"
    #: Dynamisme du montage : ``none`` (sobre), ``subtle`` (sous-titres animes seuls) ou
    #: ``punchy`` (animations, transitions aux changements de scene intenses, effets superposes).
    dynamics: str = DEFAULT_DYNAMICS
    #: Compensation du recouvrement des transitions dans CapCut (``none`` ou ``shift``).
    transition_compensation: str = DEFAULT_TRANSITION_COMPENSATION
    #: Miniature YouTube. **Desactivee par defaut** : elle coute un appel de texte et
    #: surtout une generation d'image, dont le quota est distinct et vite epuise sur un lot.
    make_thumbnail: bool = False
    #: Backend d'images (``gemini`` / ``stability`` / ``local``) ; sinon ``THUMBNAIL_IMAGE_BACKEND``.
    thumbnail_backend: str | None = None
    cta: str | None = None
    #: Recalculer a partir de cette etape (``analyze`` / ``tts`` / ``montage``) en
    #: reutilisant les precedentes : refaire la voix off apres un changement de reglage
    #: ne redepense donc aucun quota Gemini. ``force`` recalcule tout, scraping compris.
    redo: str | None = None
    force: bool = False
    #: Réinjecter dans l'analyse la fiche des personnages des épisodes précédents, pour que
    #: le héros garde le même nom d'un chapitre à l'autre (:mod:`src.modules.series_memory`).
    series_memory: bool = True
    #: Dossier où chercher les chapitres voisins. Par défaut le dossier parent de la sortie ;
    #: à préciser quand ``run --out`` range un chapitre hors du dossier partagé de la série.
    series_root: str | None = None
    #: Cases montées : ``figures`` (défaut) = uniquement les personnages détectés (zones
    #: jaunes de toonsplit, :mod:`src.modules.figure_panels`) ; ``slicer`` = cases entières
    #: du Smart Slicer (comportement historique). L'analyse IA lit toujours les cases entières.
    panels: str = "figures"
    #: Marge autour de chaque personnage (part de sa taille ; 0 = zone jaune exacte).
    figure_margin: float = 0.0
    #: ``cut`` : zone jaune telle quelle (bulles qui la chevauchent coupées) ; ``whole`` :
    #: agrandie aux bulles qu'elle touche.
    figure_bubbles: str = "cut"
    #: Écarter les zones « personne » sans tête détectée (faux positifs : drapeau, hampe).
    figure_require_head: bool = True
    #: Regrouper les personnages d'une même case dans une seule image (``False`` : une image
    #: par personnage).
    figure_group: bool = True
    #: Agrandir les cases personnages par IA (Real-ESRGAN anime) pour qu'elles remplissent
    #: ~90 % du cadre (:mod:`src.modules.upscaler`) ; ``False`` = taille d'origine.
    figure_upscale: bool = True

    @property
    def redo_rank(self) -> int:
        """Rang de la première étape à recalculer (0 = aucune, 1 = analyse, 2 = voix, 3 = montage)."""
        return STAGE_RANKS.get(self.redo or "", 0)

    def recompute(self, stage: str) -> bool:
        """Vrai si ``stage`` doit être recalculé plutôt que réutilisé."""
        rank = self.redo_rank
        return self.force or (rank > 0 and rank <= STAGE_RANKS[stage])

    def figure_options(self) -> FigureOptions:
        """Réglages des cases personnages."""
        return FigureOptions(
            margin=self.figure_margin, bubbles=self.figure_bubbles, require_head=self.figure_require_head,
            group=self.figure_group,
        )

    def frame(self) -> tuple[int, int]:
        """``(largeur, hauteur)`` du cadre vidéo du format choisi."""
        framing = self.profile().framing
        return framing.width, framing.height

    def profile(self):
        """Profil de format correspondant (:class:`~src.models.format_profile.FormatProfile`).

        Les réglages explicites de résolution restent prioritaires en mode long, pour ne
        rien changer aux appels historiques.
        """
        from src.modules.format_factory import VideoConfigFactory

        if str(self.video_format).upper() == "LONG":
            return VideoConfigFactory.create(
                "LONG", framing={"width": self.width, "height": self.height, "fps": self.fps}
            )
        return VideoConfigFactory.create(self.video_format, framing={"fps": self.fps})


@dataclass
class PipelineResult:
    """Chemins produits et durées par étape."""

    out_dir: Path
    chapter: ChapterMeta | None = None
    panels_json: Path | None = None
    scenes_json: Path | None = None
    report_html: Path | None = None
    voiceover_json: Path | None = None
    timeline_json: Path | None = None
    capcut_draft: Path | None = None
    capcut_copy: Path | None = None
    preview_mp4: Path | None = None
    thumbnail: Path | None = None
    n_panels: int = 0
    #: Cases personnages montées (mode ``figures``) ; ``None`` en mode ``slicer``.
    n_figures: int | None = None
    n_scenes: int = 0
    n_punch_in: int = 0
    n_sfx: int = 0
    n_bgm: int = 0
    model: str = ""
    #: Titres affichés quand il n'y a pas de ``ChapterMeta`` (compilation multi-chapitres).
    series_title: str = ""
    episode_title: str = ""
    #: Secondes passées dans les appels à l'IA du script, Claude ou Gemini (inclus dans ``timings["analyze"]``).
    gemini_seconds: float = 0.0
    total_duration_s: float = 0.0
    timings: dict[str, float] = field(default_factory=dict)
    reused: list[str] = field(default_factory=list)


def slug_from_url(url: str) -> str:
    """Nom de dossier lisible dérivé de l'URL (``<serie>_ep<episode_no>``)."""
    if asura.is_asura_url(url):
        series, number = asura.parse_url(url)
        series = re.sub(r"[^a-z0-9]+", "-", (series or "").lower()).strip("-") or "chapter"
        return f"{series}_ep{number:g}" if number is not None else series
    parsed = urlparse(url)
    segments = [s for s in parsed.path.split("/") if s]
    series = segments[2] if len(segments) >= 3 else (segments[-1] if segments else "chapter")
    series = re.sub(r"[^a-z0-9]+", "-", series.lower()).strip("-") or "chapter"
    title_no, episode_no = parse_ids_from_url(url)
    suffix = f"_ep{episode_no}" if episode_no is not None else (f"_t{title_no}" if title_no is not None else "")
    return f"{series}{suffix}"


def project_name_for(meta: ChapterMeta | None, fallback: str) -> str:
    """Nom du projet CapCut : ``<Série> - <Épisode>`` nettoyé pour un nom de dossier."""
    if meta is None or not (meta.series_title or meta.episode_title):
        return fallback
    raw = " - ".join(part for part in (meta.series_title, meta.episode_title) if part)
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', " ", raw)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    return cleaned[:80] or fallback


def _timed(result: PipelineResult, step: str, started: float) -> None:
    result.timings[step] = time.perf_counter() - started


# --- Étapes ------------------------------------------------------------------------------
def stage_scrape_slice(url: str, out_dir: str | Path, options: PipelineOptions, result: PipelineResult | None = None) -> ChapterMeta:
    """Étapes 1-2 : scraping + découpe ; écrit ``chapter.json``, ``panels.json`` et les PNG.

    Les cases ne sont pas conservées en mémoire (elles sont relues du disque par les
    étapes suivantes) : plusieurs chapitres peuvent être traités en parallèle.
    """
    result = result or PipelineResult(out_dir=Path(out_dir))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    url = normalize_webtoon_url(url, options.site_language)
    started = time.perf_counter()
    chapter_json = out_dir / "chapter.json"
    panels_json = out_dir / "panels.json"
    params_json = out_dir / "slice_params.json"
    # La sous-decoupe vise le cadre du format... mais seulement quand l'affichage est un
    # « contain » (la case entiere, fond floute sur les cotes). Le format court recadre en
    # 9:16 par saillance : viser une hauteur de cadre n'y veut rien dire, on desactive.
    framing = options.profile().framing
    slice_params = {
        "frame_height": framing.height if framing.fit == "contain" else 0,
        "frame_width": framing.width,
    }
    reusable = params_json.is_file() and json.loads(params_json.read_text(encoding="utf-8")) == slice_params
    if panels_json.is_file() and chapter_json.is_file() and reusable and not options.force:
        meta = ChapterMeta.model_validate_json(chapter_json.read_text(encoding="utf-8"))
        n_panels = len(load_panels_meta(out_dir))
        result.reused.append("scrape+slice")
    else:
        strip, meta = scrape_chapter(url, language=None)
        panels = slice_panels(strip, **slice_params)
        save_panels(panels, out_dir)
        # A cote de panels.json, jamais dedans : son jeu de cles est un contrat.
        params_json.write_text(json.dumps(slice_params, indent=2), encoding="utf-8")
        render_debug_overlay(strip, panels, out_dir / "debug_overlay.png")
        n_panels = len(panels)
        del strip, panels
        chapter_json.write_text(meta.model_dump_json(indent=2), encoding="utf-8")
    result.chapter, result.panels_json, result.n_panels = meta, panels_json, n_panels
    _timed(result, "scrape+slice", started)
    logger.info("Etape 1-2 : %d case(s) (%s)", n_panels, meta.summary())
    if options.panels == "figures":
        started = time.perf_counter()
        result.n_figures = len(ensure_figures(out_dir, options.figure_options(), force=options.force))
        _timed(result, "figures", started)
        logger.info("Etape 1-2 : %d case(s) personnage(s) detectee(s)", result.n_figures)
        if options.figure_upscale and result.n_figures:
            started = time.perf_counter()
            ensure_upscaled(out_dir, options.frame(), force=options.force)
            _timed(result, "upscale", started)
    return meta


def _series_context(
    meta: ChapterMeta, out_dir: Path, options: PipelineOptions
) -> tuple[list[CharacterCard], str]:
    """Personnages et fin de chapitre hérités des épisodes précédents de la série.

    Renvoie deux valeurs vides quand la mémoire est coupée, quand la série est inconnue,
    ou quand ``meta.episode_no`` est absent : sans numéro d'épisode, impossible de dire
    quels chapitres précèdent celui-ci, et le chapitre risquerait de se relire lui-même.
    """
    if not options.series_memory:
        return [], ""
    if meta.episode_no is None:
        logger.info("Memoire de serie ignoree : numero d'episode inconnu (%s)", meta.url)
        return [], ""
    root = Path(options.series_root) if options.series_root else out_dir.parent
    return load_series_context(root, key=series_key(meta), before_episode=meta.episode_no)


def stage_analyze(
    meta: ChapterMeta,
    out_dir: str | Path,
    options: PipelineOptions,
    result: PipelineResult | None = None,
    *,
    manager: GeminiManager | None = None,
) -> ChapterAnalysis:
    """Étape 3 : script et cases clés du chapitre, ``scenes.json``.

    Claude écrit le script par défaut (:class:`ClaudeScriptWriter`, un appel). S'il échoue
    (CLI introuvable, limite d'usage de l'abonnement, réponse invalide), le chapitre passe
    à l'analyse Gemini (avec point de reprise) plutôt que d'échouer.

    Args:
        manager: gestionnaire Gemini partagé (clés, cascade, RPM) ; construit depuis
            l'environnement si absent.
    """
    result = result or PipelineResult(out_dir=Path(out_dir))
    out_dir = Path(out_dir)
    started = time.perf_counter()
    scenes_json = out_dir / "scenes.json"
    known_characters, previous_tail = _series_context(meta, out_dir, options)
    if scenes_json.is_file() and not options.recompute("analyze"):
        analysis = load_analysis(scenes_json)
        result.reused.append("analyze")
        # Même réutilisée, l'analyse doit laisser sa fiche : un chapitre traité avant la
        # mémoire de série resterait sinon invisible pour tous les suivants.
        if options.series_memory:
            save_chapter_sheet(analysis, out_dir, meta)
    else:
        panels = load_panels(out_dir)
        analysis = None
        if options.script_ai == "claude":
            analysis = _claude_analysis(panels, meta, options, result, known_characters, previous_tail)
        if analysis is None:
            analysis = _gemini_analysis(panels, meta, out_dir, options, result, manager, known_characters, previous_tail)
        del panels
        save_analysis(analysis, scenes_json)
        if options.series_memory:
            save_chapter_sheet(analysis, out_dir, meta)
    result.scenes_json, result.n_scenes, result.model = scenes_json, analysis.n_scenes, analysis.model
    _timed(result, "analyze", started)
    logger.info(
        "Etape 3 : %d scene(s) dont %d filler (%s, %.0fs dont %.0fs chez l'IA)",
        analysis.n_scenes, analysis.n_filler, analysis.model, result.timings["analyze"], result.gemini_seconds,
    )
    return analysis


def _claude_analysis(
    panels: Sequence, meta: ChapterMeta, options: PipelineOptions, result: PipelineResult,
    known_characters: list[CharacterCard], previous_tail: str,
) -> ChapterAnalysis | None:
    """Script écrit par Claude, ou ``None`` si Claude échoue (le chapitre passe à Gemini)."""
    writer = ClaudeScriptWriter(
        model=options.claude_model, language=options.language, cta_text=options.cta,
        known_characters=known_characters, previous_tail=previous_tail,
    )
    try:
        return writer.analyze(panels, meta)
    except AiError as exc:
        logger.warning("Script Claude impossible (%s) : repli sur Gemini", exc)
        return None
    finally:
        result.gemini_seconds += writer.api_seconds


def _gemini_analysis(
    panels: Sequence, meta: ChapterMeta, out_dir: Path, options: PipelineOptions, result: PipelineResult,
    manager: GeminiManager | None, known_characters: list[CharacterCard], previous_tail: str,
) -> ChapterAnalysis:
    """Analyse Gemini : une requête si le chapitre tient dans la charge inline, sinon deux étapes."""
    checkpoint = out_dir / "analysis_checkpoint.json"
    if options.force and checkpoint.is_file():
        checkpoint.unlink()
    analyzer = GeminiAnalyzer(
        manager=manager, model=options.model, batch_size=options.batch_size, language=options.language,
        thinking_budget=options.thinking_budget, cta_text=options.cta,
        delay_between_batches=options.gemini_batch_delay_s, keyframe_workers=options.keyframe_workers,
        known_characters=known_characters, previous_tail=previous_tail,
    )
    # Un chapitre trop lourd est d'abord comprime davantage : le mode deux etapes prive
    # le redacteur des images et le reduit a reformuler des resumes deja aplatis.
    single_call = options.single_call and analyzer.fit_for_single_call(panels)
    try:
        if single_call:
            return analyzer.analyze_panels_single_call(panels, meta)
        # Point de reprise : une analyse coupee par le quota repart des appels deja reussis.
        return analyzer.analyze_panels(panels, meta, checkpoint=checkpoint)
    finally:
        result.gemini_seconds += analyzer.api_seconds  # meme en cas d'echec (quota)


def stage_tts(
    analysis: ChapterAnalysis, out_dir: str | Path, options: PipelineOptions, result: PipelineResult | None = None
) -> VoiceoverManifest:
    """Étape 4 : voix off Kokoro (``audio/``) puis rapport HTML."""
    result = result or PipelineResult(out_dir=Path(out_dir))
    out_dir = Path(out_dir)
    started = time.perf_counter()
    audio_dir = out_dir / "audio"
    voiceover_json = audio_dir / "voiceover.json"
    if voiceover_json.is_file() and not options.recompute("tts"):
        manifest = load_manifest(voiceover_json)
        result.reused.append("tts")
    else:
        # Le format court accelere la voix et resserre les silences : ces reglages
        # viennent du profil, sauf si l'utilisateur les a imposes en ligne de commande.
        audio = options.profile().audio
        defaults = PipelineOptions()
        tts = KokoroTTS(
            language=options.language, voice=options.voice,
            speed=options.speed if options.speed != defaults.speed else audio.speed,
            padding_s=options.padding_s if options.padding_s != defaults.padding_s else audio.padding_s,
            sentence_gap_s=(
                options.sentence_gap_s if options.sentence_gap_s != defaults.sentence_gap_s
                else audio.sentence_gap_s
            ),
            max_silence_s=audio.max_internal_silence_s,
            silence_threshold_db=audio.silence_threshold_db,
        )
        manifest = tts.synthesize_analysis(analysis, audio_dir)
        try:
            export_mp3(audio_dir / manifest.full_file, audio_dir / "voiceover_full.mp3")
        except TTSError as exc:
            logger.warning("MP3 non exporte : %s", exc)
    result.voiceover_json = voiceover_json
    _timed(result, "tts", started)
    logger.info("Etape 4 : %d segment(s), %.1fs de voix off", manifest.n_items, manifest.total_duration_s)

    report_html = out_dir / "scenes_report.html"
    if not report_html.is_file() or options.recompute("tts") or "tts" not in result.reused:
        panels = load_panels(out_dir)
        build_html_report(panels, analysis, report_html, audio=manifest, audio_dir=audio_dir)
        del panels
    result.report_html = report_html
    return manifest


def stage_montage(
    analysis: ChapterAnalysis,
    manifest: VoiceoverManifest,
    meta: ChapterMeta | None,
    out_dir: str | Path,
    options: PipelineOptions,
    result: PipelineResult | None = None,
) -> PipelineResult:
    """Étapes 5-7 : timeline, brouillon CapCut (copié dans CapCut si détecté), aperçu ffmpeg."""
    result = result or PipelineResult(out_dir=Path(out_dir))
    out_dir = Path(out_dir)
    audio_dir = out_dir / "audio"
    started = time.perf_counter()
    sfx_files = ensure_default_sfx(options.sfx_dir) if options.use_sfx else {}
    bgm_files: dict = {}
    if options.use_bgm and options.bgm_dir and not options.bgm_file:
        # Musiques par ambiance ; a defaut, boucles de substitution synthetisees (jamais sans musique).
        bgm_files = ensure_default_bgm(options.bgm_dir)
        logger.info("Musiques par ambiance (%.0f dB) : %s", options.bgm_gain_db, ", ".join(f"{m}={p.name}" for m, p in sorted(bgm_files.items())))
    elif options.bgm_file:
        logger.info("Musique unique (%.0f dB) : %s", options.bgm_gain_db, options.bgm_file)
    else:
        logger.info("Musique de fond desactivee")
    panels_dir, display, panels_meta = _montage_panels(analysis, out_dir, options, result)
    timeline = build_timeline(
        display, manifest, panels_meta, panels_dir=panels_dir, audio_dir=audio_dir,
        width=options.width, height=options.height, fps=options.fps,
        bgm_file=options.bgm_file, bgm_files=bgm_files, bgm_gain_db=options.bgm_gain_db,
        sfx_files=sfx_files, sfx_gain_db=options.sfx_gain_db,
        dynamics=options.dynamics, profile=options.profile(),
    )
    result.timeline_json = save_timeline(timeline, out_dir / "timeline.json")
    result.total_duration_s = timeline.total_duration_s
    result.n_punch_in = sum(1 for c in timeline.clips if c.motion == "punch_in")
    result.n_sfx, result.n_bgm = len(timeline.sfx), len(timeline.bgm)
    _timed(result, "timeline", started)

    if options.make_capcut:
        started = time.perf_counter()
        name = options.project_name or project_name_for(meta, out_dir.name)
        result.capcut_draft = build_capcut_draft(
            timeline, out_dir / "capcut", name,
            dynamics=options.dynamics != "none", transition_compensation=options.transition_compensation,
        )
        capcut_dir = Path(options.capcut_dir) if options.capcut_dir else detect_capcut_drafts_dir()
        if capcut_dir is not None and capcut_dir.is_dir():
            try:
                result.capcut_copy = copy_draft(result.capcut_draft, capcut_dir)
            except CapCutError as exc:
                # Le brouillon existe dans <out_dir>/capcut : la copie n'est qu'un confort.
                logger.warning("Copie dans CapCut impossible, brouillon disponible dans %s : %s", result.capcut_draft, exc)
        _timed(result, "capcut", started)

    if options.make_preview:
        started = time.perf_counter()
        seconds = options.preview_seconds
        label = f"{int(seconds)}s" if seconds else "full"
        preview_path = out_dir / f"preview_{label}.mp4"
        renderer = PreviewRenderer(
            timeline, dynamics=options.dynamics != "none", profile=options.profile()
        )
        renderer.render(
            preview_path, max_duration_s=seconds,
            progress=lambda i, n: logger.info("Rendu apercu : %d/%d images", i, n) if i % (timeline.fps * 10) == 1 or i == n else None,
        )
        result.preview_mp4 = preview_path
        _timed(result, "preview", started)
    return result


def _montage_panels(
    analysis: ChapterAnalysis, out_dir: Path, options: PipelineOptions, result: PipelineResult
) -> tuple[Path, ChapterAnalysis, list[dict]]:
    """Dossier des cases à monter, analyse correspondante et entrées montables de son ``panels.json``.

    En mode ``figures``, seuls les personnages retenus par :func:`select_figures` sont
    proposés au montage (ni trop petits, ni détections douteuses hors cases clés). Sans
    aucun personnage retenu, le chapitre est monté avec ses cases entières plutôt que de
    produire une vidéo vide.
    """
    if options.panels != "figures":
        return out_dir, analysis, load_panels_meta(out_dir)
    figures_map = ensure_figures(out_dir, options.figure_options())  # déjà calculé à l'étape 1 : relu
    display, ids = select_figures(analysis, figures_map, options.frame())
    result.n_figures = len(ids)
    if not ids:
        logger.warning("Aucun personnage assez grand (%d detecte(s)) : montage avec les cases entieres", len(figures_map))
        return out_dir, analysis, load_panels_meta(out_dir)
    n_big = display.n_panels
    logger.info(
        "Personnages montables : %d sur %d (%d trop petit(s), %d douteux hors cases cles)",
        len(ids), len(figures_map), len(figures_map) - n_big, n_big - len(ids),
    )
    # Agrandissement deja fait a l'etape 1 : relu du cache (calcule ici apres un --redo).
    panels_dir = ensure_upscaled(out_dir, options.frame()) if options.figure_upscale else out_dir / FIGURES_DIRNAME
    wanted = set(ids)
    return panels_dir, display, [entry for entry in load_panels_meta(panels_dir) if int(entry["index"]) in wanted]


def stage_thumbnail(
    analysis: ChapterAnalysis,
    out_dir: str | Path,
    options: PipelineOptions,
    result: PipelineResult | None = None,
    *,
    manager: GeminiManager | None = None,
) -> Path | None:
    """Étape 8 (optionnelle) : miniature YouTube. **Ne fait jamais échouer le pipeline.**

    Renvoie le chemin de la miniature, ou ``None`` si elle est désactivée ou si une
    étape a échoué (l'échec est journalisé en avertissement).
    """
    result = result or PipelineResult(out_dir=Path(out_dir))
    if not options.make_thumbnail:
        return None
    from src.modules.thumbnail.image_backends import resolve_backend
    from src.modules.thumbnail.pipeline import ThumbnailStage

    out_dir = Path(out_dir)
    started = time.perf_counter()
    try:
        backend = resolve_backend(options.thumbnail_backend, manager=None)
    except Exception as exc:  # noqa: BLE001 - un backend mal configure ne casse pas la video
        logger.warning("Miniature ignoree : backend indisponible (%s)", exc)
        return None
    produced = ThumbnailStage(manager, backend=backend).run(analysis, out_dir)
    _timed(result, "thumbnail", started)
    if produced is None:
        return None
    result.thumbnail = produced.thumbnail
    return produced.thumbnail


def run_pipeline(
    url: str, out_dir: str | Path, options: PipelineOptions | None = None, *, manager: GeminiManager | None = None
) -> PipelineResult:
    """Exécute toutes les étapes pour un chapitre et renvoie les chemins produits.

    Args:
        url: URL du viewer Webtoons.
        out_dir: dossier de sortie du chapitre.
        options: réglages (voir :class:`PipelineOptions`).
        manager: gestionnaire Gemini partagé (clés, cascade, RPM) ; sinon construit ici.
    """
    options = options or PipelineOptions()
    out_dir = Path(out_dir)
    result = PipelineResult(out_dir=out_dir)
    meta = stage_scrape_slice(url, out_dir, options, result)
    analysis = stage_analyze(meta, out_dir, options, result, manager=manager)
    manifest = stage_tts(analysis, out_dir, options, result)
    stage_montage(analysis, manifest, meta, out_dir, options, result)
    # En dernier, et sans pouvoir echouer : la video est deja complete a ce stade.
    stage_thumbnail(analysis, out_dir, options, result, manager=manager)
    return result


# --- Compilation multi-chapitres -------------------------------------------------------------
def chapter_folders(paths: Sequence[str | Path] = (), pattern: str | None = None) -> list[Path]:
    """Dossiers de chapitres exploitables (``timeline.json`` présent), triés par n° d'épisode.

    Args:
        paths: dossiers explicites.
        pattern: motif glob évalué depuis la racine du projet (ex. ``output/ma-serie_ep*``).

    Raises:
        ValueError: aucun dossier exploitable.
    """
    candidates = [Path(p) for p in paths]
    if pattern:
        candidates.extend(sorted(PROJECT_ROOT.glob(pattern)))
    folders: list[tuple[int, Path]] = []
    for folder in dict.fromkeys(candidates):
        if not (folder / "timeline.json").is_file():
            logger.warning("Ignore %s : pas de timeline.json (chapitre non traite)", folder)
            continue
        episode = 0
        chapter_json = folder / "chapter.json"
        if chapter_json.is_file():
            try:
                episode = ChapterMeta.model_validate_json(chapter_json.read_text(encoding="utf-8")).episode_no or 0
            except ValueError:
                episode = 0
        folders.append((episode, folder))
    if not folders:
        raise ValueError("Aucun chapitre exploitable (timeline.json introuvable)")
    folders.sort(key=lambda item: (item[0], item[1].name))
    return [folder for _, folder in folders]


def _episode_of(folder: Path) -> int | float | None:
    chapter_json = folder / "chapter.json"
    if not chapter_json.is_file():
        return None
    try:
        return ChapterMeta.model_validate_json(chapter_json.read_text(encoding="utf-8")).episode_no
    except ValueError:
        return None


def compilation_label(folders: Sequence[str | Path]) -> str:
    """``Chapitres 1-20`` (numéros du premier et du dernier chapitre), sinon ``N chapitres``."""
    numbers = [n for n in (_episode_of(Path(f)) for f in folders) if n is not None]
    if not numbers:
        return f"{len(folders)} chapitres"
    first, last = min(numbers), max(numbers)
    return f"Chapitre {first:g}" if first == last else f"Chapitres {first:g}-{last:g}"


def compilation_dirname(folders: Sequence[str | Path]) -> str:
    """Nom du dossier d'une compilation : ``<serie>_compilation_ch1-20``.

    La plage fait partie du nom : les chapitres 21-40 d'une série ne remplacent pas la
    compilation des chapitres 1-20. Jamais ``_ep`` dans le nom, qui ferait passer la
    compilation pour un chapitre auprès de ``merge --pattern "<serie>_ep*"``.
    """
    base = Path(folders[0]).name.rsplit("_ep", 1)[0]
    numbers = [n for n in (_episode_of(Path(f)) for f in folders) if n is not None]
    if not numbers:
        return f"{base}_compilation"
    return f"{base}_compilation_ch{min(numbers):g}-{max(numbers):g}"


def _link_or_copy(src: Path, dst: Path) -> None:
    """Lien physique (instantané, aucun octet de plus sur le disque), sinon copie."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def consolidate_media(timeline: Timeline, roots: Sequence[str | Path], media_dir: str | Path) -> Timeline:
    """Timeline dont les médias situés sous ``roots`` sont ramenés dans ``media_dir``.

    Une compilation pointe sur les cases et les voix de chaque chapitre ; avant de
    supprimer les dossiers de chapitres, chaque fichier utilisé est relié (lien physique,
    sinon copie) sous ``media_dir/<chapitre>__<chemin>``. Les fichiers hors de ``roots``
    (musiques, bruitages, effets partagés) ne bougent pas.
    """
    roots = [Path(r).resolve() for r in roots]
    media_dir = Path(media_dir).resolve()
    moved: dict[Path, str] = {}

    def local(file: str) -> str:
        if not file:
            return file
        src = Path(file).resolve()
        root = next((r for r in roots if src.is_relative_to(r)), None)
        if root is None:
            return file
        if src not in moved:
            dst = media_dir / f"{root.name}__{'_'.join(src.relative_to(root).parts)}"
            _link_or_copy(src, dst)
            moved[src] = str(dst)
        return moved[src]

    consolidated = timeline.model_copy(update={
        "clips": [c.model_copy(update={"file": local(c.file)}) for c in timeline.clips],
        "audio": [a.model_copy(update={"file": local(a.file)}) for a in timeline.audio],
        "sfx": [c.model_copy(update={"file": local(c.file)}) for c in timeline.sfx],
        "bgm": [c.model_copy(update={"file": local(c.file)}) for c in timeline.bgm],
        "vfx": [c.model_copy(update={"file": local(c.file)}) for c in timeline.vfx],
        "panels_dir": str(media_dir), "audio_dir": str(media_dir),
    })
    logger.info("Compilation autonome : %d media(s) relie(s) dans %s", len(moved), media_dir)
    return consolidated


def delete_chapter_folders(folders: Sequence[str | Path], keep: str | Path) -> int:
    """Supprime les dossiers de chapitres d'une compilation terminée ; renvoie les octets libérés.

    ``keep`` (le dossier de la compilation) et ses parents ne sont jamais supprimés.
    """
    keep = Path(keep).resolve()
    freed = 0
    for folder in folders:
        folder = Path(folder).resolve()
        if folder == keep or keep.is_relative_to(folder) or not folder.is_dir():
            continue
        # Les médias reliés à la compilation partagent leurs octets : seuls les fichiers
        # sans autre lien libèrent réellement de la place.
        freed += sum(f.stat().st_size for f in folder.rglob("*") if f.is_file() and f.stat().st_nlink <= 1)
        shutil.rmtree(folder)
        logger.info("Dossier de chapitre supprime : %s", folder)
    return freed


def build_compilation(
    folders: Sequence[str | Path],
    out_dir: str | Path,
    *,
    name: str | None = None,
    gap_s: float = DEFAULT_CHAPTER_GAP_S,
    preview_seconds: float | None = DEFAULT_COMPILATION_PREVIEW_S,
    make_capcut: bool = True,
    capcut_dir: str | None = None,
    delete_chapters: bool = False,
) -> PipelineResult:
    """Fusionne plusieurs chapitres déjà traités en **un seul projet CapCut** (et un extrait).

    La compilation est autonome : les cases et voix utilisées sont reliées dans
    ``<out_dir>/media/`` (liens physiques, aucune place en plus) et les fiches de la mémoire
    de série archivées dans ``<out_dir>/series_memory/``. Avec ``delete_chapters``, les
    dossiers de chapitres sont ensuite supprimés : le projet CapCut, l'extrait et la suite
    de la série (chapitres suivants) n'en dépendent plus.

    Args:
        folders: dossiers de chapitres (voir :func:`chapter_folders`), dans l'ordre voulu.
        out_dir: dossier de la compilation (``timeline.json``, ``media/``, ``capcut/``, extrait).
        name: nom du projet CapCut (défaut : ``<Série> - Chapitres 1-20``).
        gap_s: silence entre deux chapitres.
        preview_seconds: durée de l'extrait MP4 (``None`` = aucun, 0 = vidéo complète).
            La vidéo complète se rend dans CapCut, bien plus vite que ffmpeg.
        make_capcut: générer le brouillon CapCut.
        capcut_dir: dossier des projets CapCut (copie du brouillon).
        delete_chapters: supprimer les dossiers de chapitres une fois la compilation écrite.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    folders = [Path(folder) for folder in folders]
    result = PipelineResult(out_dir=out_dir)
    started = time.perf_counter()
    timelines = [load_timeline(folder / "timeline.json") for folder in folders]
    series = next((t.series_title for t in timelines if t.series_title), "")
    label = compilation_label(folders)
    merged = concat_timelines(
        timelines, gap_s=gap_s, series_title=series, episode_title=label, panels_dir=out_dir, audio_dir=out_dir,
    )
    merged = consolidate_media(merged, folders, out_dir / MEDIA_DIRNAME)
    archive_sheets(folders, out_dir)
    result.timeline_json = save_timeline(merged, out_dir / "timeline.json")
    result.total_duration_s = merged.total_duration_s
    result.series_title, result.episode_title = merged.series_title, merged.episode_title
    result.n_scenes = merged.n_scenes
    result.n_panels = len(merged.clips)
    result.n_punch_in = sum(1 for c in merged.clips if c.motion == "punch_in")
    result.n_sfx, result.n_bgm = len(merged.sfx), len(merged.bgm)
    _timed(result, "timeline", started)

    project = name or f"{series} - {label}".strip(" -")
    project = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', " ", project)
    project = re.sub(r"\s+", " ", project).strip(" .")[:80] or out_dir.name
    if make_capcut:
        started = time.perf_counter()
        result.capcut_draft = build_capcut_draft(merged, out_dir / "capcut", project)
        target = Path(capcut_dir) if capcut_dir else detect_capcut_drafts_dir()
        if target is not None and target.is_dir():
            try:
                result.capcut_copy = copy_draft(result.capcut_draft, target)
            except CapCutError as exc:
                logger.warning("Copie dans CapCut impossible, brouillon disponible dans %s : %s", result.capcut_draft, exc)
        _timed(result, "capcut", started)

    if preview_seconds is not None:
        started = time.perf_counter()
        label_s = f"{int(preview_seconds)}s" if preview_seconds else "full"
        preview_path = out_dir / f"preview_{label_s}.mp4"
        PreviewRenderer(merged).render(
            preview_path, max_duration_s=preview_seconds or None,
            progress=lambda i, n: logger.info("Rendu apercu : %d/%d images", i, n) if i % (merged.fps * 30) == 1 or i == n else None,
        )
        result.preview_mp4 = preview_path
        _timed(result, "preview", started)
    if delete_chapters:
        freed = delete_chapter_folders(folders, keep=out_dir)
        logger.info("%d dossier(s) de chapitre supprime(s), %.1f Go liberes", len(folders), freed / 1e9)
    logger.info(
        "Compilation ecrite : %s (%d chapitres, %.1f min)", out_dir, len(timelines), merged.total_duration_s / 60,
    )
    return result


def format_result(result: PipelineResult) -> str:
    """Résumé ASCII du pipeline (console Windows cp1252)."""
    meta = result.chapter
    lines = [
        "=" * 72,
        f"Serie      : {(meta.series_title if meta else result.series_title) or '-'}",
        f"Episode    : {(meta.episode_title if meta else result.episode_title) or '-'}",
        f"Cases      : {result.n_panels}" + (f" (personnages montes : {result.n_figures})" if result.n_figures is not None else "")
        + f"   Scenes : {result.n_scenes}   Duree : {result.total_duration_s:.1f}s   Modele : {result.model or '-'}",
        f"Montage    : {result.n_punch_in} punch-in, {result.n_sfx} bruitage(s), {result.n_bgm} segment(s) musique",
        f"Dossier    : {result.out_dir}",
    ]
    for label, path in (
        ("Cases", result.panels_json), ("Scenes", result.scenes_json), ("Rapport", result.report_html),
        ("Voix off", result.voiceover_json), ("Timeline", result.timeline_json),
        ("CapCut", result.capcut_draft), ("CapCut copie", result.capcut_copy), ("Apercu", result.preview_mp4),
        ("Miniature", result.thumbnail),
    ):
        if path is not None:
            lines.append(f"  {label:<12}: {path}")
    if result.reused:
        lines.append(f"Reutilise  : {', '.join(result.reused)}")
    lines.append("Durees     : " + ", ".join(f"{k} {v:.0f}s" for k, v in result.timings.items()))
    lines.append("=" * 72)
    return "\n".join(lines).encode("ascii", "replace").decode("ascii")


__all__ = [
    "DEFAULT_PREVIEW_SECONDS",
    "PipelineOptions",
    "PipelineResult",
    "slug_from_url",
    "project_name_for",
    "stage_scrape_slice",
    "stage_analyze",
    "stage_tts",
    "stage_montage",
    "stage_thumbnail",
    "run_pipeline",
    "chapter_folders",
    "build_compilation",
    "compilation_dirname",
    "compilation_label",
    "consolidate_media",
    "delete_chapter_folders",
    "DEFAULT_COMPILATION_PREVIEW_S",
    "MEDIA_DIRNAME",
    "format_result",
]
