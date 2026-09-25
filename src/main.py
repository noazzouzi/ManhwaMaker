"""CLI Auto-Manhwa Recap Generator.

Usage (depuis la racine du projet) :

    python -m src.main run "<url du viewer Webtoons ou du chapitre Asura>"  # toutes les etapes, un chapitre
    python -m src.main run "<url>" --out output/mon-chapitre --voice af_heart --preview-seconds 30
    python -m src.main batch "<url de la serie ou d'un episode>" --start-chapter 1 --end-chapter 20
    python -m src.main batch "<url de la serie>" --start-chapter 1 --end-chapter 20 --compile   # une seule video
    python -m src.main batch --url-list chapitres.txt --max-chapters 5 --max-gemini-rpm 10 --max-tts-workers 2
    python -m src.main preview output/<chapitre> --seconds 60          # re-rendre l'apercu seul
    python -m src.main capcut output/<chapitre> --capcut-dir ...       # regenerer le brouillon seul

Cles Gemini : GEMINI_API_KEYS=cle1,cle2 dans .env (rotation automatique), sinon GEMINI_API_KEY
ou .gemini_key. Cascade de modeles : GEMINI_MODEL_CASCADE (voir src/utils/gemini_manager.py).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer

from src.pipeline import (
    DEFAULT_COMPILATION_PREVIEW_S,
    DEFAULT_PREVIEW_SECONDS,
    PipelineOptions,
    format_result,
    run_pipeline,
    slug_from_url,
)
from src.utils.config import DEFAULT_NARRATION_LANGUAGE, PROJECT_ROOT

app = typer.Typer(add_completion=False, help="Auto-Manhwa Recap Generator : URL Webtoons ou Asura Scans -> projet CapCut + apercu.")


def _setup(verbose: bool) -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(errors="replace")
            except (ValueError, OSError):
                pass
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _ascii(text: object) -> str:
    return str(text).encode("ascii", "replace").decode("ascii")


def _pipeline_options(
    *, language, voice, speed, model, thinking_budget, preview_seconds, no_preview, no_capcut, capcut_dir,
    project_name, bgm, bgm_dir, no_bgm, sfx_dir, no_sfx, cta, no_cta, fps, force, redo=None,
    sentence_gap=None, padding=None, batch_size=None, gemini_batch_delay=None, keyframe_workers=None,
    multi_call=False, dynamics=None, transition_compensation=None, thumbnail=False, thumbnail_backend=None,
    video_format=None, no_series_memory=False, panels="figures", figure_margin=0.0, figure_bubbles="cut",
    figures_separate=False, no_upscale=False, script_ai="claude", claude_model=None,
) -> PipelineOptions:
    overrides = {"panels": panels, "figure_margin": figure_margin, "figure_bubbles": figure_bubbles,
                 "figure_group": not figures_separate, "figure_upscale": not no_upscale,
                 "script_ai": str(script_ai).lower()}
    if claude_model:
        overrides["claude_model"] = claude_model
    if video_format is not None:
        overrides["video_format"] = str(video_format).upper()
    if dynamics is not None:
        overrides["dynamics"] = dynamics
    if transition_compensation is not None:
        overrides["transition_compensation"] = transition_compensation
    if thumbnail_backend is not None:
        overrides["thumbnail_backend"] = thumbnail_backend
    overrides["make_thumbnail"] = bool(thumbnail)
    if bgm_dir is not None:
        overrides["bgm_dir"] = str(bgm_dir)
    if sfx_dir is not None:
        overrides["sfx_dir"] = str(sfx_dir)
    if sentence_gap is not None:
        overrides["sentence_gap_s"] = sentence_gap
    if padding is not None:
        overrides["padding_s"] = padding
    if batch_size is not None:
        overrides["batch_size"] = batch_size
    if gemini_batch_delay is not None:
        overrides["gemini_batch_delay_s"] = gemini_batch_delay
    if keyframe_workers is not None:
        overrides["keyframe_workers"] = keyframe_workers
    overrides["single_call"] = not multi_call
    overrides["series_memory"] = not no_series_memory
    return PipelineOptions(
        language=language, voice=voice, speed=speed, model=model, thinking_budget=thinking_budget,
        preview_seconds=preview_seconds if preview_seconds > 0 else None, make_preview=not no_preview,
        make_capcut=not no_capcut, capcut_dir=str(capcut_dir) if capcut_dir else None,
        project_name=project_name, bgm_file=str(bgm) if bgm else None, fps=fps, force=force,
        cta="" if no_cta else cta, use_sfx=not no_sfx, use_bgm=not no_bgm, redo=redo, **overrides,
    )


@app.command()
def run(
    url: Annotated[str, typer.Argument(help="URL du viewer Webtoons (.../viewer?title_no=..&episode_no=..) ou d'un chapitre Asura Scans (.../comics/<serie>/chapter/<n>).")],
    out: Annotated[Optional[Path], typer.Option("--out", "-o", help="Dossier de sortie (defaut output/<serie>_epN).")] = None,
    language: Annotated[str, typer.Option("--language", help="Langue de la narration et de la voix.")] = DEFAULT_NARRATION_LANGUAGE,
    voice: Annotated[Optional[str], typer.Option("--voice", help="Voix Kokoro (defaut am_fenrir,am_michael en anglais, melange des deux ; voir la commande voices).")] = None,
    speed: Annotated[float, typer.Option("--speed", help="Vitesse de la voix.")] = 1.0,
    sentence_gap: Annotated[Optional[float], typer.Option("--sentence-gap", help="Silence ajoute entre deux phrases (defaut 0 : pauses laissees a Kokoro).")] = None,
    padding: Annotated[Optional[float], typer.Option("--padding", help="Silence ajoute en fin de scene (defaut 0 : silence naturel de Kokoro).")] = None,
    script_ai: Annotated[str, typer.Option("--script-ai", help="IA qui ecrit le script : claude (defaut, CLI local et abonnement ; Gemini en repli si Claude echoue) ou gemini.")] = "claude",
    claude_model: Annotated[Optional[str], typer.Option("--claude-model", help="Modele Claude du script (defaut claude-opus-5-5).")] = None,
    model: Annotated[Optional[str], typer.Option("--model", help="Modele Gemini prefere, pour --script-ai gemini ou le repli (la cascade de secours suit).")] = None,
    thinking_budget: Annotated[Optional[int], typer.Option("--thinking-budget", help="Budget de reflexion Gemini.")] = None,
    batch_size: Annotated[Optional[int], typer.Option("--batch-size", help="Images par lot Gemini (10-15, defaut 12) ; 15 = ~20 %% d'appels en moins.")] = None,
    gemini_batch_delay: Annotated[Optional[float], typer.Option("--gemini-batch-delay", help="Pause forcee entre deux envois (defaut 2,5 s ; 0 est sur avec --max-gemini-rpm).")] = None,
    keyframe_workers: Annotated[Optional[int], typer.Option("--keyframe-workers", help="Appels 'cases cles' en parallele (defaut 4, mode deux etapes).")] = None,
    multi_call: Annotated[bool, typer.Option("--multi-call", help="Mode historique en deux etapes (~18 appels) au lieu de la requete unique.")] = False,
    video_format: Annotated[str, typer.Option("--format", help="Format de sortie : LONG (16:9, defaut) ou SHORT (9:16 TikTok : recadrage 9:16, cadence 1,2 s, voix x1,20, outro).")] = "LONG",
    dynamics: Annotated[str, typer.Option("--dynamics", help="Dynamisme du montage : none (sobre), subtle (sous-titres animes) ou punchy (defaut : + transitions et effets).")] = "punchy",
    transition_compensation: Annotated[str, typer.Option("--transition-compensation", help="Recouvrement des transitions CapCut : none (defaut) ou shift (a verifier dans l'editeur).")] = "none",
    thumbnail: Annotated[bool, typer.Option("--thumbnail", help="Generer aussi la miniature YouTube (1 appel texte + 1 generation d'image).")] = False,
    thumbnail_backend: Annotated[Optional[str], typer.Option("--thumbnail-backend", help="Backend d'images : gemini (defaut), stability ou local.")] = None,
    preview_seconds: Annotated[float, typer.Option("--preview-seconds", help="Duree de l'apercu (0 = complet).")] = DEFAULT_PREVIEW_SECONDS,
    no_preview: Annotated[bool, typer.Option("--no-preview", help="Ne pas rendre l'apercu MP4.")] = False,
    no_capcut: Annotated[bool, typer.Option("--no-capcut", help="Ne pas generer le brouillon CapCut.")] = False,
    capcut_dir: Annotated[Optional[Path], typer.Option("--capcut-dir", help="Dossier des projets CapCut (copie du brouillon).")] = None,
    project_name: Annotated[Optional[str], typer.Option("--name", help="Nom du projet CapCut.")] = None,
    bgm: Annotated[Optional[Path], typer.Option("--bgm", help="Musique de fond unique (WAV/MP3), bouclee, mixee a -22 dB.")] = None,
    bgm_dir: Annotated[Optional[Path], typer.Option("--bgm-dir", help="Dossier des musiques par ambiance calm/tense/action (defaut config/bgm ; vide = musiques de substitution).")] = None,
    no_bgm: Annotated[bool, typer.Option("--no-bgm", help="Aucune musique de fond.")] = False,
    sfx_dir: Annotated[Optional[Path], typer.Option("--sfx-dir", help="Dossier des bruitages swoosh/impact/roar (defaut config/sfx).")] = None,
    no_sfx: Annotated[bool, typer.Option("--no-sfx", help="Aucun bruitage aux transitions des scenes d'action.")] = False,
    cta: Annotated[Optional[str], typer.Option("--cta", help="Phrase d'appel a l'abonnement (defaut : selon la langue).")] = None,
    no_cta: Annotated[bool, typer.Option("--no-cta", help="Aucun appel a l'abonnement dans le script.")] = False,
    fps: Annotated[int, typer.Option("--fps", help="Images par seconde.")] = 60,
    max_gemini_rpm: Annotated[int, typer.Option("--max-gemini-rpm", help="Requetes Gemini par minute (toutes cles).")] = 10,
    force: Annotated[bool, typer.Option("--force", help="Recalculer toutes les etapes.")] = False,
    no_series_memory: Annotated[bool, typer.Option("--no-series-memory", help="Ne pas reutiliser la fiche des personnages des episodes precedents.")] = False,
    panels: Annotated[str, typer.Option("--panels", help="Cases montees : figures (defaut, personnages detectes seuls) ou slicer (cases entieres).")] = "figures",
    figure_margin: Annotated[float, typer.Option("--figure-margin", help="Marge autour des personnages (part de leur taille, defaut 0).")] = 0.0,
    figure_bubbles: Annotated[str, typer.Option("--figure-bubbles", help="cut (defaut : zone jaune exacte) ou whole (agrandie aux bulles touchees).")] = "cut",
    figures_separate: Annotated[bool, typer.Option("--figures-separate", help="Une image par personnage au lieu d'une image par case (personnages d'une meme case regroupes par defaut).")] = False,
    no_upscale: Annotated[bool, typer.Option("--no-upscale", help="Garder les personnages a leur taille d'origine (defaut : agrandis par IA a ~90 % du cadre).")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Logs DEBUG.")] = False,
) -> None:
    """Enchaine scraping, decoupe, analyse Gemini, voix off Kokoro, brouillon CapCut et apercu."""
    from src.utils.gemini_manager import GeminiManager, GeminiManagerError

    _setup(verbose)
    out_dir = out or (PROJECT_ROOT / "output" / slug_from_url(url))
    options = _pipeline_options(
        language=language, voice=voice, speed=speed, model=model, thinking_budget=thinking_budget,
        script_ai=script_ai, claude_model=claude_model,
        preview_seconds=preview_seconds, no_preview=no_preview, no_capcut=no_capcut, capcut_dir=capcut_dir,
        project_name=project_name, bgm=bgm, bgm_dir=bgm_dir, no_bgm=no_bgm, sfx_dir=sfx_dir, no_sfx=no_sfx,
        cta=cta, no_cta=no_cta, fps=fps, force=force, sentence_gap=sentence_gap, padding=padding, batch_size=batch_size,
        gemini_batch_delay=gemini_batch_delay, keyframe_workers=keyframe_workers, multi_call=multi_call,
        dynamics=dynamics, transition_compensation=transition_compensation,
        thumbnail=thumbnail, thumbnail_backend=thumbnail_backend, video_format=video_format,
        no_series_memory=no_series_memory, panels=panels, figure_margin=figure_margin, figure_bubbles=figure_bubbles,
        figures_separate=figures_separate, no_upscale=no_upscale,
    )
    try:
        manager = GeminiManager(preferred_model=model, max_rpm=max_gemini_rpm)
        result = run_pipeline(url, out_dir, options, manager=manager)
    except GeminiManagerError as exc:
        print(_ascii(f"FAILED: {exc}"))
        raise typer.Exit(code=1)
    except Exception as exc:  # noqa: BLE001 - la CLI doit afficher toute erreur
        logging.getLogger(__name__).exception("Pipeline en echec")
        print(_ascii(f"FAILED: {type(exc).__name__}: {exc}"))
        raise typer.Exit(code=1)
    print(format_result(result))


@app.command()
def batch(
    url: Annotated[Optional[str], typer.Argument(help="URL de la serie (liste Webtoons, page de serie Asura) ou de n'importe quel episode.")] = None,
    url_list: Annotated[Optional[Path], typer.Option("--url-list", help="Fichier texte : une URL de chapitre par ligne.")] = None,
    start_chapter: Annotated[Optional[int], typer.Option("--start-chapter", help="Premier episode_no de la plage.")] = None,
    end_chapter: Annotated[Optional[int], typer.Option("--end-chapter", help="Dernier episode_no de la plage.")] = None,
    max_chapters: Annotated[int, typer.Option("--max-chapters", help="Chapitres analyses simultanement par Gemini.")] = 5,
    max_scrape_workers: Annotated[Optional[int], typer.Option("--max-scrape-workers", help="Telechargements + decoupes simultanes (defaut : --max-chapters).")] = None,
    max_gemini_rpm: Annotated[int, typer.Option("--max-gemini-rpm", help="Requetes Gemini par minute, toutes cles et chapitres confondus.")] = 10,
    max_tts_workers: Annotated[int, typer.Option("--max-tts-workers", help="Syntheses Kokoro simultanees (CPU) ; optimum mesure = 2 sur 16 coeurs, 4 regresse.")] = 2,
    max_render_workers: Annotated[int, typer.Option("--max-render-workers", help="Montages + rendus ffmpeg simultanes (CPU).")] = 1,
    status_file: Annotated[Optional[Path], typer.Option("--status-file", help="Fichier de suivi (defaut batch_status.json a la racine).")] = None,
    out_root: Annotated[Optional[Path], typer.Option("--out-root", help="Dossier racine des sorties (defaut output/).")] = None,
    no_retry_failed: Annotated[bool, typer.Option("--no-retry-failed", help="Ne pas reprendre les chapitres en echec.")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Lister les chapitres cibles sans rien traiter.")] = False,
    no_series_order: Annotated[bool, typer.Option("--no-series-order", help="Analyser les episodes d'une serie en parallele plutot que dans l'ordre (plus rapide, mais le chapitre N n'herite plus des noms du N-1).")] = False,
    language: Annotated[str, typer.Option("--language", help="Langue de la narration et de la voix.")] = DEFAULT_NARRATION_LANGUAGE,
    voice: Annotated[Optional[str], typer.Option("--voice", help="Voix Kokoro (defaut am_fenrir,am_michael en anglais, melange des deux ; voir la commande voices).")] = None,
    speed: Annotated[float, typer.Option("--speed", help="Vitesse de la voix.")] = 1.0,
    sentence_gap: Annotated[Optional[float], typer.Option("--sentence-gap", help="Silence ajoute entre deux phrases (defaut 0 : pauses laissees a Kokoro).")] = None,
    padding: Annotated[Optional[float], typer.Option("--padding", help="Silence ajoute en fin de scene (defaut 0 : silence naturel de Kokoro).")] = None,
    script_ai: Annotated[str, typer.Option("--script-ai", help="IA qui ecrit le script : claude (defaut, CLI local et abonnement ; Gemini en repli si Claude echoue) ou gemini.")] = "claude",
    claude_model: Annotated[Optional[str], typer.Option("--claude-model", help="Modele Claude du script (defaut claude-opus-5-5).")] = None,
    model: Annotated[Optional[str], typer.Option("--model", help="Modele Gemini prefere, pour --script-ai gemini ou le repli (la cascade de secours suit).")] = None,
    thinking_budget: Annotated[Optional[int], typer.Option("--thinking-budget", help="Budget de reflexion Gemini.")] = None,
    batch_size: Annotated[Optional[int], typer.Option("--batch-size", help="Images par lot Gemini (10-15, defaut 12) ; 15 = ~20 %% d'appels en moins.")] = None,
    gemini_batch_delay: Annotated[Optional[float], typer.Option("--gemini-batch-delay", help="Pause forcee entre deux envois (defaut 2,5 s ; 0 est sur avec --max-gemini-rpm).")] = None,
    keyframe_workers: Annotated[Optional[int], typer.Option("--keyframe-workers", help="Appels 'cases cles' en parallele (defaut 4, mode deux etapes).")] = None,
    multi_call: Annotated[bool, typer.Option("--multi-call", help="Mode historique en deux etapes (~18 appels) au lieu de la requete unique.")] = False,
    video_format: Annotated[str, typer.Option("--format", help="Format de sortie : LONG (16:9, defaut) ou SHORT (9:16 TikTok : recadrage 9:16, cadence 1,2 s, voix x1,20, outro).")] = "LONG",
    dynamics: Annotated[str, typer.Option("--dynamics", help="Dynamisme du montage : none (sobre), subtle (sous-titres animes) ou punchy (defaut : + transitions et effets).")] = "punchy",
    transition_compensation: Annotated[str, typer.Option("--transition-compensation", help="Recouvrement des transitions CapCut : none (defaut) ou shift (a verifier dans l'editeur).")] = "none",
    thumbnail: Annotated[bool, typer.Option("--thumbnail", help="Generer aussi la miniature YouTube (1 appel texte + 1 generation d'image).")] = False,
    thumbnail_backend: Annotated[Optional[str], typer.Option("--thumbnail-backend", help="Backend d'images : gemini (defaut), stability ou local.")] = None,
    preview_seconds: Annotated[float, typer.Option("--preview-seconds", help="Duree de l'apercu (0 = complet).")] = DEFAULT_PREVIEW_SECONDS,
    no_preview: Annotated[bool, typer.Option("--no-preview", help="Ne pas rendre les apercus MP4.")] = False,
    no_capcut: Annotated[bool, typer.Option("--no-capcut", help="Ne pas generer les brouillons CapCut.")] = False,
    capcut_dir: Annotated[Optional[Path], typer.Option("--capcut-dir", help="Dossier des projets CapCut (copie des brouillons).")] = None,
    bgm: Annotated[Optional[Path], typer.Option("--bgm", help="Musique de fond unique (WAV/MP3), bouclee, -22 dB.")] = None,
    bgm_dir: Annotated[Optional[Path], typer.Option("--bgm-dir", help="Dossier des musiques par ambiance.")] = None,
    no_bgm: Annotated[bool, typer.Option("--no-bgm", help="Aucune musique de fond.")] = False,
    sfx_dir: Annotated[Optional[Path], typer.Option("--sfx-dir", help="Dossier des bruitages.")] = None,
    no_sfx: Annotated[bool, typer.Option("--no-sfx", help="Aucun bruitage.")] = False,
    cta: Annotated[Optional[str], typer.Option("--cta", help="Phrase d'appel a l'abonnement.")] = None,
    no_cta: Annotated[bool, typer.Option("--no-cta", help="Aucun appel a l'abonnement.")] = False,
    fps: Annotated[int, typer.Option("--fps", help="Images par seconde.")] = 60,
    redo: Annotated[Optional[str], typer.Option("--redo", help="Recalculer a partir de cette etape : analyze, tts ou montage (les precedentes sont reutilisees, donc aucun quota Gemini pour --redo tts).")] = None,
    compile_video: Annotated[bool, typer.Option("--compile", help="Une seule video pour tous les chapitres : compilation (projet CapCut + extrait de 2 min), puis suppression des dossiers de chapitres.")] = False,
    keep_chapters: Annotated[bool, typer.Option("--keep-chapters", help="Avec --compile : garder les dossiers de chapitres.")] = False,
    force: Annotated[bool, typer.Option("--force", help="Recalculer toutes les etapes, chapitres deja faits compris.")] = False,
    no_series_memory: Annotated[bool, typer.Option("--no-series-memory", help="Ne pas reutiliser la fiche des personnages des episodes precedents.")] = False,
    panels: Annotated[str, typer.Option("--panels", help="Cases montees : figures (defaut, personnages detectes seuls) ou slicer (cases entieres).")] = "figures",
    figure_margin: Annotated[float, typer.Option("--figure-margin", help="Marge autour des personnages (part de leur taille, defaut 0).")] = 0.0,
    figure_bubbles: Annotated[str, typer.Option("--figure-bubbles", help="cut (defaut : zone jaune exacte) ou whole (agrandie aux bulles touchees).")] = "cut",
    figures_separate: Annotated[bool, typer.Option("--figures-separate", help="Une image par personnage au lieu d'une image par case (personnages d'une meme case regroupes par defaut).")] = False,
    no_upscale: Annotated[bool, typer.Option("--no-upscale", help="Garder les personnages a leur taille d'origine (defaut : agrandis par IA a ~90 % du cadre).")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Logs DEBUG.")] = False,
) -> None:
    """Traite une serie (plage d'episodes ou liste d'URL) en parallele, avec suivi batch_status.json."""
    from src.modules.batch_processor import BatchError, BatchOptions, format_report, resolve_chapter_urls, run_batch
    from src.utils.gemini_manager import GeminiManager, GeminiManagerError

    from src.pipeline import STAGE_RANKS

    _setup(verbose)
    if redo is not None and redo not in STAGE_RANKS:
        print(_ascii(f"FAILED: --redo doit valoir {', '.join(STAGE_RANKS)}"))
        raise typer.Exit(code=2)
    try:
        urls = resolve_chapter_urls(url, start_chapter=start_chapter, end_chapter=end_chapter, url_list=url_list)
    except BatchError as exc:
        print(_ascii(f"FAILED: {exc}"))
        raise typer.Exit(code=2)
    print(_ascii(f"{len(urls)} chapitre(s) cible(s) :"))
    for chapter_url in urls:
        print(_ascii(f"  - {chapter_url}"))
    if dry_run:
        return
    # Compilation : ni apercu ni brouillon CapCut par chapitre (les dossiers vont disparaitre,
    # un brouillon par chapitre pointerait sur des fichiers supprimes) ; seule la compilation en a.
    options = _pipeline_options(
        language=language, voice=voice, speed=speed, model=model, thinking_budget=thinking_budget,
        script_ai=script_ai, claude_model=claude_model,
        preview_seconds=preview_seconds, no_preview=no_preview or compile_video, no_capcut=no_capcut or compile_video,
        capcut_dir=capcut_dir, project_name=None, bgm=bgm, bgm_dir=bgm_dir, no_bgm=no_bgm, sfx_dir=sfx_dir, no_sfx=no_sfx,
        cta=cta, no_cta=no_cta, fps=fps, force=force, redo=redo,
        sentence_gap=sentence_gap, padding=padding, batch_size=batch_size,
        gemini_batch_delay=gemini_batch_delay, keyframe_workers=keyframe_workers, multi_call=multi_call,
        dynamics=dynamics, transition_compensation=transition_compensation,
        thumbnail=thumbnail, thumbnail_backend=thumbnail_backend, video_format=video_format,
        no_series_memory=no_series_memory, panels=panels, figure_margin=figure_margin, figure_bubbles=figure_bubbles,
        figures_separate=figures_separate, no_upscale=no_upscale,
    )
    batch_options = BatchOptions(
        max_chapters=max_chapters, max_gemini_rpm=max_gemini_rpm, max_tts_workers=max_tts_workers,
        max_render_workers=max_render_workers, max_scrape_workers=max_scrape_workers, retry_failed=not no_retry_failed,
        series_order=not no_series_order,
        **({"status_file": status_file} if status_file else {}), **({"out_root": out_root} if out_root else {}),
    )
    try:
        manager = GeminiManager(preferred_model=model, max_rpm=max_gemini_rpm)
        report = run_batch(urls, options, batch_options, manager=manager)
    except GeminiManagerError as exc:
        print(_ascii(f"FAILED: {exc}"))
        raise typer.Exit(code=1)
    print(format_report(report))
    if report.errors and not report.results:
        raise typer.Exit(code=1)
    if compile_video:
        if report.errors:
            print(_ascii(f"Compilation non faite : {len(report.errors)} chapitre(s) en echec (relancer la meme commande)."))
            raise typer.Exit(code=1)
        # Tous les chapitres demandes, y compris ceux deja faits lors d'un lot precedent.
        folders = [batch_options.out_root / slug_from_url(u) for u in urls]
        missing = [f.name for f in folders if not (f / "timeline.json").is_file()]
        if missing:
            print(_ascii(f"Compilation non faite : chapitre(s) sans montage {missing} (deja compiles et supprimes ? --force pour les refaire)."))
            raise typer.Exit(code=1)
        _compile(folders, out=None, name=None, gap=0.6, preview_seconds=DEFAULT_COMPILATION_PREVIEW_S,
                 make_capcut=not no_capcut, capcut_dir=capcut_dir, delete_chapters=not keep_chapters,
                 root=batch_options.out_root)


def _compile(folders, *, out, name, gap, preview_seconds, make_capcut, capcut_dir, delete_chapters, pattern=None, root=None) -> None:
    """Compilation commune a ``merge`` et ``batch --compile`` (affiche le resultat, sort en code 1 sur erreur)."""
    from src.pipeline import build_compilation, chapter_folders, compilation_dirname

    try:
        chapters = chapter_folders(folders or [], pattern)
    except ValueError as exc:
        print(_ascii(f"FAILED: {exc}"))
        raise typer.Exit(code=2)
    print(_ascii(f"{len(chapters)} chapitre(s) a fusionner :"))
    for folder in chapters:
        print(_ascii(f"  - {folder}"))
    target = out or (Path(root or PROJECT_ROOT / "output") / compilation_dirname(chapters))
    try:
        result = build_compilation(
            chapters, target, name=name, gap_s=gap, preview_seconds=preview_seconds,
            make_capcut=make_capcut, capcut_dir=str(capcut_dir) if capcut_dir else None,
            delete_chapters=delete_chapters,
        )
    except Exception as exc:  # noqa: BLE001 - la CLI doit afficher toute erreur
        logging.getLogger(__name__).exception("Compilation en echec")
        print(_ascii(f"FAILED: {type(exc).__name__}: {exc}"))
        raise typer.Exit(code=1)
    print(format_result(result))
    if delete_chapters:
        print(_ascii(f"Dossiers de chapitres supprimes ({len(chapters)}) : la compilation est autonome."))


@app.command()
def merge(
    folders: Annotated[Optional[list[Path]], typer.Argument(help="Dossiers de chapitres deja traites (contenant timeline.json).")] = None,
    pattern: Annotated[Optional[str], typer.Option("--pattern", help="Motif glob depuis la racine, ex. \"output/ma-serie_ep*\".")] = None,
    out: Annotated[Optional[Path], typer.Option("--out", "-o", help="Dossier de la compilation (defaut output/<serie>_compilation_ch<premier>-<dernier>).")] = None,
    name: Annotated[Optional[str], typer.Option("--name", help="Nom du projet CapCut.")] = None,
    gap: Annotated[float, typer.Option("--gap", help="Silence entre deux chapitres (secondes).")] = 0.6,
    preview_seconds: Annotated[float, typer.Option("--preview-seconds", help="Duree de l'extrait MP4 (defaut 120 ; 0 = video complete, -1 = aucun). La video complete se rend dans CapCut.")] = DEFAULT_COMPILATION_PREVIEW_S,
    keep_chapters: Annotated[bool, typer.Option("--keep-chapters", help="Garder les dossiers de chapitres (supprimes par defaut, la compilation etant autonome).")] = False,
    no_capcut: Annotated[bool, typer.Option("--no-capcut", help="Ne pas generer le brouillon CapCut.")] = False,
    capcut_dir: Annotated[Optional[Path], typer.Option("--capcut-dir", help="Dossier des projets CapCut.")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Fusionne plusieurs chapitres deja traites en UN SEUL projet CapCut (compilation autonome).

    Les dossiers de chapitres sont ensuite supprimes (--keep-chapters pour les garder).
    """
    _setup(verbose)
    _compile(folders, pattern=pattern, out=out, name=name, gap=gap,
             preview_seconds=None if preview_seconds < 0 else preview_seconds, make_capcut=not no_capcut,
             capcut_dir=capcut_dir, delete_chapters=not keep_chapters)


@app.command()
def voices(
    out: Annotated[Optional[Path], typer.Option("--out", "-o", help="Dossier des extraits (defaut output/voices).")] = None,
    voice_list: Annotated[Optional[str], typer.Option("--voices", help="Voix a comparer, separees par des virgules (defaut : les 28 voix anglaises).")] = None,
    text: Annotated[Optional[str], typer.Option("--text", help="Extrait a lire (defaut : une accroche de recap).")] = None,
    language: Annotated[str, typer.Option("--language", help="Langue du pipeline Kokoro.")] = DEFAULT_NARRATION_LANGUAGE,
    speed: Annotated[float, typer.Option("--speed", help="Vitesse de lecture.")] = 1.0,
    workers: Annotated[int, typer.Option("--workers", help="Syntheses simultanees (mesure le gain du parallelisme CPU).")] = 1,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Synthetise le meme extrait avec toutes les voix Kokoro et ecrit un rapport HTML comparatif."""
    from src.modules.voice_lab import (
        DEFAULT_SAMPLE_TEXT,
        ENGLISH_VOICES,
        build_voice_report,
        compare_voices,
        format_voice_table,
    )

    _setup(verbose)
    out_dir = out or (PROJECT_ROOT / "output" / "voices")
    wanted = [v.strip() for v in voice_list.split(",") if v.strip()] if voice_list else list(ENGLISH_VOICES)
    sample_text = text or DEFAULT_SAMPLE_TEXT
    try:
        samples = compare_voices(
            wanted, out_dir, text=sample_text, language=language, speed=speed, workers=max(1, workers),
        )
    except Exception as exc:  # noqa: BLE001 - la CLI doit afficher toute erreur
        logging.getLogger(__name__).exception("Comparatif des voix en echec")
        print(_ascii(f"FAILED: {type(exc).__name__}: {exc}"))
        raise typer.Exit(code=1)
    report = build_voice_report(samples, out_dir / "voices.html", text=sample_text, speed=speed)
    print(format_voice_table(samples))
    print(_ascii(f"\n{len(samples)} extrait(s) dans {out_dir}\nRapport : {report}"))


@app.command()
def stats(
    status_file: Annotated[Optional[Path], typer.Option("--status-file", help="Fichier de suivi (defaut batch_status.json a la racine).")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Affiche les temps mesures d'un lot : duree par etape, par chapitre et par seconde de video."""
    from src.modules.batch_processor import DEFAULT_STATUS_FILE, BatchStatus, format_status

    _setup(verbose)
    path = status_file or DEFAULT_STATUS_FILE
    if not Path(path).is_file():
        print(_ascii(f"Aucun fichier de suivi : {path}"))
        raise typer.Exit(code=1)
    print(format_status(BatchStatus(path)))


@app.command()
def preview(
    project: Annotated[Path, typer.Argument(help="Dossier d'un chapitre deja traite (contient timeline.json).")],
    seconds: Annotated[float, typer.Option("--seconds", help="Duree de l'apercu (0 = complet).")] = DEFAULT_PREVIEW_SECONDS,
    out: Annotated[Optional[Path], typer.Option("--out", help="Fichier MP4 de sortie.")] = None,
    fps: Annotated[Optional[int], typer.Option("--fps", help="Images par seconde (defaut : timeline).")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Re-rend l'apercu MP4 a partir de timeline.json."""
    from src.modules.preview_renderer import PreviewRenderer
    from src.modules.timeline_builder import load_timeline

    _setup(verbose)
    timeline = load_timeline(project / "timeline.json")
    label = f"{int(seconds)}s" if seconds > 0 else "full"
    target = out or project / f"preview_{label}.mp4"
    path = PreviewRenderer(timeline, fps=fps).render(target, max_duration_s=seconds if seconds > 0 else None)
    print(_ascii(f"Apercu : {path}"))


@app.command()
def capcut(
    project: Annotated[Path, typer.Argument(help="Dossier d'un chapitre deja traite (contient timeline.json).")],
    name: Annotated[Optional[str], typer.Option("--name", help="Nom du projet CapCut.")] = None,
    capcut_dir: Annotated[Optional[Path], typer.Option("--capcut-dir", help="Dossier des projets CapCut.")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Regenere le brouillon CapCut a partir de timeline.json."""
    from src.modules.capcut_builder import build_capcut_draft, copy_draft, detect_capcut_drafts_dir
    from src.modules.timeline_builder import load_timeline

    _setup(verbose)
    timeline = load_timeline(project / "timeline.json")
    draft = build_capcut_draft(timeline, project / "capcut", name or project.name)
    print(_ascii(f"Brouillon : {draft}"))
    target = capcut_dir or detect_capcut_drafts_dir()
    if target is not None and target.is_dir():
        print(_ascii(f"Copie     : {copy_draft(draft, target)}"))
    else:
        print("CapCut    : dossier des projets introuvable, copier le brouillon a la main")


@app.command()
def thumbnail(
    project: Annotated[Path, typer.Argument(help="Dossier d'un chapitre deja analyse (contient scenes.json).")],
    backend: Annotated[Optional[str], typer.Option("--backend", help="Backend d'images : gemini (defaut), stability ou local.")] = None,
    text: Annotated[Optional[str], typer.Option("--text", help="Imposer l'accroche (1 a 3 mots) au lieu de la demander au modele.")] = None,
    arrow: Annotated[Optional[str], typer.Option("--arrow", help="Cote de la fleche : left ou right (defaut : choix du modele).")] = None,
    image: Annotated[Optional[Path], typer.Option("--image", help="Reutiliser cette illustration au lieu d'en generer une (aucun quota consomme).")] = None,
    white: Annotated[bool, typer.Option("--white", help="Accroche en blanc au lieu du jaune.")] = False,
    arrow_asset: Annotated[Optional[Path], typer.Option("--arrow-asset", help="PNG de la fleche (defaut config/thumbnail/arrow_yellow.png).")] = None,
    font: Annotated[Optional[Path], typer.Option("--font", help="Police d'affichage (defaut : Bangers puis Impact).")] = None,
    out: Annotated[Optional[Path], typer.Option("--out", "-o", help="Fichier de sortie (defaut <projet>/thumbnail.jpg).")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Genere la miniature YouTube d'un chapitre deja analyse (analyse, image, compositing)."""
    from src.modules.analyzer import load_analysis
    from src.modules.thumbnail.compositor import (
        TEXT_FILL_WHITE,
        TEXT_FILL_YELLOW,
        ThumbnailCompositor,
        build_thumbnail,
    )
    from src.modules.thumbnail.pipeline import THUMBNAIL_NAME, ThumbnailStage

    _setup(verbose)
    if arrow is not None and arrow not in ("left", "right"):
        print(_ascii("FAILED: --arrow doit valoir left ou right"))
        raise typer.Exit(code=2)
    fill = TEXT_FILL_WHITE if white else TEXT_FILL_YELLOW
    target = out or project / THUMBNAIL_NAME

    # Accroche et illustration fournies : compositing seul, aucune requete, aucun quota.
    if image is not None and text:
        try:
            path = build_thumbnail(
                image, text, arrow or "right", out_path=target,
                fill=fill, arrow_path=arrow_asset, font_path=font,
            )
        except Exception as exc:  # noqa: BLE001 - la CLI doit afficher toute erreur
            print(_ascii(f"FAILED: {type(exc).__name__}: {exc}"))
            raise typer.Exit(code=1)
        print(_ascii(f"Miniature : {path}"))
        return

    scenes_json = project / "scenes.json"
    if not scenes_json.is_file():
        print(_ascii(f"FAILED: {scenes_json} introuvable (lancer d'abord l'analyse du chapitre)"))
        raise typer.Exit(code=2)
    stage = ThumbnailStage(
        backend=_thumbnail_backend(backend) if backend else None,
        compositor=ThumbnailCompositor(fill=fill, arrow_path=arrow_asset, font_path=font),
    )
    try:
        analysis = load_analysis(scenes_json)
        brief, _ = stage.design(analysis, project)
        if text or arrow:
            brief = brief.model_copy(update={
                **({"hook_text": text.upper()[:18]} if text else {}),
                **({"arrow_position": arrow} if arrow else {}),
            })
        base = Path(image) if image is not None else stage.generate(brief, project)
        path = stage.compose(brief, base, project, target)
    except Exception as exc:  # noqa: BLE001 - la CLI doit afficher toute erreur
        print(_ascii(f"FAILED: {type(exc).__name__}: {exc}"))
        raise typer.Exit(code=1)
    print(_ascii(
        f"Accroche  : {brief.hook_text}\n"
        f"Fleche    : {brief.arrow_position} (sujet {brief.subject_position})\n"
        f"Scene     : {brief.scene_description[:160]}\n"
        f"Image     : {base}\n"
        f"Miniature : {path}"
    ))


def _thumbnail_backend(name: str):
    from src.modules.thumbnail.image_backends import ImageBackendError, resolve_backend

    try:
        return resolve_backend(name)
    except ImageBackendError as exc:
        print(_ascii(f"FAILED: {exc}"))
        raise typer.Exit(code=2)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
