"""Enchaînement des trois étapes de la miniature, avec frontière d'erreur stricte.

Règle de conception : **une miniature ratée ne doit jamais empêcher la vidéo d'exister.**
L'API d'images peut être en panne, sans quota, refuser un prompt, ou la police manquer :
:meth:`ThumbnailStage.run` absorbe tout, journalise, et renvoie ``None``. Seul
:meth:`ThumbnailStage.build` propage, pour la commande CLI dédiée où l'utilisateur
attend une erreur explicite.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from src.models.scene import ChapterAnalysis
from src.models.thumbnail import ThumbnailBrief
from src.modules.thumbnail.analyzer import ThumbnailAnalysisError, ThumbnailAnalyzer
from src.modules.thumbnail.compositor import (
    TEXT_FILL_YELLOW,
    ThumbnailCompositor,
    ThumbnailCompositorError,
)
from src.modules.thumbnail.image_backends import (
    ImageBackend,
    ImageBackendError,
    build_image_prompt,
    build_negative_prompt,
    resolve_backend,
    save_image,
)
from src.utils.gemini_manager import GeminiManager, GeminiManagerError

logger = logging.getLogger(__name__)

#: Nom des fichiers produits dans le dossier du chapitre.
BRIEF_NAME: str = "thumbnail_brief.json"
BASE_IMAGE_NAME: str = "thumbnail_base.png"
THUMBNAIL_NAME: str = "thumbnail.jpg"

#: Erreurs métier attendues : elles sont dégradées en avertissement par :meth:`run`.
EXPECTED_ERRORS: tuple[type[Exception], ...] = (
    ThumbnailAnalysisError,
    ImageBackendError,
    ThumbnailCompositorError,
    GeminiManagerError,
    OSError,
)


@dataclass
class ThumbnailResult:
    """Fichiers produits et durées des trois étapes."""

    brief: ThumbnailBrief
    brief_json: Path
    base_image: Path
    thumbnail: Path
    seconds: dict[str, float]

    @property
    def total_s(self) -> float:
        return sum(self.seconds.values())


class ThumbnailStage:
    """Analyse, génère et compose la miniature d'un chapitre.

    Args:
        manager: gestionnaire Gemini partagé pour l'analyse (texte).
        backend: générateur d'images ; sinon résolu depuis ``THUMBNAIL_IMAGE_BACKEND``.
        analyzer: analyseur injectable (tests).
        compositor: compositeur injectable (tests).
        keep_base: conserver l'illustration brute à côté de la miniature.
    """

    def __init__(
        self,
        manager: GeminiManager | None = None,
        *,
        backend: ImageBackend | None = None,
        analyzer: ThumbnailAnalyzer | None = None,
        compositor: ThumbnailCompositor | None = None,
        keep_base: bool = True,
    ) -> None:
        self._manager = manager
        self._backend = backend
        self.analyzer = analyzer or ThumbnailAnalyzer(manager)
        self.compositor = compositor or ThumbnailCompositor(fill=TEXT_FILL_YELLOW)
        self.keep_base = keep_base

    @property
    def backend(self) -> ImageBackend:
        if self._backend is None:
            self._backend = resolve_backend()
        return self._backend

    # -- etapes, utilisables separement ---------------------------------------------------
    def design(self, analysis: ChapterAnalysis, out_dir: str | Path) -> tuple[ThumbnailBrief, Path]:
        """Étape 1 : consigne de miniature, écrite en JSON pour la traçabilité."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        brief = self.analyzer.design(analysis)
        brief_json = out_dir / BRIEF_NAME
        brief_json.write_text(brief.model_dump_json(indent=2), encoding="utf-8")
        return brief, brief_json

    def generate(self, brief: ThumbnailBrief, out_dir: str | Path) -> Path:
        """Étape 2 : illustration de base 16:9."""
        data = self.backend.generate(build_image_prompt(brief), negative=build_negative_prompt(brief))
        return save_image(data, Path(out_dir) / BASE_IMAGE_NAME)

    def compose(
        self, brief: ThumbnailBrief, base_image: str | Path, out_dir: str | Path, out_path: str | Path | None = None
    ) -> Path:
        """Étape 3 : compositing (saturation, flèche, accroche)."""
        return self.compositor.build(
            base_image, brief.hook_text, brief.arrow_position, out_path or Path(out_dir) / THUMBNAIL_NAME
        )

    def build(
        self,
        analysis: ChapterAnalysis,
        out_dir: str | Path,
        *,
        base_image: str | Path | None = None,
        out_path: str | Path | None = None,
    ) -> ThumbnailResult:
        """Exécute les trois étapes et renvoie les chemins produits.

        Args:
            analysis: chapitre analysé.
            out_dir: dossier de sortie.
            base_image: illustration déjà disponible ; l'étape 2 est alors **sautée**
                (aucun quota d'image consommé).
            out_path: fichier de la miniature (défaut ``<out_dir>/thumbnail.jpg``).

        Raises:
            ThumbnailAnalysisError, ImageBackendError, ThumbnailCompositorError: selon l'étape.
        """
        out_dir = Path(out_dir)
        seconds: dict[str, float] = {}

        started = time.perf_counter()
        brief, brief_json = self.design(analysis, out_dir)
        seconds["analyse"] = time.perf_counter() - started

        started = time.perf_counter()
        if base_image is None:
            base = self.generate(brief, out_dir)
        else:
            base = Path(base_image)
            logger.info("Illustration reutilisee : %s (aucune generation)", base)
        seconds["image"] = time.perf_counter() - started

        started = time.perf_counter()
        thumbnail = self.compose(brief, base, out_dir, out_path)
        seconds["compositing"] = time.perf_counter() - started

        if not self.keep_base and base_image is None:
            base.unlink(missing_ok=True)
        result = ThumbnailResult(
            brief=brief, brief_json=brief_json, base_image=base, thumbnail=thumbnail, seconds=seconds
        )
        logger.info(
            "Miniature terminee en %.0fs (analyse %.0fs, image %.0fs, compositing %.1fs) : %s",
            result.total_s, seconds["analyse"], seconds["image"], seconds["compositing"], thumbnail,
        )
        return result

    def run(self, analysis: ChapterAnalysis, out_dir: str | Path) -> ThumbnailResult | None:
        """Comme :meth:`build`, mais **n'échoue jamais** : renvoie ``None`` et journalise.

        C'est la porte d'entrée du pipeline vidéo : la miniature est un bonus, pas une
        dépendance. Une erreur inattendue est aussi absorbée, avec sa pile en DEBUG.
        """
        try:
            return self.build(analysis, out_dir)
        except EXPECTED_ERRORS as exc:
            logger.warning("Miniature ignoree (%s) : %s", type(exc).__name__, exc)
        except Exception as exc:  # noqa: BLE001 - aucune miniature ne doit casser la video
            logger.warning("Miniature ignoree (erreur inattendue %s) : %s", type(exc).__name__, exc)
            logger.debug("Detail de l'echec de la miniature", exc_info=True)
        return None


__all__ = [
    "BRIEF_NAME",
    "BASE_IMAGE_NAME",
    "THUMBNAIL_NAME",
    "EXPECTED_ERRORS",
    "ThumbnailResult",
    "ThumbnailStage",
]
