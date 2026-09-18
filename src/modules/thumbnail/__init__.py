"""Module 7 — Générateur automatique de miniatures YouTube.

Trois étapes séparées, chacune utilisable seule :

1. :mod:`~src.modules.thumbnail.analyzer` — le modèle choisit une scène visuellement
   contrastée, une accroche de 1 à 3 mots et le côté de la flèche (un appel de texte) ;
2. :mod:`~src.modules.thumbnail.image_backends` — génération de l'illustration 16:9
   (Gemini par défaut, Stability ou serveur local en option) ;
3. :mod:`~src.modules.thumbnail.compositor` — compositing Pillow : saturation +15 %,
   flèche jaune, accroche contournée et inclinée.

:class:`~src.modules.thumbnail.pipeline.ThumbnailStage` les enchaîne. Sa méthode ``run``
**n'échoue jamais** : une miniature ratée ne doit pas empêcher la vidéo d'être produite.
"""

from src.modules.thumbnail.analyzer import ThumbnailAnalysisError, ThumbnailAnalyzer
from src.modules.thumbnail.compositor import (
    ThumbnailCompositor,
    ThumbnailCompositorError,
    build_thumbnail,
)
from src.modules.thumbnail.image_backends import (
    GeminiImageBackend,
    ImageBackend,
    ImageBackendError,
    LocalWebuiBackend,
    StabilityBackend,
    resolve_backend,
)
from src.modules.thumbnail.pipeline import ThumbnailResult, ThumbnailStage

__all__ = [
    "ThumbnailAnalyzer",
    "ThumbnailAnalysisError",
    "ThumbnailCompositor",
    "ThumbnailCompositorError",
    "build_thumbnail",
    "ImageBackend",
    "ImageBackendError",
    "GeminiImageBackend",
    "StabilityBackend",
    "LocalWebuiBackend",
    "resolve_backend",
    "ThumbnailStage",
    "ThumbnailResult",
]
