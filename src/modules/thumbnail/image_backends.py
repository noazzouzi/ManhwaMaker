"""Étape 2 de la miniature : générer l'illustration de base (16:9).

Le générateur est **enfichable** (:class:`ImageBackend`) pour ne pas lier le projet à un
fournisseur. Trois implémentations :

- :class:`GeminiImageBackend` (**défaut**) : réutilise les clés déjà configurées via un
  :class:`~src.utils.gemini_manager.GeminiManager` dédié, doté de sa propre cascade de
  modèles d'images. Aucun abonnement supplémentaire.
- :class:`StabilityBackend` : API REST Stability (SDXL / SD3), clé ``STABILITY_API_KEY``.
- :class:`LocalWebuiBackend` : serveur local compatible AUTOMATIC1111 (``LOCAL_SD_URL``).

Le backend actif est choisi par ``THUMBNAIL_IMAGE_BACKEND`` (voir :func:`resolve_backend`).

⚠ Seul le backend Gemini a été exercé sur cette installation ; les deux autres sont
fournis pour être branchés et restent à valider avec une vraie clé / un vrai serveur.
"""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from src.models.thumbnail import ThumbnailBrief
from src.utils.gemini_manager import GeminiManager

logger = logging.getLogger(__name__)

#: Cascade de modèles d'images, du plus soigné au plus économe. Vérifiée sur ce compte
#: le 2026-09-12 (``client.models.list()``).
DEFAULT_IMAGE_MODEL_CASCADE: tuple[str, ...] = (
    "gemini-3-pro-image",
    "gemini-3.1-flash-image",
    "gemini-2.5-flash-image",
    "gemini-3.1-flash-lite-image",
)
#: Style imposé à toute miniature, quel que soit le backend.
STYLE_PROMPT: str = "high quality anime, manhwa, ultra-detailed, dramatic lighting"
#: Éléments toujours exclus : la miniature reçoit son texte au compositing, pas au rendu.
BASE_NEGATIVE: str = (
    "text, letters, words, captions, subtitles, watermark, logo, signature, speech bubbles, "
    "ui, frame borders, collage, deformed hands, extra limbs, extra fingers, blurry, low resolution"
)
DEFAULT_ASPECT_RATIO: str = "16:9"
#: Variable d'environnement choisissant le backend.
BACKEND_ENV_VAR: str = "THUMBNAIL_IMAGE_BACKEND"
STABILITY_KEY_ENV_VAR: str = "STABILITY_API_KEY"
STABILITY_URL_ENV_VAR: str = "STABILITY_URL"
LOCAL_URL_ENV_VAR: str = "LOCAL_SD_URL"
DEFAULT_STABILITY_URL: str = "https://api.stability.ai/v2beta/stable-image/generate/core"
DEFAULT_LOCAL_URL: str = "http://127.0.0.1:7860"
HTTP_TIMEOUT_S: float = 180.0


class ImageBackendError(RuntimeError):
    """La génération d'image a échoué (quota, réseau, réponse sans image...)."""


@runtime_checkable
class ImageBackend(Protocol):
    """Contrat d'un générateur d'images."""

    name: str

    def generate(self, prompt: str, *, negative: str = "", aspect_ratio: str = DEFAULT_ASPECT_RATIO) -> bytes:
        """Octets d'une image (PNG ou JPEG).

        Raises:
            ImageBackendError: génération impossible.
        """
        ...


def build_image_prompt(brief: ThumbnailBrief, *, style: str = STYLE_PROMPT) -> str:
    """Prompt d'illustration : la scène décrite par le modèle, puis le style imposé.

    La position du sujet est rappelée explicitement pour que la flèche, posée ensuite au
    compositing, tombe bien en face de lui.
    """
    placement = {
        "left": "the main subject is placed on the left third of the frame",
        "right": "the main subject is placed on the right third of the frame",
        "center": "the main subject is centred in the frame",
    }[brief.subject_position]
    return f"{brief.scene_description}. {placement}. {style}, 16:9 cinematic composition"


def build_negative_prompt(brief: ThumbnailBrief, *, base: str = BASE_NEGATIVE) -> str:
    """Prompt négatif : les exclusions permanentes, plus celles demandées par le modèle."""
    extra = brief.avoid.strip().rstrip(".")
    return f"{base}, {extra}" if extra else base


def _image_bytes_from(response: Any) -> bytes | None:
    """Premiers octets d'image trouvés dans une réponse du SDK Gemini.

    Parcourt toutes les parties plutôt que de supposer une position : la forme exacte de
    la réponse varie d'un modèle d'images à l'autre.
    """
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            blob = getattr(part, "inline_data", None) or getattr(part, "blob", None)
            data = getattr(blob, "data", None)
            mime = str(getattr(blob, "mime_type", "") or "")
            if data and mime.startswith("image/"):
                return data if isinstance(data, (bytes, bytearray)) else base64.b64decode(data)
    return None


class GeminiImageBackend:
    """Génération via les modèles d'images Gemini, avec rotation de clés et cascade.

    Args:
        manager: gestionnaire dédié ; sinon construit avec :data:`DEFAULT_IMAGE_MODEL_CASCADE`
            et les clés habituelles (``GEMINI_API_KEYS``, ``.env``, ``.gemini_key``).
        models: cascade de modèles d'images.
        max_rpm: requêtes par minute du gestionnaire dédié.
    """

    name = "gemini"

    def __init__(
        self,
        manager: GeminiManager | None = None,
        *,
        models: tuple[str, ...] = DEFAULT_IMAGE_MODEL_CASCADE,
        max_rpm: int = 5,
    ) -> None:
        self._manager = manager
        self._models = models
        self._max_rpm = max_rpm

    @property
    def manager(self) -> GeminiManager:
        if self._manager is None:
            self._manager = GeminiManager(models=self._models, max_rpm=self._max_rpm)
        return self._manager

    def generate(self, prompt: str, *, negative: str = "", aspect_ratio: str = DEFAULT_ASPECT_RATIO) -> bytes:
        from google.genai import types

        # Les modèles d'images n'acceptent pas de prompt négatif séparé : il est formulé
        # comme une consigne d'exclusion dans le prompt lui-même.
        text = f"{prompt}. Do not include: {negative}." if negative else prompt
        config = types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(aspect_ratio=aspect_ratio),
            http_options=types.HttpOptions(timeout=int(HTTP_TIMEOUT_S * 1000)),
        )
        response = self.manager.generate([types.Part.from_text(text=text)], config, label="image de miniature")
        data = _image_bytes_from(response)
        if not data:
            raise ImageBackendError("Reponse Gemini sans image (prompt refuse ou modalite non servie)")
        return bytes(data)


class StabilityBackend:
    """Génération via l'API REST Stability (SDXL / SD3). **Non exercé ici.**"""

    name = "stability"

    def __init__(self, api_key: str | None = None, *, url: str | None = None) -> None:
        self.api_key = api_key or os.environ.get(STABILITY_KEY_ENV_VAR, "").strip()
        self.url = url or os.environ.get(STABILITY_URL_ENV_VAR, "").strip() or DEFAULT_STABILITY_URL

    def generate(self, prompt: str, *, negative: str = "", aspect_ratio: str = DEFAULT_ASPECT_RATIO) -> bytes:
        import requests

        if not self.api_key:
            raise ImageBackendError(f"{STABILITY_KEY_ENV_VAR} absent de l'environnement ou de .env")
        try:
            response = requests.post(
                self.url,
                headers={"Authorization": f"Bearer {self.api_key}", "Accept": "image/*"},
                files={"none": ""},
                data={"prompt": prompt, "negative_prompt": negative, "aspect_ratio": aspect_ratio,
                      "output_format": "png"},
                timeout=HTTP_TIMEOUT_S,
            )
        except requests.RequestException as exc:
            raise ImageBackendError(f"Stability injoignable : {exc}") from exc
        if response.status_code != 200:
            raise ImageBackendError(f"Stability a repondu {response.status_code} : {response.text[:200]}")
        return response.content


class LocalWebuiBackend:
    """Génération via un serveur local compatible AUTOMATIC1111. **Non exercé ici.**"""

    name = "local"

    def __init__(self, url: str | None = None, *, width: int = 1344, height: int = 768, steps: int = 30) -> None:
        self.url = (url or os.environ.get(LOCAL_URL_ENV_VAR, "").strip() or DEFAULT_LOCAL_URL).rstrip("/")
        self.width, self.height, self.steps = width, height, steps

    def generate(self, prompt: str, *, negative: str = "", aspect_ratio: str = DEFAULT_ASPECT_RATIO) -> bytes:
        import requests

        payload = {
            "prompt": prompt, "negative_prompt": negative, "width": self.width, "height": self.height,
            "steps": self.steps, "cfg_scale": 7.0, "sampler_name": "DPM++ 2M Karras",
        }
        try:
            response = requests.post(f"{self.url}/sdapi/v1/txt2img", json=payload, timeout=HTTP_TIMEOUT_S)
            response.raise_for_status()
            images = response.json().get("images") or []
        except Exception as exc:  # noqa: BLE001 - requests ou JSON
            raise ImageBackendError(f"Serveur local injoignable ({self.url}) : {exc}") from exc
        if not images:
            raise ImageBackendError("Le serveur local n'a renvoye aucune image")
        return base64.b64decode(images[0])


def resolve_backend(name: str | None = None, *, manager: GeminiManager | None = None) -> ImageBackend:
    """Backend demandé (argument, sinon ``THUMBNAIL_IMAGE_BACKEND``, sinon Gemini).

    Raises:
        ImageBackendError: nom de backend inconnu.
    """
    from src.utils.config import load_dotenv

    load_dotenv()
    wanted = (name or os.environ.get(BACKEND_ENV_VAR, "") or "gemini").strip().lower()
    if wanted == "gemini":
        return GeminiImageBackend(manager)
    if wanted == "stability":
        return StabilityBackend()
    if wanted == "local":
        return LocalWebuiBackend()
    raise ImageBackendError(f"Backend d'images inconnu : {wanted} (gemini, stability ou local)")


def save_image(data: bytes, path: str | Path) -> Path:
    """Écrit les octets d'une image et renvoie son chemin.

    Raises:
        ImageBackendError: octets vides ou illisibles par Pillow.
    """
    from io import BytesIO

    from PIL import Image, UnidentifiedImageError

    if not data:
        raise ImageBackendError("Aucun octet d'image a ecrire")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(BytesIO(data)) as img:
            img.load()
            size = img.size
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageBackendError(f"Octets d'image illisibles : {exc}") from exc
    path.write_bytes(data)
    logger.info("Image de base : %s (%dx%d, %d Ko)", path, size[0], size[1], len(data) // 1024)
    return path


__all__ = [
    "DEFAULT_IMAGE_MODEL_CASCADE",
    "STYLE_PROMPT",
    "BASE_NEGATIVE",
    "DEFAULT_ASPECT_RATIO",
    "BACKEND_ENV_VAR",
    "ImageBackendError",
    "ImageBackend",
    "GeminiImageBackend",
    "StabilityBackend",
    "LocalWebuiBackend",
    "build_image_prompt",
    "build_negative_prompt",
    "resolve_backend",
    "save_image",
]
