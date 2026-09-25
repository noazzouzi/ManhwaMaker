"""Étapes 3 et 5 : spécification IA d'un bloc et juge des candidats (Claude CLI ou Gemini).

L'IA décide **quoi** garder, jamais **où** couper : elle rend des fractions grossières
de la hauteur du bloc (pas de 0,05), lues sur une règle graduée ajoutée de part et
d'autre de l'image ; les coordonnées exactes viennent des détecteurs et de la recherche.

- Sortie JSON stricte (schéma Pydantic transmis au modèle), revalidée ici ; une réponse
  invalide est redemandée avec l'erreur (``retries`` fois), puis on abandonne.
- Les réponses sont **mises en cache par hash du bloc** (et de la version du prompt et du
  fournisseur) : à entrée égale, même sortie, et zéro appel au 2e passage.
- Deux transports interchangeables (:class:`JsonClient`) :

  - :class:`ClaudeCliJson` : le CLI Claude Code installé sur la machine (``claude -p``),
    donc l'abonnement déjà connecté, sans clé API ; une session d'un échange par appel,
    prompt système remplacé, aucun outil ;
  - :class:`GeminiJson` : :class:`GeminiManager` du projet (clés, 429, cascade, 5 req/min),
    température 0 et graine fixe.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from pydantic import BaseModel, ValidationError

from src.utils.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

SPEC_VERSION = "toonsplit-spec-v2"
JUDGE_VERSION = "toonsplit-judge-v3"
#: Modèles qui aboutissent sur le compte gratuit (3.7 / 3.8-flash : 503 en boucle).
TOONSPLIT_MODELS: tuple[str, ...] = ("gemini-3.5-flash", "gemini-2.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite")
DEFAULT_RPM = 5
DEFAULT_CACHE_DIR = PROJECT_ROOT / "output" / ".cache" / "toonsplit"
#: Hauteur maximale de l'image envoyée (les fractions ne dépendent pas de l'échelle).
MAX_SPEC_HEIGHT = 3072
#: Tolérance au-delà de [0, 1] ramenée au bord (au-delà : réponse invalide).
Y_TOLERANCE = 0.05

Role = Literal["key", "insert", "text_only", "transition"]
DropKind = Literal["narration", "detached_bubble", "sfx", "background"]
ROLES: tuple[str, ...] = ("key", "insert", "text_only", "transition")
DROP_KINDS: tuple[str, ...] = ("narration", "detached_bubble", "sfx", "background")


class KeepZone(BaseModel):
    what: str
    y: list[float]
    priority: int = 1


class DropZone(BaseModel):
    what: str
    y: list[float]
    kind: DropKind


class BlockSpec(BaseModel):
    """Spécification d'un bloc (schéma de la mission)."""

    role: Role
    keep: list[KeepZone] = []
    drop: list[DropZone] = []
    tts: list[str] = []
    speaker_hints: list[str] = []


class JudgeVerdict(BaseModel):
    """Choix du juge parmi les candidats numérotés ; ``confident`` faux → on garde le n° 1."""

    best: int
    reason: str
    confident: bool = True


class AiError(RuntimeError):
    """Réponse IA toujours invalide après les nouvelles tentatives."""


# --- Validation ------------------------------------------------------------------------------
def _check_range(label: str, y: Sequence[float]) -> tuple[list[float] | None, str | None]:
    if len(y) != 2:
        return None, f"{label}: y must be [top, bottom], got {list(y)}"
    top, bottom = float(y[0]), float(y[1])
    if top < -Y_TOLERANCE or bottom > 1 + Y_TOLERANCE:
        return None, f"{label}: y {list(y)} outside [0, 1]"
    top, bottom = min(max(top, 0.0), 1.0), min(max(bottom, 0.0), 1.0)
    if not top < bottom:
        return None, f"{label}: top {top} must be smaller than bottom {bottom}"
    return [round(top, 3), round(bottom, 3)], None


def normalize_spec(spec: BlockSpec) -> tuple[BlockSpec, list[str]]:
    """Borne les fractions à [0, 1] et liste les erreurs qui justifient une nouvelle demande."""
    errors: list[str] = []
    keep, drop = [], []
    for i, zone in enumerate(spec.keep):
        y, err = _check_range(f"keep[{i}]", zone.y)
        if err:
            errors.append(err)
        else:
            keep.append(zone.model_copy(update={"y": y, "priority": min(max(zone.priority, 1), 3)}))
    for i, zone in enumerate(spec.drop):
        y, err = _check_range(f"drop[{i}]", zone.y)
        if err:
            errors.append(err)
        else:
            drop.append(zone.model_copy(update={"y": y}))
    if spec.role in ("key", "insert") and not keep:
        errors.append(f"role {spec.role!r} needs at least one keep zone")
    tts = [t.strip() for t in spec.tts if t and t.strip()]
    hints = [h.strip() for h in spec.speaker_hints if h is not None]
    return spec.model_copy(update={"keep": keep, "drop": drop, "tts": tts, "speaker_hints": hints}), errors


def check_verdict(verdict: JudgeVerdict, n: int) -> list[str]:
    return [] if 1 <= verdict.best <= n else [f"best must be between 1 and {n}, got {verdict.best}"]


def fallback_spec(boxes: Sequence[Any], height: int) -> BlockSpec:
    """Spec sans IA : garder les sujets détectés (têtes, personnes), sinon tout le bloc."""
    subjects = [b for b in boxes if getattr(b, "kind", "") in ("head", "person")]
    if subjects and height > 0:
        top = min(b.y0 for b in subjects) / height
        bottom = max(b.y1 for b in subjects) / height
        keep = [KeepZone(what="detected subjects", y=[round(top, 3), round(max(bottom, top + 0.01), 3)])]
    else:
        keep = [KeepZone(what="whole block", y=[0.0, 1.0])]
    return BlockSpec(role="key", keep=keep)


# --- Images ----------------------------------------------------------------------------------
def ruler_image(block: np.ndarray, *, max_height: int = MAX_SPEC_HEIGHT) -> np.ndarray:
    """Bloc entouré d'une règle graduée à gauche et à droite (traits tous les 5 %, valeurs tous les 10 %).

    0.0 est le bord haut du dessin, 1.0 son bord bas ; une marge blanche en haut et en
    bas laisse lisibles les graduations extrêmes.
    """
    h, w = block.shape[:2]
    if h > max_height:
        w = max(1, round(w * max_height / h))
        h = max_height
        block = cv2.resize(block, (w, h), interpolation=cv2.INTER_AREA)
    margin = max(56, round(0.09 * w))
    pad = max(20, round(0.02 * w))
    canvas = np.full((h + 2 * pad, w + 2 * margin, 3), 255, np.uint8)
    canvas[pad:pad + h, margin:margin + w] = block
    color = (30, 30, 200)
    font_scale = margin / 110
    thickness = max(1, round(margin / 40))
    for k in range(21):
        y = pad + round(k / 20 * h)
        major = k % 2 == 0
        length = round(margin * (0.35 if major else 0.18))
        cv2.line(canvas, (margin - length, y), (margin - 1, y), color, thickness)
        cv2.line(canvas, (margin + w, y), (margin + w + length, y), color, thickness)
        if major:
            label = f"{k / 20:.1f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            ty = min(max(y + th // 2, th + 1), canvas.shape[0] - 2)
            cv2.putText(canvas, label, (2, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
            cv2.putText(canvas, label, (canvas.shape[1] - tw - 2, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color,
                        thickness, cv2.LINE_AA)
    return canvas


def encode_jpeg(img: np.ndarray, quality: int = 90) -> bytes:
    ok, data = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("encodage JPEG impossible")
    return data.tobytes()


CANDIDATE_COLORS = ((40, 170, 40), (200, 90, 20), (20, 120, 230))


def judge_images(block: np.ndarray, windows: Sequence[tuple[int, int]], *, max_side: int = 1568) -> list[tuple[str, np.ndarray]]:
    """Images du juge : le bloc entier avec les fenêtres numérotées, puis chaque candidat recadré.

    Une image par vue (et non une planche unique) : chaque vue garde assez de détail une
    fois ramenée par le modèle à ~1568 px de grand côté.
    """
    h, w = block.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    overview = cv2.resize(block, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    thick = max(3, round(overview.shape[1] / 90))
    for n, (y0, y1) in enumerate(windows):
        color = CANDIDATE_COLORS[n % len(CANDIDATE_COLORS)]
        inset = thick * (n + 1)
        cv2.rectangle(overview, (inset, round(y0 * scale) + inset), (overview.shape[1] - 1 - inset,
                      round(y1 * scale) - 1 - inset), color, thick)
        cv2.putText(overview, str(n + 1), (inset + 8 + 44 * n, round(y0 * scale) + inset + 48), cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, color, 4, cv2.LINE_AA)
    views = [("Image 1: the whole block, candidate windows outlined and numbered.", overview)]
    for n, (y0, y1) in enumerate(windows):
        crop = block[y0:y1]
        f = min(1.0, max_side / max(crop.shape[:2]))
        if f < 1.0:
            crop = cv2.resize(crop, (max(1, round(crop.shape[1] * f)), max(1, round(crop.shape[0] * f))),
                              interpolation=cv2.INTER_AREA)
        views.append((f"Image {n + 2}: candidate {n + 1}.", crop))
    return views


# --- Prompts ---------------------------------------------------------------------------------
SPEC_SYSTEM = """You analyse ONE block cut from a vertical webtoon strip (Korean manhwa, English edition).
A video editor will turn the block into full-width video shots: you decide WHAT matters, code decides where to cut.

The image has a ruler on both sides: small ticks every 0.05, labelled ticks every 0.1.
0.0 is the top edge of the artwork, 1.0 its bottom edge (the white bands above and below are outside).
Give every vertical position as a fraction of the block height read from the ruler, as a multiple of 0.05.
Only vertical positions matter: shots always keep the full width.

role:
- key: characters or action that must appear in the video.
- insert: an object, place or detail without characters (flag, building, item, close-up of a hand).
- text_only: only text on a plain background (a narration line, a bubble on its own), no artwork worth showing.
- transition: a plain fade, a black or white screen, a decorative filler.

keep: zones that must stay in the image: characters with their whole heads (even seen from behind), bodies as far
as they matter, the object of an insert, and the speech bubbles attached to the characters.
priority 1 = essential, 2 = nice to have. key and insert blocks need at least one keep zone.

drop: zones that should leave the image when possible:
- narration: narration boxes or captions (their text goes to the voice-over),
- detached_bubble: speech bubbles that sit outside the panel border or far from their speaker. A bubble drawn
  over or right next to its speaker inside the panel is attached: it belongs to the keep zone, not to drop,
- sfx: sound-effect lettering (onomatopoeia),
- background: empty sky, blank margins, filler.
A drop zone may overlap a keep zone only when they really overlap in the image.

tts: every readable piece of text of the block, narration and dialogue, in reading order (top to bottom, left to
right), in normal sentence case ("MY NAME IS SUHO KIM." -> "My name is Suho Kim."). Leave out sound effects.
speaker_hints: who says each tts entry, same order ("narrator", "hero (brown hair)", "creatures"...);
an empty list when unknown.

Return only the JSON object."""

JUDGE_SYSTEM = """You judge candidate framings of ONE block of a vertical webtoon strip for a narrated video.
Image 1 shows the whole block with the numbered candidate windows outlined; the next images are the candidate
crops, in order.
Pick the crop that works best as a video shot:
1. no head or face is cut, no speech bubble or sound effect is sliced through;
2. the subject described below is fully visible and reasonably centred;
3. narration boxes and detached bubbles are left out when that costs nothing to the subject;
4. portrait framing close to 2:3 is preferred.
Set confident to false when the candidates are equally good or you cannot tell.
Return only the JSON object."""


def describe_spec(spec: BlockSpec) -> str:
    keep = "; ".join(f"{z.what} (y {z.y[0]:.2f}-{z.y[1]:.2f})" for z in spec.keep) or "nothing"
    drop = "; ".join(f"{z.kind}: {z.what} (y {z.y[0]:.2f}-{z.y[1]:.2f})" for z in spec.drop) or "nothing"
    return f"Block role: {spec.role}. Keep: {keep}. Drop if possible: {drop}."


# --- Cache -----------------------------------------------------------------------------------
class ResponseCache:
    """Réponses JSON validées, une par fichier, indexées par empreinte SHA-256."""

    def __init__(self, directory: Path | None) -> None:
        self.directory = directory

    @staticmethod
    def key(*parts: bytes | str) -> str:
        digest = hashlib.sha256()
        for part in parts:
            digest.update(part.encode("utf-8") if isinstance(part, str) else part)
            digest.update(b"\x00")
        return digest.hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        if self.directory is None:
            return None
        path = self.directory / f"{key}.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def put(self, key: str, value: dict[str, Any]) -> None:
        if self.directory is None:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{key}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)


def block_digest(block: np.ndarray) -> bytes:
    return hashlib.sha256(str(block.shape).encode() + np.ascontiguousarray(block).tobytes()).digest()


# --- Clients IA à sortie JSON ----------------------------------------------------------------
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
#: Image envoyée : ``(légende, JPEG)`` ; la légende précède l'image dans la requête.
Image = tuple[str, bytes]


class JsonClient:
    """Appel IA à sortie JSON validée par Pydantic, avec nouvelles tentatives motivées.

    Les sous-classes n'implémentent que le transport (:meth:`_send`). ``cache_tag`` distingue
    les réponses de fournisseurs différents dans le cache (vide pour Gemini : clés historiques).
    """

    name = "ai"
    cache_tag = ""

    def __init__(self, *, retries: int = 2) -> None:
        self.retries = retries
        self.n_calls = 0
        #: Coût équivalent au tarif API, quand le fournisseur le rapporte.
        self.cost_usd = 0.0

    def _send(self, system: str, text: str, images: Sequence[Image], schema: type[BaseModel], label: str) -> str | dict:
        raise NotImplementedError

    def ask(
        self, system: str, text: str, images: Sequence[Image], schema: type[BaseModel], *, label: str,
        check: Callable[[Any], tuple[Any, list[str]]] | None = None,
    ) -> BaseModel:
        """Envoie le texte et les images ; renvoie l'objet validé (``check`` peut le normaliser et lister des erreurs)."""
        prompt = text
        last = "aucune reponse"
        for attempt in range(1, self.retries + 2):
            self.n_calls += 1
            raw = self._send(system, prompt, images, schema, label)
            shown = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
            try:
                if isinstance(raw, dict):
                    obj = schema.model_validate(raw)
                else:
                    obj = schema.model_validate_json(_FENCE.sub("", raw.strip()))
                errors: list[str] = []
                if check is not None:
                    obj, errors = check(obj)
                if not errors:
                    return obj
                last = "; ".join(errors)
            except (ValidationError, ValueError) as exc:
                last = str(exc).splitlines()[0][:400] if str(exc) else type(exc).__name__
            logger.warning("%s : reponse invalide (essai %d/%d) : %s", label, attempt, self.retries + 1, last)
            prompt = (f"{text}\n\nYour previous answer was rejected: {last}\nPrevious answer: {shown[:2000]}\n"
                      "Answer again with the corrected JSON object only.")
        raise AiError(f"{label} : reponse invalide apres {self.retries + 1} essai(s) : {last} ; derniere reponse : "
                      f"{shown[:200]}")


class GeminiJson(JsonClient):
    """Transport Gemini (clés, cascade, RPM via :class:`GeminiManager`)."""

    name = "gemini"

    def __init__(
        self,
        manager: Any | None = None,
        *,
        models: Sequence[str] = TOONSPLIT_MODELS,
        max_rpm: int = DEFAULT_RPM,
        temperature: float = 0.0,
        seed: int = 7,
        retries: int = 2,
        max_output_tokens: int = 8192,
    ) -> None:
        super().__init__(retries=retries)
        self._manager = manager
        self.models = tuple(models)
        self.max_rpm = max_rpm
        self.temperature = temperature
        self.seed = seed
        self.max_output_tokens = max_output_tokens

    @property
    def manager(self) -> Any:
        if self._manager is None:
            from src.utils.gemini_manager import GeminiManager

            self._manager = GeminiManager(models=list(self.models), max_rpm=self.max_rpm)
        return self._manager

    def config(self, system: str, schema: type[BaseModel]) -> Any:
        from google.genai import types

        return types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=schema,
            temperature=self.temperature,
            seed=self.seed,
            max_output_tokens=self.max_output_tokens,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

    def _send(self, system: str, text: str, images: Sequence[Image], schema: type[BaseModel], label: str) -> str:
        from google.genai import types

        parts = [types.Part.from_text(text=text)]
        for caption, data in images:
            if caption:
                parts.append(types.Part.from_text(text=caption))
            parts.append(types.Part.from_bytes(data=data, mime_type="image/jpeg"))
        response = self.manager.generate(parts, self.config(system, schema), label=label)
        return getattr(response, "text", None) or ""


#: Variable d'environnement : chemin explicite du binaire ``claude``.
CLAUDE_BIN_ENV = "TOONSPLIT_CLAUDE_BIN"
#: Modèle par défaut du transport Claude CLI (``--claude-model`` pour un autre : alias ``sonnet``...).
CLAUDE_DEFAULT_MODEL = "claude-opus-5"
#: Erreur de quota ou de limite d'usage : inutile d'insister sur les blocs suivants.
_LIMIT = re.compile(r"usage limit|rate limit|limit reached|quota|credit balance|out of (?:extra )?usage", re.IGNORECASE)


def find_claude() -> str:
    """Binaire du CLI Claude Code (sous Windows, l'exécutable derrière le raccourci npm ``claude.cmd``)."""
    explicit = os.environ.get(CLAUDE_BIN_ENV, "").strip()
    if explicit:
        return explicit
    found = shutil.which("claude")
    if not found:
        raise AiError(f"CLI 'claude' introuvable dans le PATH (ou definir {CLAUDE_BIN_ENV})")
    path = Path(found)
    if os.name == "nt" and path.suffix.lower() in ("", ".cmd", ".bat"):
        # Un .cmd passe par cmd.exe, qui casse les arguments multilignes (prompt système) :
        # on appelle directement l'exécutable installé par npm.
        exe = path.parent / "node_modules" / "@anthropic-ai" / "claude-code" / "bin" / "claude.exe"
        if exe.is_file():
            return str(exe)
    return found


def _child_env() -> dict[str, str]:
    """Environnement d'un appel autonome : sans les variables d'une session Claude Code parente."""
    return {k: v for k, v in os.environ.items() if not k.startswith("CLAUDE") or k == "CLAUDE_CONFIG_DIR"}


def _inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Schéma JSON Pydantic sans ``$ref`` / ``$defs`` (plus robuste pour la sortie structurée)."""
    defs = schema.get("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return walk(defs[node["$ref"].rsplit("/", 1)[-1]])
            return {k: walk(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(schema)


class ClaudeCliJson(JsonClient):
    """Transport Claude par le CLI Claude Code local (``claude -p``) : l'abonnement de la machine, sans clé API.

    Chaque appel est une session d'un seul échange : prompt système remplacé (celui de
    Claude Code n'est pas envoyé), aucun outil, session non enregistrée, sortie structurée
    validée par ``--json-schema``, images en base64 (entrée ``stream-json``). Aucun fichier
    de réglages n'est chargé (``--setting-sources ""``) : sans cela le ``CLAUDE.md`` global de
    l'utilisateur (« Réponds en français »...) s'invite dans chaque requête. Aucun serveur MCP
    non plus (``--strict-mcp-config``) : les connecteurs claude.ai du compte ajoutaient ~19 000
    jetons de définitions d'outils à chaque appel (0,19 $ par appel Opus, mesuré le 25/09). Modèle explicite
    (défaut :data:`CLAUDE_DEFAULT_MODEL`) pour que le cache ne mélange pas deux modèles.
    """

    name = "claude"

    def __init__(
        self, *, model: str | None = None, effort: str | None = None, executable: str | None = None,
        retries: int = 2, timeout_s: float = 300.0,
    ) -> None:
        model = model or CLAUDE_DEFAULT_MODEL
        super().__init__(retries=retries)
        self.model = model
        self.effort = effort
        self.executable = executable
        self.timeout_s = timeout_s
        self.cache_tag = f"claude-cli:{model}"
        self.models_used: set[str] = set()
        self._exhausted: str | None = None

    def command(self, system: str, schema: type[BaseModel]) -> list[str]:
        cmd = [
            self.executable or find_claude(), "-p", "--input-format", "stream-json", "--output-format", "stream-json",
            "--verbose", "--tools", "", "--no-session-persistence", "--setting-sources", "", "--strict-mcp-config", "--system-prompt", system,
            "--json-schema", json.dumps(_inline_refs(schema.model_json_schema())), "--model", self.model,
        ]
        if self.effort:
            cmd += ["--effort", self.effort]
        return cmd

    def _send(self, system: str, text: str, images: Sequence[Image], schema: type[BaseModel], label: str) -> str | dict:
        if self._exhausted:
            raise AiError(self._exhausted)
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        for caption, data in images:
            if caption:
                content.append({"type": "text", "text": caption})
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/jpeg", "data": base64.b64encode(data).decode("ascii")}})
        message = {"type": "user", "message": {"role": "user", "content": content}}
        try:
            proc = subprocess.run(
                self.command(system, schema), input=json.dumps(message) + "\n", capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=self.timeout_s, cwd=tempfile.gettempdir(), env=_child_env(),
            )
        except subprocess.TimeoutExpired:
            return f"(no answer: claude CLI timed out after {self.timeout_s:.0f}s)"
        result = None
        for line in proc.stdout.splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict) and event.get("type") == "result":
                result = event
        if result is None:
            detail = (proc.stderr or proc.stdout or "").strip()[:300]
            if _LIMIT.search(detail):
                self._exhausted = f"claude CLI : limite d'usage ({detail})"
                raise AiError(self._exhausted)
            raise AiError(f"{label} : claude CLI sans resultat (code {proc.returncode}) : {detail}")
        self.cost_usd += float(result.get("total_cost_usd") or 0.0)
        self.models_used.update(result.get("modelUsage") or {})
        if result.get("is_error"):
            detail = str(result.get("result") or result.get("subtype") or "erreur")
            if _LIMIT.search(detail):
                self._exhausted = f"claude CLI : limite d'usage ({detail[:200]})"
                raise AiError(self._exhausted)
            return f"(error: {detail[:300]})"
        structured = result.get("structured_output")
        return structured if isinstance(structured, dict) else str(result.get("result") or "")


def make_client(backend: str, **options: Any) -> JsonClient:
    """Client IA par nom : ``claude`` (CLI local) ou ``gemini``."""
    if backend == "claude":
        return ClaudeCliJson(**{k: v for k, v in options.items() if k in ("model", "effort", "executable", "retries")})
    if backend == "gemini":
        return GeminiJson(**{k: v for k, v in options.items() if k in ("max_rpm", "retries")})
    raise ValueError(f"fournisseur IA inconnu : {backend} (claude | gemini)")


class AiSpecProvider:
    """Spec IA d'un bloc (étape 3), mise en cache par hash du bloc et par fournisseur."""

    def __init__(self, client: JsonClient | None = None, *, cache_dir: Path | None = DEFAULT_CACHE_DIR) -> None:
        self.client = client or ClaudeCliJson()
        self.cache = ResponseCache(cache_dir)
        self.cache_hits = 0

    @property
    def source(self) -> str:
        return self.client.name

    def __call__(self, block: np.ndarray, index: int) -> BlockSpec:
        tag = [self.client.cache_tag] if self.client.cache_tag else []
        key = ResponseCache.key(SPEC_VERSION, *tag, block_digest(block))
        cached = self.cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return BlockSpec.model_validate(cached)
        h, w = block.shape[:2]
        spec = self.client.ask(
            SPEC_SYSTEM, f"Webtoon block, {w} x {h} pixels. Analyse it.", [("", encode_jpeg(ruler_image(block)))],
            BlockSpec, label=f"toonsplit spec bloc {index}", check=normalize_spec,
        )
        self.cache.put(key, spec.model_dump())
        return spec


class AiJudge:
    """Juge IA (étape 5) : choisit parmi les candidats ; incertain ou invalide → n° 1."""

    def __init__(self, client: JsonClient | None = None, *, cache_dir: Path | None = DEFAULT_CACHE_DIR) -> None:
        self.client = client or ClaudeCliJson()
        self.cache = ResponseCache(cache_dir)
        self.cache_hits = 0

    def __call__(
        self, block: np.ndarray, windows: Sequence[tuple[int, int]], spec: BlockSpec, index: int,
        notes: Sequence[str] = (),
    ) -> JudgeVerdict:
        tag = [self.client.cache_tag] if self.client.cache_tag else []
        key = ResponseCache.key(JUDGE_VERSION, *tag, block_digest(block), json.dumps([list(w) for w in windows]),
                                spec.model_dump_json(), json.dumps(list(notes)))
        cached = self.cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return JudgeVerdict.model_validate(cached)
        h = block.shape[0]
        listing = "\n".join(
            f"Candidate {n}: from {y0 / h:.2f} to {y1 / h:.2f} of the block height"
            + (f" ({notes[n - 1]})" if n - 1 < len(notes) and notes[n - 1] else "")
            for n, (y0, y1) in enumerate(windows, start=1)
        )
        images = [(caption, encode_jpeg(img)) for caption, img in judge_images(block, windows)]
        verdict = self.client.ask(
            JUDGE_SYSTEM, f"{describe_spec(spec)}\n{listing}\nWhich candidate is best?", images, JudgeVerdict,
            label=f"toonsplit juge bloc {index}", check=lambda v: (v, check_verdict(v, len(windows))),
        )
        self.cache.put(key, verdict.model_dump())
        return verdict


class ManualSpecProvider:
    """Specs écrites à la main (format ``ai_spec.json`` : une entrée par bloc, clé ``block``)."""

    source = "manual"

    def __init__(self, specs: Sequence[dict[str, Any]]) -> None:
        self.specs: dict[int, BlockSpec] = {}
        for i, raw in enumerate(specs):
            data = {k: v for k, v in raw.items() if k != "block"}
            spec, errors = normalize_spec(BlockSpec.model_validate(data))
            if errors:
                raise ValueError(f"spec manuelle du bloc {raw.get('block', i)} invalide : {'; '.join(errors)}")
            self.specs[int(raw.get("block", i))] = spec

    @classmethod
    def from_file(cls, path: str | Path) -> ManualSpecProvider:
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def __call__(self, block: np.ndarray, index: int) -> BlockSpec:
        if index not in self.specs:
            raise KeyError(f"pas de spec manuelle pour le bloc {index}")
        return self.specs[index]


__all__ = [
    "SPEC_VERSION", "JUDGE_VERSION", "TOONSPLIT_MODELS", "DEFAULT_RPM", "DEFAULT_CACHE_DIR", "ROLES", "DROP_KINDS",
    "KeepZone", "DropZone", "BlockSpec", "JudgeVerdict", "AiError", "normalize_spec", "check_verdict",
    "fallback_spec", "ruler_image", "encode_jpeg", "judge_images", "describe_spec", "SPEC_SYSTEM", "JUDGE_SYSTEM",
    "ResponseCache", "block_digest", "Image", "JsonClient", "GeminiJson", "CLAUDE_BIN_ENV", "CLAUDE_DEFAULT_MODEL",
    "find_claude",
    "ClaudeCliJson", "make_client", "AiSpecProvider", "AiJudge", "ManualSpecProvider",
]
