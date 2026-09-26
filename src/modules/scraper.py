"""Module 1 — Scraper & Stitcher (Webtoons, Asura Scans).

Deux sites : Webtoons (``webtoons.com``, décrit ci-dessous) et Asura Scans
(``asurascans.com``, parsing dans :mod:`src.modules.asura`). Le site est reconnu à
l'URL ; téléchargement, assemblage et découpe sont communs.

Pipeline :

1. :func:`fetch_chapter_image_urls` télécharge la page du chapitre et en extrait
   les URLs des morceaux d'image (``div#_imageList img._images``, attribut
   ``data-url``) ainsi que les titres → :class:`~src.models.chapter.ChapterMeta`.
2. :func:`download_images` télécharge séquentiellement les morceaux **en mémoire**
   (PIL, RGB), avec délai de politesse et retries.
3. :func:`stitch_webtoon_pages` normalise les largeurs puis concatène les
   morceaux verticalement via ``np.vstack`` en **une seule bande continue**.
4. :func:`scrape_chapter` enchaîne les trois étapes ; :func:`save_strip`
   enregistre la bande sur disque (optionnel).

Garde-fou : chaque requête (page + images) porte ``User-Agent`` et
``Referer: https://www.webtoons.com/`` — sans eux Webtoons répond 403. Les requêtes
Asura portent le ``Referer`` d'Asura (:data:`src.modules.asura.REFERER`).

CLI : ``python -m src.modules.scraper <url> [--out output/strip.png]``.
"""

from __future__ import annotations

import logging
import re
import time
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import parse_qs, urljoin, urlparse, urlunparse

import numpy as np
import requests
import typer
from bs4 import BeautifulSoup, Tag
from PIL import Image, UnidentifiedImageError

from src.models.chapter import ChapterMeta
from src.modules import asura
from src.utils import progress
from src.utils.config import DEFAULT_SITE_LANGUAGE
from src.utils.http import ensure_mandatory_headers, get_with_retry

logger = logging.getLogger(__name__)

__all__ = [
    "ScraperError",
    "PLACEHOLDER_SRC_MARKER",
    "IMAGE_SELECTORS",
    "FORMAT_MAX_DIMENSION",
    "DEFAULT_SITE_LANGUAGE",
    "normalize_webtoon_url",
    "parse_ids_from_url",
    "episode_url",
    "series_list_url",
    "parse_episode_links",
    "discover_episodes",
    "chapter_ids",
    "is_series_url",
    "parse_chapter_html",
    "fetch_chapter_image_urls",
    "download_images",
    "stitch_webtoon_pages",
    "scrape_chapter",
    "save_strip",
    "ascii_safe",
    "format_summary",
    "main",
]

#: Fragment identifiant l'image ``src`` de remplissage (pixel transparent) de Webtoons.
PLACEHOLDER_SRC_MARKER: str = "bg_transparency"

#: Sélecteurs CSS des balises image, du plus précis au plus permissif.
IMAGE_SELECTORS: tuple[str, ...] = ("div#_imageList img._images", "img._images")

#: Suffixe que Webtoons ajoute à la balise ``<title>`` (``"Tower of God | WEBTOON"``).
_SITE_TITLE_SUFFIX: str = "WEBTOON"

#: Dimension maximale (largeur ou hauteur, en px) par extension de fichier. Les
#: bandes réelles dépassent couramment 16 383 px (WebP) et parfois 65 500 px
#: (JPEG) ; seul le PNG est sans limite pratique.
FORMAT_MAX_DIMENSION: dict[str, int] = {
    ".jpg": 65_500,
    ".jpeg": 65_500,
    ".webp": 16_383,
}

# Indirection vers ``time.sleep`` (neutralisable dans les tests).
_sleep = time.sleep

#: Segment de langue d'une URL Webtoons (``/en/``, ``/fr/``, ``/zh-hant/``...).
_LANGUAGE_SEGMENT = re.compile(r"^[a-z]{2}(?:-[a-z]+)?$")


class ScraperError(RuntimeError):
    """Erreur fonctionnelle du scraper (page inattendue, image indécodable...)."""


def normalize_webtoon_url(url: str, language: str | None = DEFAULT_SITE_LANGUAGE) -> str:
    """Force le segment de langue d'une URL Webtoons (``/fr/...`` → ``/en/...``).

    Convention du projet : les chapitres sont lus sur la version **anglaise** du
    site par défaut. Le site redirige vers les slugs canoniques de la langue cible
    à partir de ``title_no`` / ``episode_no`` : seul le premier segment du chemin
    est modifié. Une URL sans segment de langue reconnaissable, ou
    ``language`` vide/``None``, est renvoyée telle quelle.

    Args:
        url: URL du viewer Webtoons.
        language: code de langue cible (``"en"``, ``"fr"``...), ``None`` pour ne rien changer.
    """
    if not language:
        return url
    parsed = urlparse(url)
    segments = parsed.path.split("/")
    # segments[0] est vide (chemin absolu) ; segments[1] est le code de langue.
    if len(segments) < 2 or not _LANGUAGE_SEGMENT.match(segments[1]):
        return url
    target = language.lower()
    if segments[1] == target:
        return url
    segments[1] = target
    normalized = urlunparse(parsed._replace(path="/".join(segments)))
    logger.info("URL Webtoons normalisee en /%s/ : %s", target, normalized)
    return normalized


# --- Parsing ------------------------------------------------------------------------


def _query_int(query: dict[str, list[str]], key: str) -> int | None:
    """Lit un entier dans une query-string parsée ; ``None`` si absent/invalide."""
    values = query.get(key)
    if not values:
        return None
    try:
        return int(values[0])
    except (TypeError, ValueError):
        logger.debug("Parametre %s non numerique: %r", key, values[0])
        return None


def parse_ids_from_url(url: str) -> tuple[int | None, int | None]:
    """Extrait ``(title_no, episode_no)`` de la query-string d'une URL Webtoons.

    Exemple : ``.../viewer?title_no=95&episode_no=1`` → ``(95, 1)``.
    Les valeurs absentes ou non numériques donnent ``None``.
    """
    query = parse_qs(urlparse(url).query)
    return _query_int(query, "title_no"), _query_int(query, "episode_no")


def episode_url(url: str, episode_no: int | float) -> str:
    """URL du viewer d'un autre épisode de la même série (``episode_no`` remplacé).

    URL Asura : URL du chapitre ``episode_no`` (:func:`src.modules.asura.chapter_url`).

    Webtoons ignore le slug d'épisode du chemin et redirige vers l'URL canonique à
    partir de ``title_no`` / ``episode_no`` : une URL de liste (``.../list?title_no=N``)
    ou de viewer convient.
    """
    if asura.is_asura_url(url):
        return asura.chapter_url(url, episode_no)
    if episode_no < 1:
        raise ValueError("episode_no doit etre >= 1")
    parsed = urlparse(url)
    segments = [s for s in parsed.path.split("/") if s]
    if segments and segments[-1] == "list":
        segments = segments[:-1] + ["episode", "viewer"]
    elif not segments or segments[-1] != "viewer":
        segments = segments + ["viewer"]
    query = {k: v[0] for k, v in parse_qs(parsed.query).items() if v}
    query["episode_no"] = str(episode_no)
    query_string = "&".join(f"{k}={v}" for k, v in query.items())
    return urlunparse(parsed._replace(path="/" + "/".join(segments), query=query_string))


def series_list_url(url: str) -> str:
    """URL de la liste des épisodes d'une série (``.../<serie>/list?title_no=N``) depuis une URL de viewer ou de liste."""
    parsed = urlparse(url)
    segments = [s for s in parsed.path.split("/") if s]
    title_no, _ = parse_ids_from_url(url)
    if segments and segments[-1] == "viewer":
        segments = segments[:-2] if len(segments) >= 2 else []
    elif segments and segments[-1] == "list":
        segments = segments[:-1]
    query = f"title_no={title_no}" if title_no is not None else ""
    return urlunparse(parsed._replace(path="/" + "/".join(segments + ["list"]), query=query, fragment=""))


def parse_episode_links(html: str, base_url: str) -> dict[int, str]:
    """``{episode_no: URL du viewer}`` trouvés dans une page de liste Webtoons (liens ``episode_no=``)."""
    soup = BeautifulSoup(html, "lxml")
    found: dict[int, str] = {}
    for anchor in soup.select("a[href*='episode_no=']"):
        href = urljoin(base_url, anchor.get("href", ""))
        _, episode_no = parse_ids_from_url(href)
        if episode_no is not None and "viewer" in urlparse(href).path and episode_no not in found:
            found[episode_no] = href
    return found


def discover_episodes(
    url: str,
    session: requests.Session | None = None,
    *,
    language: str | None = DEFAULT_SITE_LANGUAGE,
    max_pages: int = 60,
) -> dict[int | float, str]:
    """Liste tous les épisodes d'une série (``{episode_no: URL}``) en parcourant les pages de liste.

    Asura liste tous ses chapitres sur la page de la série (numéros décimaux compris).

    Args:
        url: URL de la série (liste) ou de n'importe quel épisode.
        session: session HTTP optionnelle (en-têtes obligatoires garantis).
        language: langue du site imposée à l'URL.
        max_pages: garde-fou sur la pagination (``&page=N``).
    """
    if asura.is_asura_url(url):
        page_url = asura.series_url(url)
        with _session_scope(session) as http:
            response = get_with_retry(http, page_url, **_site_kwargs(page_url))
        chapters = asura.parse_chapter_links(html_text(response), page_url)
        logger.info("%d chapitre(s) trouve(s) pour %s", len(chapters), page_url)
        return dict(sorted(chapters.items()))
    list_url = series_list_url(normalize_webtoon_url(url, language))
    episodes: dict[int | float, str] = {}
    with _session_scope(session) as http:
        for page in range(1, max_pages + 1):
            page_url = f"{list_url}&page={page}" if page > 1 else list_url
            response = get_with_retry(http, page_url)
            links = parse_episode_links(html_text(response), page_url)
            new = {no: link for no, link in links.items() if no not in episodes}
            if not new:
                break
            episodes.update(new)
    logger.info("%d episode(s) trouve(s) pour %s", len(episodes), list_url)
    return dict(sorted(episodes.items()))


def chapter_ids(url: str) -> tuple[str | None, int | float | None]:
    """``(clé de série, numéro de chapitre)`` d'une URL, quel que soit le site.

    La clé est stable d'un chapitre à l'autre (``t<title_no>`` sur Webtoons,
    ``asura:<slug>`` sur Asura) ; ``None`` quand l'URL ne la donne pas.
    """
    if asura.is_asura_url(url):
        series, number = asura.parse_url(url)
        return (f"asura:{series}" if series else None), number
    title_no, episode_no = parse_ids_from_url(url)
    return (f"t{title_no}" if title_no is not None else None), episode_no


def is_series_url(url: str) -> bool:
    """Vrai pour une page de série (liste des épisodes) plutôt qu'un chapitre."""
    if asura.is_asura_url(url):
        return asura.is_series_url(url)
    segments = [s for s in urlparse(url).path.split("/") if s]
    return bool(segments) and segments[-1] == "list"


def _site_kwargs(url: str) -> dict:
    """Arguments de requête propres au site (``Referer`` d'Asura ; rien pour Webtoons)."""
    return {"headers": {"Referer": asura.REFERER}} if asura.is_asura_url(url) else {}


def _text_of(soup: BeautifulSoup, selector: str) -> str:
    """Texte (normalisé) du premier élément correspondant au sélecteur, sinon ``""``."""
    node = soup.select_one(selector)
    if node is None:
        return ""
    return " ".join(node.get_text(" ", strip=True).split())


def _split_page_title(raw_title: str) -> tuple[str, str]:
    """Découpe un titre de page en ``(série, épisode)``.

    - ``"Tower of God - [Season 1] Ep. 0"`` → ``("Tower of God", "[Season 1] Ep. 0")``
    - ``"Tower of God | WEBTOON"`` → ``("Tower of God", "")`` (suffixe du site retiré)
    - ``"Tower of God"`` → ``("Tower of God", "")``

    L'épisode n'est **jamais** dupliqué à partir de la série : s'il est
    inconnu, la chaîne vide est renvoyée.
    """
    parts = [part.strip() for part in raw_title.split(" | ")]
    if len(parts) > 1 and parts[-1].upper() == _SITE_TITLE_SUFFIX:
        parts = parts[:-1]
    title = " | ".join(part for part in parts if part)
    if " - " in title:
        series, episode = title.split(" - ", 1)
        return series.strip(), episode.strip()
    return title.strip(), ""


def _extract_titles(soup: BeautifulSoup) -> tuple[str, str]:
    """Extrait ``(series_title, episode_title)`` avec repli sur ``og:title`` puis ``<title>``.

    Sélecteurs principaux : ``div.subj_info a.subj`` (série) et
    ``h1.subj_episode`` (épisode). Repli : ``meta[property=og:title]`` au format
    ``"Serie - Episode"``, puis la balise ``<title>`` (``"Serie | WEBTOON"``).
    Un titre d'épisode introuvable reste ``""`` (jamais une copie de la série).
    """
    series = _text_of(soup, "div.subj_info a.subj") or _text_of(soup, "a.subj")
    episode = _text_of(soup, "h1.subj_episode")

    if not series or not episode:
        og = soup.select_one('meta[property="og:title"]')
        og_title = " ".join(str(og.get("content") or "").split()) if og is not None else ""
        if not og_title:
            og_title = _text_of(soup, "title")
        if og_title:
            og_series, og_episode = _split_page_title(og_title)
            series = series or og_series
            episode = episode or og_episode

    return series, episode


def _extract_episode_label_no(soup: BeautifulSoup) -> int | None:
    """Lit le numéro d'épisode affiché dans ``span.tx`` (ex. ``"#1"`` → ``1``)."""
    label = _text_of(soup, "span.tx")
    digits = label.lstrip("#").strip()
    if digits.isdigit():
        return int(digits)
    return None


def _image_url_from_tag(tag: Tag) -> str:
    """Retourne l'URL réelle d'une balise ``img._images`` (``data-url`` prioritaire).

    ``src`` n'est utilisé qu'en repli, et seulement s'il ne s'agit pas de
    l'image de remplissage transparente de Webtoons. Retourne ``""`` si aucune
    URL exploitable n'est trouvée.
    """
    data_url = str(tag.get("data-url") or "").strip()
    if data_url:
        return data_url
    src = str(tag.get("src") or "").strip()
    if src and PLACEHOLDER_SRC_MARKER not in src:
        return src
    return ""


def _extract_image_urls(soup: BeautifulSoup, base_url: str) -> list[str]:
    """Extrait, dans l'ordre du DOM, les URLs des morceaux d'image du viewer.

    Le premier sélecteur de :data:`IMAGE_SELECTORS` qui renvoie au moins une
    balise est retenu. Les URLs sont conservées telles quelles (``?type=q90``
    inclus) ; seules les URLs relatives sont résolues contre ``base_url``.
    """
    tags: list[Tag] = []
    for selector in IMAGE_SELECTORS:
        tags = soup.select(selector)
        if tags:
            logger.debug("Selecteur %r: %d balise(s)", selector, len(tags))
            break

    urls: list[str] = []
    for position, tag in enumerate(tags):
        url = _image_url_from_tag(tag)
        if not url:
            logger.warning("Balise image #%d sans URL exploitable, ignoree", position)
            continue
        urls.append(urljoin(base_url, url))
    return urls


def parse_chapter_html(html: str, url: str, final_url: str | None = None) -> ChapterMeta:
    """Analyse le HTML d'une page de chapitre et construit un :class:`ChapterMeta`.

    Fonction pure (sans réseau), directement testable sur un fixture HTML.

    Args:
        html: Contenu HTML de la page du viewer.
        url: URL demandée (stockée dans ``ChapterMeta.url``).
        final_url: URL effective après redirections ; ``url`` si ``None``.

    Returns:
        Métadonnées du chapitre, ``chunk_sizes`` vide.
    """
    final_url = final_url or url
    soup = BeautifulSoup(html, "lxml")

    series_title, episode_title = _extract_titles(soup)
    title_no, episode_no = parse_ids_from_url(url)
    if title_no is None or episode_no is None:
        alt_title_no, alt_episode_no = parse_ids_from_url(final_url)
        title_no = title_no if title_no is not None else alt_title_no
        episode_no = episode_no if episode_no is not None else alt_episode_no
    if episode_no is None:
        episode_no = _extract_episode_label_no(soup)
    if not episode_title and episode_no is not None:
        # Dernier repli : libellé neutre construit à partir du numéro d'épisode.
        episode_title = f"Episode {episode_no}"

    image_urls = _extract_image_urls(soup, base_url=final_url)

    meta = ChapterMeta(
        url=url,
        final_url=final_url,
        series_title=series_title,
        episode_title=episode_title,
        title_no=title_no,
        episode_no=episode_no,
        image_urls=image_urls,
    )
    logger.info("Chapitre analyse: %s", meta.summary())
    return meta


# --- Réseau -------------------------------------------------------------------------


@contextmanager
def _session_scope(session: requests.Session | None) -> Iterator[requests.Session]:
    """Fournit une session munie des en-têtes obligatoires, fermée si créée ici.

    Une session passée par l'appelant lui appartient : elle est complétée
    (:func:`ensure_mandatory_headers`) mais jamais fermée. Une session créée
    localement (``session is None``) est fermée en sortie pour libérer les
    connexions keep-alive vers webtoons.com et le CDN.
    """
    owns_session = session is None
    session = ensure_mandatory_headers(session)
    try:
        yield session
    finally:
        if owns_session:
            session.close()


def fetch_chapter_image_urls(
    url: str,
    session: requests.Session | None = None,
    *,
    language: str | None = DEFAULT_SITE_LANGUAGE,
) -> ChapterMeta:
    """Télécharge la page du chapitre et en extrait les URLs d'images + titres.

    Args:
        url: URL du viewer Webtoons (``.../viewer?title_no=..&episode_no=..``) ou
            d'un chapitre Asura (``.../comics/<serie>/chapter/<n>``).
        session: Session HTTP ; créée (puis fermée) avec les en-têtes
            obligatoires si ``None``.
        language: langue du site imposée à l'URL Webtoons (voir
            :func:`normalize_webtoon_url`) ; ``None`` pour garder l'URL telle quelle.

    Returns:
        :class:`ChapterMeta` avec ``image_urls`` renseigné (``url`` = URL normalisée).

    Raises:
        ScraperError: Si aucune image n'est trouvée sur la page.
        requests.HTTPError: Si la page ne peut pas être téléchargée (statut
            d'erreur définitif, ou erreur réseau persistante après retries).
    """
    on_asura = asura.is_asura_url(url)
    if not on_asura:
        url = normalize_webtoon_url(url, language)
    with _session_scope(session) as http:
        logger.info("Telechargement de la page: %s", url)
        response = get_with_retry(http, url, **_site_kwargs(url))
        final_url = str(getattr(response, "url", "") or url)
        if final_url != url:
            logger.info("Redirige vers: %s", final_url)
        html = html_text(response)

    parse = asura.parse_chapter_html if on_asura else parse_chapter_html
    meta = parse(html, url=url, final_url=final_url)
    if not meta.image_urls:
        selectors = (asura.IMAGE_SELECTOR,) if on_asura else IMAGE_SELECTORS
        raise ScraperError(
            f"Aucune image trouvee sur {final_url} "
            "(selecteurs testes: " + ", ".join(selectors) + ")"
        )
    return meta


def html_text(response: Any) -> str:
    """Page HTML décodée. Sans encodage annoncé par le serveur, UTF-8 d'abord.

    ``requests`` suppose alors Latin-1 (RFC 2616) : Asura Scans ne l'annonce pas, et le titre
    « Genius Archer’s Streaming » devenait « Genius Archerâs Streaming » (26/09).
    """
    content_type = str((getattr(response, "headers", None) or {}).get("content-type", "")).lower()
    content = getattr(response, "content", None)
    if content and "charset=" not in content_type:
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            pass
    return response.text


def _to_rgb(img: Image.Image) -> Image.Image:
    """Convertit une image PIL en RGB (fond blanc sous les zones transparentes)."""
    if img.mode == "RGB":
        return img
    has_alpha = img.mode in ("RGBA", "LA") or (
        img.mode == "P" and "transparency" in img.info
    )
    if has_alpha:
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return img.convert("RGB")


def _decode_image(content: bytes, url: str) -> Image.Image:
    """Décode des octets en image PIL RGB entièrement chargée en mémoire."""
    try:
        img = Image.open(BytesIO(content))
        img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ScraperError(
            f"Impossible de decoder l'image {url} ({len(content)} octets): {exc}"
        ) from exc
    return _to_rgb(img)


def download_images(
    urls: list[str],
    session: requests.Session | None = None,
    *,
    delay: float = 0.15,
    max_retries: int = 3,
    referer: str | None = None,
) -> list[Image.Image]:
    """Télécharge séquentiellement les morceaux d'image, en mémoire, en RGB.

    L'ordre de ``urls`` est strictement préservé. Un délai de politesse
    ``delay`` est observé entre deux téléchargements (jamais avant le premier).

    Args:
        urls: URLs des morceaux (issues de :func:`fetch_chapter_image_urls`).
        session: Session HTTP ; créée (puis fermée) avec les en-têtes
            obligatoires si ``None``.
        delay: Pause (secondes) entre deux requêtes.
        max_retries: Nouvelles tentatives par image (cf. :func:`get_with_retry`).
        referer: ``Referer`` propre au site (remplace celui de la session).

    Returns:
        Liste d'images PIL en mode ``RGB``, une par URL.

    Raises:
        ScraperError: Si une image ne peut pas être décodée.
        requests.HTTPError: Si un téléchargement échoue définitivement (statut
            d'erreur, ou erreur réseau — y compris coupure en cours de lecture
            du corps — persistante après ``max_retries`` nouvelles tentatives).
    """
    images: list[Image.Image] = []
    total = len(urls)
    extra = {"headers": {"Referer": referer}} if referer else {}
    with _session_scope(session) as http:
        for index, url in enumerate(urls):
            if index > 0 and delay > 0:
                _sleep(delay)
            logger.info("Image %d/%d: %s", index + 1, total, url)
            progress.step("download", "Téléchargement des images", index, total)
            response = get_with_retry(http, url, max_retries=max_retries, **extra)
            img = _decode_image(response.content, url)
            logger.debug(
                "Image %d/%d decodee: %dx%d", index + 1, total, img.width, img.height
            )
            images.append(img)
    logger.info("%d image(s) telechargee(s)", len(images))
    return images


# --- Stitching ----------------------------------------------------------------------


def stitch_webtoon_pages(
    images: list[Image.Image], target_width: int | None = None
) -> Image.Image:
    """Assemble les morceaux en une bande verticale continue (``np.vstack``).

    Les largeurs sont d'abord normalisées : tout morceau dont la largeur
    diffère de ``target_width`` est redimensionné (LANCZOS, ratio conservé).
    Par défaut ``target_width`` est la largeur la plus fréquente parmi les
    morceaux (en cas d'égalité, la première rencontrée).

    Args:
        images: Morceaux PIL dans l'ordre de lecture (tout mode accepté).
        target_width: Largeur cible en pixels ; auto si ``None``.

    Returns:
        Bande PIL en mode ``RGB``, hauteur = somme des hauteurs (redimensionnées).

    Raises:
        ValueError: Liste vide ou largeur cible invalide.
    """
    if not images:
        raise ValueError("Aucune image a assembler (liste vide)")

    if target_width is None:
        widths = Counter(img.width for img in images)
        target_width = widths.most_common(1)[0][0]
    target_width = int(target_width)
    if target_width <= 0:
        raise ValueError(f"Largeur cible invalide: {target_width}")

    arrays: list[np.ndarray] = []
    for index, img in enumerate(images):
        img = _to_rgb(img)
        if img.width != target_width:
            new_height = max(1, round(img.height * target_width / img.width))
            logger.info(
                "Morceau %d redimensionne %dx%d -> %dx%d",
                index, img.width, img.height, target_width, new_height,
            )
            img = img.resize((target_width, new_height), Image.Resampling.LANCZOS)
        arrays.append(np.asarray(img, dtype=np.uint8))

    strip_array = np.vstack(arrays)
    # Les copies par morceau ne servent plus : les libérer AVANT la copie que
    # Pillow effectue dans ``fromarray`` (mode RGB non mappable) divise le pic
    # mémoire par deux sur les très longs chapitres.
    del arrays
    strip = Image.fromarray(strip_array, mode="RGB")
    del strip_array
    logger.info("Bande assemblee: %dx%d (%d morceaux)", strip.width, strip.height, len(images))
    return strip


# --- Orchestration ------------------------------------------------------------------


def scrape_chapter(
    url: str,
    session: requests.Session | None = None,
    *,
    delay: float = 0.15,
    language: str | None = DEFAULT_SITE_LANGUAGE,
) -> tuple[Image.Image, ChapterMeta]:
    """Pipeline complet : page → URLs → téléchargement → bande continue.

    Args:
        url: URL du viewer Webtoons ou d'un chapitre Asura Scans.
        session: Session HTTP optionnelle (en-têtes obligatoires garantis) ;
            créée puis fermée localement si ``None``.
        delay: Pause (secondes) entre deux téléchargements d'image.
        language: langue du site imposée à l'URL (``"en"`` par défaut, ``None``
            pour garder l'URL telle quelle).

    Returns:
        ``(bande PIL RGB, ChapterMeta)`` avec ``chunk_sizes`` renseigné.
    """
    on_asura = asura.is_asura_url(url)
    with _session_scope(session) as http:
        meta = fetch_chapter_image_urls(url, session=http, language=language)
        images = download_images(meta.image_urls, session=http, delay=delay, referer=asura.REFERER if on_asura else None)
    if on_asura:
        images = _drop_credit_banners(images, meta)
    meta.chunk_sizes = [(img.width, img.height) for img in images]
    strip = stitch_webtoon_pages(images)
    # Les morceaux PIL ne sont plus utiles une fois la bande assemblée.
    del images
    return strip, meta


def _drop_credit_banners(images: list[Image.Image], meta: ChapterMeta) -> list[Image.Image]:
    """Retire les bannières de crédits d'Asura (images et URLs) ; jamais toutes les pages."""
    portrait = [img.width for img in images if img.height >= img.width]
    if len(images) < 2 or not portrait:
        return images
    page_width = Counter(portrait).most_common(1)[0][0]
    keep = [i for i, img in enumerate(images) if not asura.is_credits_banner(img.size, page_width)]
    if len(keep) == len(images) or not keep:
        return images
    for i in sorted(set(range(len(images))) - set(keep)):
        logger.info("Banniere de credits ecartee (%dx%d) : %s", images[i].width, images[i].height, meta.image_urls[i])
    meta.image_urls = [meta.image_urls[i] for i in keep]
    return [images[i] for i in keep]


def save_strip(strip: Image.Image, path: str | Path) -> Path:
    """Enregistre la bande sur disque (format déduit de l'extension, PNG conseillé).

    Args:
        strip: Bande PIL.
        path: Chemin de destination ; les dossiers parents sont créés.

    Returns:
        Le chemin (``Path``) du fichier écrit.

    Raises:
        ValueError: Bande trop grande pour le format demandé (voir
            :data:`FORMAT_MAX_DIMENSION` : 65 500 px en JPEG, 16 383 px en WebP).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    limit = FORMAT_MAX_DIMENSION.get(path.suffix.lower())
    if limit is not None and max(strip.width, strip.height) > limit:
        raise ValueError(
            f"Bande de {strip.width}x{strip.height}px trop grande pour le format "
            f"{path.suffix.lower()} (max {limit}px) : utiliser .png"
        )
    strip.save(path)
    logger.info("Bande enregistree: %s", path)
    return path


# --- Rapport & CLI ------------------------------------------------------------------


def ascii_safe(text: str) -> str:
    """Rend une chaîne sûre pour la console Windows (cp1252) : ASCII pur.

    Les caractères non ASCII (guillemets typographiques, alphabets non
    latins...) sont remplacés par ``?``.
    """
    return text.encode("ascii", "replace").decode("ascii")


def format_summary(
    meta: ChapterMeta,
    strip: Image.Image,
    saved: Path | None = None,
    elapsed: float | None = None,
) -> str:
    """Construit le bloc récapitulatif ASCII affiché par la CLI et le script local.

    Args:
        meta: Métadonnées du chapitre.
        strip: Bande assemblée.
        saved: Chemin du fichier écrit (ligne omise si ``None``).
        elapsed: Durée totale en secondes (ligne omise si ``None``).

    Returns:
        Texte multi-lignes, ASCII pur, encadré de lignes ``=``.
    """
    lines = [
        "=" * 60,
        f"Series     : {meta.series_title}",
        f"Episode    : {meta.episode_title}",
        f"IDs        : title_no={meta.title_no} episode_no={meta.episode_no}",
        f"Final URL  : {meta.final_url}",
        f"Chunks     : {meta.n_chunks}",
        f"Strip size : {strip.width}x{strip.height} px (mode {strip.mode})",
    ]
    if saved is not None:
        lines.append(f"Saved to   : {saved}")
    if elapsed is not None:
        lines.append(f"Elapsed    : {elapsed:.1f}s")
    lines.append("=" * 60)
    return ascii_safe("\n".join(lines))


def _run_cli(
    url: str,
    out: Path,
    delay: float,
    verbose: bool,
    language: str | None = DEFAULT_SITE_LANGUAGE,
) -> int:
    """Corps de la commande CLI ; retourne le code de sortie."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    started = time.perf_counter()
    try:
        strip, meta = scrape_chapter(url, delay=delay, language=language)
        saved = save_strip(strip, out)
    except Exception as exc:  # noqa: BLE001 - la CLI doit afficher toute erreur
        logger.exception("Echec du scraping")
        print(ascii_safe(f"FAILED: {type(exc).__name__}: {exc}"))
        return 1

    print(format_summary(meta, strip, saved, time.perf_counter() - started))
    return 0


#: Application Typer ; une seule commande => appelable directement sans sous-commande.
app = typer.Typer(add_completion=False, help="Scraper & Stitcher Webtoons (Module 1).")


@app.command()
def scrape(
    url: Annotated[str, typer.Argument(help="URL du viewer Webtoons.")],
    out: Annotated[
        Path, typer.Option("--out", "-o", help="Fichier image de sortie (PNG conseille).")
    ] = Path("output/strip.png"),
    delay: Annotated[
        float, typer.Option("--delay", help="Pause (s) entre deux images.")
    ] = 0.15,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Logs de niveau DEBUG.")
    ] = False,
    site_language: Annotated[
        str,
        typer.Option(
            "--site-language",
            help="Langue du site Webtoons imposee a l'URL (defaut en ; vide = inchangee).",
        ),
    ] = DEFAULT_SITE_LANGUAGE,
) -> None:
    """Telecharge un chapitre, assemble la bande continue et l'enregistre."""
    raise typer.Exit(code=_run_cli(url, out, delay, verbose, site_language or None))


def main() -> None:
    """Point d'entrée CLI : ``python -m src.modules.scraper <url> [--out ...]``."""
    app()


if __name__ == "__main__":
    main()
