"""Asura Scans (``asurascans.com``) : page de chapitre, liste des chapitres, bannière de crédits.

Fonctions pures (sans réseau) : :mod:`src.modules.scraper` télécharge les pages et
aiguille vers ce module dès que l'URL est celle d'Asura (:func:`is_asura_url`).

Structure du site (relevée le 25/09/2026, HTML rendu côté serveur, sans protection) :

- chapitre ``/comics/<serie>-<id>/chapter/<n>`` : une balise ``img[data-page-index]`` par
  page, ``src`` sur ``cdn.asurascans.com`` (WebP, 800 px de large, 7 000 à 16 000 px de
  haut) ; titres dans le JSON-LD ``BreadcrumbList`` ;
- série ``/comics/<serie>-<id>`` : tous les chapitres listés sur une seule page, numéros
  décimaux compris (``/chapter/74.5``, chapitres bonus) ;
- la première page d'un chapitre est la bannière de crédits de l'équipe (paysage
  1200x800) : :func:`is_credits_banner` la repère pour qu'elle ne soit jamais montée ;
- le CDN répond sans ``Referer`` particulier.

Le suffixe ``-<id>`` (8 caractères hexadécimaux) du slug de série est écarté des clés et
des noms de dossiers : Asura le faisait tourner sur son ancien domaine (``asuracomic.net``),
le nom de la série, lui, reste.
"""

from __future__ import annotations

import json
import logging
import re
from urllib.parse import urljoin, urlparse, urlunparse

from bs4 import BeautifulSoup

from src.models.chapter import ChapterMeta

logger = logging.getLogger(__name__)

__all__ = [
    "HOSTS", "REFERER", "IMAGE_SELECTOR", "is_asura_url", "parse_number", "parse_url", "series_url",
    "chapter_url", "is_series_url", "parse_chapter_html", "parse_chapter_links", "is_credits_banner",
]

#: Domaines d'Asura Scans (``asuracomic.net`` : ancien domaine).
HOSTS: tuple[str, ...] = ("asurascans.com", "asuracomic.net")
REFERER: str = "https://asurascans.com/"
#: Pages du lecteur, dans l'ordre de ``data-page-index``.
IMAGE_SELECTOR: str = "img[data-page-index]"

_PATH = re.compile(r"^/(?:comics|series)/(?P<series>[^/]+)(?:/chapter/(?P<no>\d+(?:\.\d+)?))?/?$")
_SERIES_ID = re.compile(r"-[0-9a-f]{8}$")
_TITLE = re.compile(r"^(?P<series>.+?)\s+Chapter\s+(?P<no>\d+(?:\.\d+)?)", re.IGNORECASE)


def is_asura_url(url: str) -> bool:
    """Vrai pour une URL d'Asura Scans (série ou chapitre, sous-domaine ``www`` compris)."""
    host = (urlparse(url).hostname or "").lower()
    return any(host == h or host.endswith("." + h) for h in HOSTS)


def parse_number(text: str) -> int | float | None:
    """Numéro de chapitre : ``"12"`` → ``12``, ``"74.5"`` → ``74.5`` ; ``None`` sinon."""
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return int(value) if value.is_integer() else value


def parse_url(url: str) -> tuple[str | None, int | float | None]:
    """``(slug de la série sans son identifiant, numéro de chapitre)`` d'une URL Asura."""
    match = _PATH.match(urlparse(url).path)
    if not match:
        return None, None
    series = _SERIES_ID.sub("", match["series"]) or match["series"]
    return series, parse_number(match["no"]) if match["no"] else None


def _series_segment(url: str) -> str:
    match = _PATH.match(urlparse(url).path)
    if not match:
        raise ValueError(f"URL Asura Scans non reconnue : {url}")
    return match["series"]


def series_url(url: str) -> str:
    """URL de la page de la série (liste des chapitres) depuis une URL de série ou de chapitre."""
    parsed = urlparse(url)
    return urlunparse(parsed._replace(path=f"/comics/{_series_segment(url)}", query="", fragment=""))


def chapter_url(url: str, number: int | float) -> str:
    """URL du chapitre ``number`` de la même série."""
    if number <= 0:
        raise ValueError("le numero de chapitre doit etre > 0")
    parsed = urlparse(url)
    return urlunparse(parsed._replace(path=f"/comics/{_series_segment(url)}/chapter/{number:g}", query="", fragment=""))


def is_series_url(url: str) -> bool:
    """Vrai pour une page de série (pas de ``/chapter/<n>``)."""
    match = _PATH.match(urlparse(url).path)
    return bool(match) and not match["no"]


def _breadcrumb(soup: BeautifulSoup) -> list[str]:
    """Noms du fil d'Ariane JSON-LD (``Home``, ``Comics``, série, ``Chapter N``)."""
    for script in soup.select('script[type="application/ld+json"]'):
        try:
            data = json.loads(script.string or "")
        except ValueError:
            continue
        if isinstance(data, dict) and data.get("@type") == "BreadcrumbList":
            items = sorted(data.get("itemListElement") or [], key=lambda item: item.get("position", 0))
            return [str(item.get("name") or "").strip() for item in items]
    return []


def _titles(soup: BeautifulSoup) -> tuple[str, str]:
    """``(série, chapitre)`` : fil d'Ariane JSON-LD, sinon ``og:title`` / ``<title>``."""
    crumbs = _breadcrumb(soup)
    if len(crumbs) >= 4 and crumbs[-1]:
        return crumbs[-2], crumbs[-1]
    og = soup.select_one('meta[property="og:title"]')
    raw = str(og.get("content") or "") if og is not None else ""
    if not raw and soup.title is not None:
        raw = soup.title.get_text()
    match = _TITLE.match(" ".join(raw.split()))
    if match:
        return match["series"].strip(), f"Chapter {match['no']}"
    return raw.split(" - ")[0].split(" | ")[0].strip(), ""


def parse_chapter_html(html: str, url: str, final_url: str | None = None) -> ChapterMeta:
    """Métadonnées et URLs des pages d'un chapitre Asura (``chunk_sizes`` vide)."""
    final_url = final_url or url
    soup = BeautifulSoup(html, "lxml")
    series_title, episode_title = _titles(soup)
    _, number = parse_url(final_url)
    if number is None:
        _, number = parse_url(url)
    if not episode_title and number is not None:
        episode_title = f"Chapter {number:g}"

    pages: list[tuple[int, str]] = []
    for position, tag in enumerate(soup.select(IMAGE_SELECTOR)):
        src = str(tag.get("src") or tag.get("data-src") or "").strip()
        if not src:
            logger.warning("Page #%d sans URL exploitable, ignoree", position)
            continue
        index = parse_number(str(tag.get("data-page-index")))
        pages.append((int(index) if index is not None else position, urljoin(final_url, src)))
    pages.sort(key=lambda page: page[0])

    meta = ChapterMeta(
        url=url, final_url=final_url, series_title=series_title, episode_title=episode_title,
        title_no=None, episode_no=number, image_urls=[src for _, src in pages],
    )
    logger.info("Chapitre Asura analyse: %s", meta.summary())
    return meta


def parse_chapter_links(html: str, base_url: str) -> dict[int | float, str]:
    """``{numéro: URL du chapitre}`` trouvés dans une page de série Asura."""
    soup = BeautifulSoup(html, "lxml")
    found: dict[int | float, str] = {}
    for anchor in soup.select("a[href*='/chapter/']"):
        href = urljoin(base_url, str(anchor.get("href") or ""))
        _, number = parse_url(href)
        if number is not None and number not in found:
            found[number] = href
    return found


def is_credits_banner(size: tuple[int, int], page_width: int) -> bool:
    """Vrai pour la bannière de crédits : image paysage, plus large que les pages du chapitre."""
    width, height = size
    return width > height and width != page_width
