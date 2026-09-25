"""Asura Scans : parsing des pages, aiguillage du scraper, lots et noms de dossiers."""

from __future__ import annotations

from io import BytesIO

from PIL import Image

from src.modules import asura
from src.modules import batch_processor as bp
from src.modules.scraper import (
    chapter_ids,
    discover_episodes,
    episode_url,
    fetch_chapter_image_urls,
    is_series_url,
    scrape_chapter,
)
from src.pipeline import slug_from_url
from tests.test_scraper import FakeResponse, FakeSession

SERIES = "https://asurascans.com/comics/bad-born-blood-05c7df14"
CHAPTER = SERIES + "/chapter/1"
CDN = "https://cdn.asurascans.com/asura-images/chapters/bad-born-blood/1/"

# Extrait de la vraie page (25/09/2026) : fil d'Ariane JSON-LD, couverture hors lecteur, pages.
CHAPTER_HTML = f"""<!DOCTYPE html><html lang="en"><head>
<title>Bad Born Blood Chapter 1 - Read Online | Asura Scans</title>
<meta property="og:title" content="Bad Born Blood Chapter 1 - Read Online | Asura Scans">
<script type="application/ld+json">{{"@context":"https://schema.org","@type":"BreadcrumbList","itemListElement":[
{{"@type":"ListItem","position":1,"name":"Home","item":"https://asurascans.com"}},
{{"@type":"ListItem","position":3,"name":"Bad Born Blood","item":"{SERIES}"}},
{{"@type":"ListItem","position":2,"name":"Comics","item":"https://asurascans.com/browse"}},
{{"@type":"ListItem","position":4,"name":"Chapter 1","item":"{CHAPTER}"}}]}}</script>
</head><body>
<img src="https://cdn.asurascans.com/asura-images/covers/bad-born-blood.3008f6-400.webp" alt="Bad Born Blood">
<img src="{CDN}002.webp?v=1" alt="Page 2" data-page-index="1" class="w-full block">
<img src="{CDN}001.webp?v=1" alt="Page 1" data-page-index="0" class="w-full block">
<img data-src="{CDN}003.webp?v=1" alt="Page 3" data-page-index="2" class="w-full block">
<a href="/comics/bad-born-blood-05c7df14/chapter/2">Next</a>
</body></html>"""

SERIES_HTML = """<html><body>
<a href="/comics/bad-born-blood-05c7df14/chapter/2">Chapter 2</a>
<a href="/comics/bad-born-blood-05c7df14/chapter/1">Chapter 1</a>
<a href="/comics/bad-born-blood-05c7df14/chapter/1">Start reading</a>
<a href="/comics/bad-born-blood-05c7df14/chapter/74.5">Chapter 74.5</a>
<a href="/comics/other-series-aaaaaaaa">Other</a>
</body></html>"""


def png(width: int, height: int, color: tuple[int, int, int] = (200, 30, 30)) -> bytes:
    out = BytesIO()
    Image.new("RGB", (width, height), color).save(out, format="PNG")
    return out.getvalue()


# --- URLs ------------------------------------------------------------------------------------
def test_urls_are_recognised_and_parsed() -> None:
    assert asura.is_asura_url(CHAPTER) and asura.is_asura_url("https://www.asurascans.com/comics/x-05c7df14")
    assert asura.is_asura_url("https://asuracomic.net/series/x-05c7df14/chapter/3")
    assert not asura.is_asura_url("https://www.webtoons.com/en/action/x/list?title_no=1")
    # L'identifiant de fin de slug change de temps en temps : il n'entre pas dans la clé.
    assert asura.parse_url(CHAPTER) == ("bad-born-blood", 1)
    assert asura.parse_url(SERIES + "/chapter/74.5") == ("bad-born-blood", 74.5)
    assert asura.parse_url(SERIES) == ("bad-born-blood", None)
    assert asura.parse_url("https://asurascans.com/browse") == (None, None)
    assert asura.series_url(CHAPTER + "?x=1") == SERIES
    assert asura.chapter_url(SERIES, 74.5) == SERIES + "/chapter/74.5"
    assert asura.chapter_url(CHAPTER, 12) == SERIES + "/chapter/12"
    assert asura.is_series_url(SERIES) and not asura.is_series_url(CHAPTER)


def test_scraper_helpers_route_asura_urls() -> None:
    assert chapter_ids(CHAPTER) == ("asura:bad-born-blood", 1)
    assert chapter_ids("https://asurascans.com/comics/bad-born-blood-ffffffff/chapter/2")[0] == "asura:bad-born-blood"
    assert chapter_ids("https://www.webtoons.com/en/a/b/c/viewer?title_no=95&episode_no=3") == ("t95", 3)
    assert is_series_url(SERIES) and not is_series_url(CHAPTER)
    assert is_series_url("https://www.webtoons.com/en/a/b/list?title_no=95")
    assert episode_url(CHAPTER, 5) == SERIES + "/chapter/5"
    assert slug_from_url(CHAPTER) == "bad-born-blood_ep1"
    assert slug_from_url(SERIES + "/chapter/74.5") == "bad-born-blood_ep74.5"


# --- Pages ------------------------------------------------------------------------------------
def test_parse_chapter_reads_breadcrumb_and_pages_in_reader_order() -> None:
    meta = asura.parse_chapter_html(CHAPTER_HTML, CHAPTER)
    assert (meta.series_title, meta.episode_title) == ("Bad Born Blood", "Chapter 1")
    assert (meta.title_no, meta.episode_no) == (None, 1)
    assert meta.image_urls == [CDN + "001.webp?v=1", CDN + "002.webp?v=1", CDN + "003.webp?v=1"]  # couverture exclue


def test_parse_chapter_falls_back_to_og_title() -> None:
    html = ('<html><head><meta property="og:title" content="Solo Max-Level Newbie Chapter 74.5 - Read Online | Asura Scans">'
            '</head><body><img src="/p/1.webp" data-page-index="0"></body></html>')
    meta = asura.parse_chapter_html(html, "https://asurascans.com/comics/solo-12345678/chapter/74.5")
    assert (meta.series_title, meta.episode_title, meta.episode_no) == ("Solo Max-Level Newbie", "Chapter 74.5", 74.5)
    assert meta.image_urls == ["https://asurascans.com/p/1.webp"]


def test_parse_series_lists_every_chapter_once() -> None:
    links = asura.parse_chapter_links(SERIES_HTML, SERIES)
    assert links == {2: SERIES + "/chapter/2", 1: SERIES + "/chapter/1", 74.5: SERIES + "/chapter/74.5"}


def test_credits_banner_is_the_landscape_page() -> None:
    assert asura.is_credits_banner((1200, 800), 800)
    assert not asura.is_credits_banner((800, 15652), 800)
    assert not asura.is_credits_banner((800, 600), 800)  # case large mais de la largeur du chapitre


# --- Réseau (fausse session) ------------------------------------------------------------------
def test_fetch_sends_asura_referer_and_never_rewrites_the_url() -> None:
    session = FakeSession(lambda _u: FakeResponse(text=CHAPTER_HTML, url=CHAPTER))
    meta = fetch_chapter_image_urls(CHAPTER, session=session)
    assert session.calls[0][0] == CHAPTER and session.calls[0][1]["headers"]["Referer"] == asura.REFERER
    assert meta.n_chunks == 3 and meta.series_title == "Bad Born Blood"


def test_scrape_drops_the_credits_banner_and_stitches_the_rest() -> None:
    html = CHAPTER_HTML.replace('<img data-src="' + CDN + '003.webp?v=1" alt="Page 3" data-page-index="2" class="w-full block">', "")
    images = {CDN + "001.webp?v=1": png(1200, 800), CDN + "002.webp?v=1": png(800, 1500, (10, 200, 10))}

    def route(url: str) -> FakeResponse:
        return FakeResponse(text=html, url=CHAPTER) if url == CHAPTER else FakeResponse(content=images[url], url=url)

    session = FakeSession(route)
    strip, meta = scrape_chapter(CHAPTER, session=session, delay=0)
    assert strip.size == (800, 1500) and strip.getpixel((5, 5)) == (10, 200, 10)
    assert meta.image_urls == [CDN + "002.webp?v=1"] and meta.chunk_sizes == [(800, 1500)]
    assert all(kwargs["headers"]["Referer"] == asura.REFERER for _, kwargs in session.calls)


def test_discover_reads_the_series_page() -> None:
    session = FakeSession(lambda _u: FakeResponse(text=SERIES_HTML, url=SERIES))
    assert list(discover_episodes(CHAPTER, session=session)) == [1, 2, 74.5]
    assert session.calls[0][0] == SERIES


# --- Lots --------------------------------------------------------------------------------------
def test_batch_range_includes_bonus_chapters() -> None:
    found = {n: asura.chapter_url(SERIES, n) for n in (1, 2, 2.5, 3, 4)}
    urls = bp.resolve_chapter_urls(CHAPTER, start_chapter=2, end_chapter=3, discover=lambda u: found)
    assert urls == [found[2], found[2.5], found[3]]
    assert bp.resolve_chapter_urls(SERIES, discover=lambda u: found) == list(found.values())  # série entière
    assert bp.resolve_chapter_urls(CHAPTER, discover=lambda u: found) == [CHAPTER]  # un seul chapitre


def test_batch_orders_asura_chapters_of_one_series() -> None:
    urls = [asura.chapter_url(SERIES, 3), asura.chapter_url(SERIES, 2.5), asura.chapter_url(SERIES, 2)]
    assert bp.series_predecessors(urls) == {urls[0]: urls[1], urls[1]: urls[2]}
