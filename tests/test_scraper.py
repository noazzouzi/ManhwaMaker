"""Tests unitaires du Module 1 (scraper / stitcher) — SANS réseau.

Les requêtes HTTP sont simulées soit par une fausse session (duck typing sur
``session.get`` / ``session.headers``), soit par une vraie ``requests.Session``
sur laquelle est monté un adaptateur factice (pour vérifier les en-têtes
réellement envoyés). Les images sont générées en mémoire.
Lancer : ``.\\.venv\\Scripts\\python.exe -m pytest tests/test_scraper.py -q``.
"""

from __future__ import annotations

from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import requests
from PIL import Image
from pydantic import ValidationError

import src.utils.http as http_mod
from src.models.chapter import ChapterMeta
from src.modules import scraper
from src.modules.scraper import (
    ScraperError,
    ascii_safe,
    download_images,
    fetch_chapter_image_urls,
    format_summary,
    parse_chapter_html,
    parse_ids_from_url,
    save_strip,
    scrape_chapter,
    stitch_webtoon_pages,
)
from src.utils.http import (
    DEFAULT_HEADERS,
    DEFAULT_USER_AGENT,
    MAX_RETRY_AFTER,
    WEBTOONS_REFERER,
    build_session,
    ensure_mandatory_headers,
    get_with_retry,
)

# --- Fixtures HTML ------------------------------------------------------------------

PAGE_URL = "https://www.webtoons.com/en/fantasy/tower-of-god/season-1-ep-1/viewer?title_no=95&episode_no=1"
FINAL_URL = "https://www.webtoons.com/en/fantasy/tower-of-god/season-1-ep-0/viewer?title_no=95&episode_no=1"
PLACEHOLDER = "https://webtoons-static.pstatic.net/image/bg_transparency.png"

CHAPTER_HTML = f"""
<!DOCTYPE html>
<html><head>
  <meta property="og:title" content="Tower of God - [Season 1] Ep. 0">
  <title>Tower of God | WEBTOON</title>
</head><body>
  <div class="subj_info">
    <a class="subj" href="/en/fantasy/tower-of-god/list?title_no=95">Tower of God</a>
  </div>
  <div class="subj_episode_wrap">
    <span class="tx">#1</span>
    <h1 class="subj_episode" title="[Season 1] Ep. 0">[Season 1] Ep. 0</h1>
  </div>
  <div class="viewer_img _img_viewer_area" id="_imageList">
    <img class="_images" src="{PLACEHOLDER}"
         data-url="https://webtoon-phinf.pstatic.net/a/chunk_1.jpg?type=q90" width="700" height="1140">
    <img class="_images" src="{PLACEHOLDER}"
         data-url="https://webtoon-phinf.pstatic.net/a/chunk_2.jpg?type=q90">
    <img class="_images" src="https://webtoon-phinf.pstatic.net/a/chunk_3.jpg?type=q90">
    <img class="_images" src="{PLACEHOLDER}">
    <img class="other" data-url="https://webtoon-phinf.pstatic.net/a/not_a_chunk.jpg">
  </div>
  <div class="recommend"><img class="_images" data-url="https://example.com/outside.jpg"></div>
</body></html>
"""

EXPECTED_URLS = [
    "https://webtoon-phinf.pstatic.net/a/chunk_1.jpg?type=q90",
    "https://webtoon-phinf.pstatic.net/a/chunk_2.jpg?type=q90",
    "https://webtoon-phinf.pstatic.net/a/chunk_3.jpg?type=q90",
]

NO_LIST_HTML = """
<html><head><meta property="og:title" content="Solo Leveling - Episode 12"></head><body>
  <img class="_images" data-url="//webtoon-phinf.pstatic.net/b/1.jpg?type=q90">
  <img class="_images" data-url="https://webtoon-phinf.pstatic.net/b/2.jpg?type=q90">
</body></html>
"""

TITLE_ONLY_HTML = """
<html><head><title>Tower of God | WEBTOON</title></head><body>
  <div id="_imageList"><img class="_images" data-url="https://webtoon-phinf.pstatic.net/c/1.jpg"></div>
</body></html>
"""


# --- Faux objets HTTP ---------------------------------------------------------------


class FakeResponse:
    """Réponse minimale imitant ``requests.Response``."""

    def __init__(
        self,
        *,
        status_code: int = 200,
        content: bytes = b"",
        text: str = "",
        url: str = "",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.content = content
        self.text = text
        self.url = url
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)  # type: ignore[arg-type]


class FakeSession:
    """Fausse session : ``router`` est une liste séquentielle ou un callable(url)."""

    def __init__(self, router: list[Any] | Callable[[str], Any]) -> None:
        self.headers: dict[str, str] = {}
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.closed = 0
        self._router = router

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append((url, kwargs))
        if callable(self._router):
            result = self._router(url)
        else:
            result = self._router.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def close(self) -> None:
        self.closed += 1


class CapturingAdapter(requests.adapters.BaseAdapter):
    """Adaptateur ``requests`` factice : capture les requêtes préparées (en-têtes réels)."""

    def __init__(self, router: Callable[[str], tuple[bytes, str]]) -> None:
        super().__init__()
        self.sent: list[requests.PreparedRequest] = []
        self._router = router

    def send(  # type: ignore[override]
        self, request: requests.PreparedRequest, **kwargs: Any
    ) -> requests.Response:
        self.sent.append(request)
        body, content_type = self._router(str(request.url))
        response = requests.Response()
        response.status_code = 200
        response._content = body
        response.encoding = "utf-8"
        response.url = str(request.url)
        response.request = request
        response.headers["Content-Type"] = content_type
        return response

    def close(self) -> None:
        pass


def png_bytes(width: int, height: int, color: Any, mode: str = "RGB") -> bytes:
    """Encode une image unie en PNG (en mémoire)."""
    buf = BytesIO()
    Image.new(mode, (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


def palette_png_with_transparency(width: int, height: int) -> bytes:
    """PNG en mode ``P`` dont l'index 0 (utilisé partout) est déclaré transparent."""
    img = Image.new("P", (width, height), 0)
    img.putpalette([0, 0, 0, 255, 0, 0] + [0] * (256 * 3 - 6))
    buf = BytesIO()
    img.save(buf, format="PNG", transparency=0)
    return buf.getvalue()


@pytest.fixture
def sleep_calls(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[float]]:
    """Neutralise les attentes tout en les enregistrant (retries / délai de politesse)."""
    calls: dict[str, list[float]] = {"http": [], "scraper": []}
    monkeypatch.setattr(http_mod, "_sleep", lambda s: calls["http"].append(s))
    monkeypatch.setattr(scraper, "_sleep", lambda s: calls["scraper"].append(s))
    return calls


@pytest.fixture(autouse=True)
def _no_real_sleep(sleep_calls: dict[str, list[float]]) -> None:
    """Aucun test ne doit dormir réellement."""


# --- Parsing HTML -------------------------------------------------------------------


def test_parse_extracts_data_urls_in_order_and_ignores_placeholder() -> None:
    meta = parse_chapter_html(CHAPTER_HTML, url=PAGE_URL, final_url=FINAL_URL)
    assert meta.image_urls == EXPECTED_URLS
    # Le suffixe ?type=q90 est conservé tel quel.
    assert all(u.endswith("?type=q90") for u in meta.image_urls)
    # Le placeholder transparent n'est jamais retenu, ni les balises hors #_imageList.
    assert all("bg_transparency" not in u for u in meta.image_urls)
    assert all("outside" not in u for u in meta.image_urls)


def test_parse_titles_and_ids() -> None:
    meta = parse_chapter_html(CHAPTER_HTML, url=PAGE_URL, final_url=FINAL_URL)
    assert isinstance(meta, ChapterMeta)
    assert meta.series_title == "Tower of God"
    assert meta.episode_title == "[Season 1] Ep. 0"
    assert meta.title_no == 95
    assert meta.episode_no == 1
    assert meta.url == PAGE_URL
    assert meta.final_url == FINAL_URL
    assert meta.chunk_sizes == []
    assert meta.n_chunks == 3


def test_parse_fallback_selector_and_og_title() -> None:
    url = "https://www.webtoons.com/en/action/solo-leveling/episode-12/viewer"
    meta = parse_chapter_html(NO_LIST_HTML, url=url)
    assert meta.image_urls == [
        "https://webtoon-phinf.pstatic.net/b/1.jpg?type=q90",
        "https://webtoon-phinf.pstatic.net/b/2.jpg?type=q90",
    ]
    assert meta.series_title == "Solo Leveling"
    assert meta.episode_title == "Episode 12"
    assert meta.title_no is None
    assert meta.episode_no is None
    assert meta.final_url == url


def test_parse_title_only_fallback_strips_site_suffix_and_never_duplicates_series() -> None:
    meta = parse_chapter_html(TITLE_ONLY_HTML, url=PAGE_URL)
    assert meta.series_title == "Tower of God"
    assert meta.episode_title != meta.series_title
    assert "WEBTOON" not in meta.episode_title
    # Repli ultime : libelle construit a partir de episode_no (issu de l'URL).
    assert meta.episode_title == "Episode 1"


def test_parse_title_only_fallback_without_ids_leaves_episode_empty() -> None:
    meta = parse_chapter_html(TITLE_ONLY_HTML, url="https://www.webtoons.com/en/x/viewer")
    assert meta.series_title == "Tower of God"
    assert meta.episode_title == ""


def test_episode_no_falls_back_to_span_tx_when_missing_from_url() -> None:
    meta = parse_chapter_html(CHAPTER_HTML, url="https://www.webtoons.com/en/x/y/z/viewer")
    assert meta.title_no is None
    assert meta.episode_no == 1


def test_parse_ids_from_url() -> None:
    assert parse_ids_from_url(PAGE_URL) == (95, 1)
    assert parse_ids_from_url("https://www.webtoons.com/en/x/viewer?title_no=abc") == (None, None)
    assert parse_ids_from_url("https://www.webtoons.com/") == (None, None)


def test_parse_no_images_gives_empty_list() -> None:
    meta = parse_chapter_html("<html><body><p>nothing</p></body></html>", url=PAGE_URL)
    assert meta.image_urls == []
    assert meta.series_title == ""


# --- Modele ChapterMeta -------------------------------------------------------------


def test_chapter_meta_requires_ids_and_image_urls() -> None:
    # Contrat : seul chunk_sizes a une valeur par defaut.
    with pytest.raises(ValidationError):
        ChapterMeta(url="u", final_url="f", series_title="s", episode_title="e")  # type: ignore[call-arg]
    meta = ChapterMeta(
        url="u", final_url="f", series_title="s", episode_title="e",
        title_no=None, episode_no=None, image_urls=[],
    )
    assert meta.chunk_sizes == []
    assert meta.n_chunks == 0


def test_chapter_meta_summary_is_pure_ascii() -> None:
    meta = ChapterMeta(
        url="u", final_url="f",
        series_title="Solo Leveling — “Arise”",
        episode_title="하늘 Ep. 1",
        title_no=1, episode_no=2, image_urls=["a", "b"],
    )
    text = meta.summary()
    text.encode("ascii")  # ne doit pas lever
    assert "Solo Leveling" in text
    assert "chunks=2" in text


# --- Session & headers --------------------------------------------------------------


def test_build_session_has_mandatory_headers() -> None:
    session = build_session()
    assert session.headers["User-Agent"] == DEFAULT_USER_AGENT
    assert session.headers["Referer"] == WEBTOONS_REFERER
    assert session.headers["Referer"] == "https://www.webtoons.com/"
    assert "Accept" in session.headers and "Accept-Language" in session.headers
    assert "Mozilla/5.0" in DEFAULT_HEADERS["User-Agent"]


def test_build_session_merges_custom_headers() -> None:
    session = build_session({"X-Test": "1", "Accept-Language": "fr-FR"})
    assert session.headers["X-Test"] == "1"
    assert session.headers["Accept-Language"] == "fr-FR"
    assert session.headers["Referer"] == WEBTOONS_REFERER  # jamais perdu


def test_ensure_mandatory_headers_replaces_python_requests_user_agent() -> None:
    # Une requests.Session() nue porte "python-requests/x.y" : ce n'est PAS un
    # navigateur desktop, il doit etre remplace (PRD garde-fou n1).
    session = requests.Session()
    assert str(session.headers["User-Agent"]).startswith("python-requests")
    ensure_mandatory_headers(session)
    assert session.headers["User-Agent"] == DEFAULT_USER_AGENT
    assert session.headers["Referer"] == WEBTOONS_REFERER
    session.close()


def test_ensure_mandatory_headers_keeps_custom_user_agent() -> None:
    session = build_session({"User-Agent": "MyBot/1.0", "Referer": "https://example.org/"})
    ensure_mandatory_headers(session)
    assert session.headers["User-Agent"] == "MyBot/1.0"
    assert session.headers["Referer"] == "https://example.org/"
    session.close()


def test_ensure_mandatory_headers_none_builds_session() -> None:
    session = ensure_mandatory_headers(None)
    assert isinstance(session, requests.Session)
    assert session.headers["User-Agent"] == DEFAULT_USER_AGENT
    session.close()


def test_download_injects_headers_on_bare_session() -> None:
    session = FakeSession(lambda _u: FakeResponse(content=png_bytes(10, 10, "red")))
    download_images(["https://x/1.jpg"], session=session, delay=0)
    assert session.headers["Referer"] == WEBTOONS_REFERER
    assert session.headers["User-Agent"] == DEFAULT_USER_AGENT


def test_real_bare_session_sends_desktop_headers_on_page_and_images() -> None:
    """Verifie les en-tetes reellement emis (requete preparee) par une vraie Session."""
    image = png_bytes(8, 8, "red")

    def router(url: str) -> tuple[bytes, str]:
        if url == PAGE_URL:
            return CHAPTER_HTML.encode("utf-8"), "text/html; charset=UTF-8"
        return image, "image/jpeg"

    adapter = CapturingAdapter(router)
    session = requests.Session()  # nue : UA python-requests, pas de Referer
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    try:
        meta = fetch_chapter_image_urls(PAGE_URL, session=session)
        images = download_images(meta.image_urls, session=session, delay=0)
    finally:
        session.close()

    assert len(images) == 3
    assert [str(r.url) for r in adapter.sent] == [PAGE_URL, *EXPECTED_URLS]
    for prepared in adapter.sent:
        assert prepared.headers["User-Agent"] == DEFAULT_USER_AGENT
        assert prepared.headers["Referer"] == WEBTOONS_REFERER


# --- get_with_retry -----------------------------------------------------------------


def test_get_with_retry_recovers_from_transient_errors(sleep_calls: dict[str, list[float]]) -> None:
    session = FakeSession(
        [
            FakeResponse(status_code=503),
            requests.ConnectionError("boom"),
            FakeResponse(status_code=200, text="ok"),
        ]
    )
    resp = get_with_retry(session, "https://x/", max_retries=3)
    assert resp.text == "ok"
    assert len(session.calls) == 3
    assert session.calls[0][1]["timeout"] == 20
    assert sleep_calls["http"] == [0.5, 1.0]


def test_get_with_retry_gives_up_after_max_retries(sleep_calls: dict[str, list[float]]) -> None:
    session = FakeSession(lambda _u: FakeResponse(status_code=503))
    with pytest.raises(requests.HTTPError):
        get_with_retry(session, "https://x/", max_retries=2)
    assert len(session.calls) == 3  # 1 tentative + 2 retries
    assert sleep_calls["http"] == [0.5, 1.0]  # backoff exponentiel, pas d'attente finale


def test_get_with_retry_backoff_sequence(sleep_calls: dict[str, list[float]]) -> None:
    session = FakeSession(lambda _u: FakeResponse(status_code=500))
    with pytest.raises(requests.HTTPError):
        get_with_retry(session, "https://x/", max_retries=3, backoff=0.25)
    assert sleep_calls["http"] == [0.25, 0.5, 1.0]


@pytest.mark.parametrize("status", [501, 505, 507, 520, 522, 599])
def test_get_with_retry_retries_every_5xx(status: int) -> None:
    session = FakeSession(lambda _u, s=status: FakeResponse(status_code=s))
    with pytest.raises(requests.HTTPError):
        get_with_retry(session, "https://x/", max_retries=2)
    assert len(session.calls) == 3


def test_get_with_retry_does_not_retry_403_or_404() -> None:
    for status in (403, 404):
        session = FakeSession(lambda _u, s=status: FakeResponse(status_code=s))
        with pytest.raises(requests.HTTPError):
            get_with_retry(session, "https://x/", max_retries=3)
        assert len(session.calls) == 1


def test_get_with_retry_honours_retry_after_and_caps_it(sleep_calls: dict[str, list[float]]) -> None:
    session = FakeSession(
        [
            FakeResponse(status_code=429, headers={"Retry-After": "3"}),
            FakeResponse(status_code=429, headers={"Retry-After": "1000"}),
            FakeResponse(status_code=429, headers={"Retry-After": "not-a-number"}),
            FakeResponse(status_code=200, text="ok"),
        ]
    )
    resp = get_with_retry(session, "https://x/", max_retries=3, backoff=0.5)
    assert resp.text == "ok"
    assert len(session.calls) == 4
    # 3 s (Retry-After > backoff), 30 s (plafond), 2.0 s (backoff 0.5 * 2**2).
    assert sleep_calls["http"] == [3.0, MAX_RETRY_AFTER, 2.0]


def test_get_with_retry_retries_mid_body_failures() -> None:
    session = FakeSession(
        [
            requests.exceptions.ChunkedEncodingError("cut"),
            requests.exceptions.ContentDecodingError("bad gzip"),
            requests.Timeout("slow"),
            FakeResponse(status_code=200, text="ok"),
        ]
    )
    resp = get_with_retry(session, "https://x/", max_retries=3)
    assert resp.text == "ok"
    assert len(session.calls) == 4


def test_get_with_retry_persistent_network_error_raises_http_error() -> None:
    # Contrat : HTTPError apres epuisement des tentatives, meme pour une erreur reseau.
    session = FakeSession(lambda _u: requests.ConnectionError("down"))
    with pytest.raises(requests.HTTPError) as excinfo:
        get_with_retry(session, "https://x/", max_retries=2)
    assert len(session.calls) == 3
    assert isinstance(excinfo.value.__cause__, requests.ConnectionError)
    assert "down" in str(excinfo.value)


# --- fetch / download avec fausse session -------------------------------------------


def test_fetch_chapter_image_urls_with_fake_session() -> None:
    session = FakeSession(lambda _u: FakeResponse(text=CHAPTER_HTML, url=FINAL_URL))
    meta = fetch_chapter_image_urls(PAGE_URL, session=session)
    assert meta.image_urls == EXPECTED_URLS
    assert meta.final_url == FINAL_URL
    assert meta.url == PAGE_URL
    assert session.calls[0][0] == PAGE_URL
    assert session.headers["Referer"] == WEBTOONS_REFERER
    assert session.closed == 0  # session de l'appelant : jamais fermee


def test_fetch_chapter_raises_when_no_images() -> None:
    session = FakeSession(lambda _u: FakeResponse(text="<html></html>", url=PAGE_URL))
    with pytest.raises(ScraperError):
        fetch_chapter_image_urls(PAGE_URL, session=session)


def test_fetch_chapter_closes_session_it_created(monkeypatch: pytest.MonkeyPatch) -> None:
    session = FakeSession(lambda _u: FakeResponse(text=CHAPTER_HTML, url=FINAL_URL))
    monkeypatch.setattr(http_mod, "build_session", lambda headers=None: session)
    meta = fetch_chapter_image_urls(PAGE_URL)  # session=None -> creee localement
    assert meta.image_urls == EXPECTED_URLS
    assert session.closed == 1


def test_download_images_preserves_order_and_converts_to_rgb(
    sleep_calls: dict[str, list[float]],
) -> None:
    payloads = {
        "https://x/1.png": png_bytes(70, 114, "red"),
        "https://x/2.png": png_bytes(70, 114, 128, mode="L"),
        "https://x/3.png": png_bytes(80, 85, (0, 0, 255, 255), mode="RGBA"),
        "https://x/4.png": palette_png_with_transparency(20, 20),
    }
    session = FakeSession(lambda u: FakeResponse(content=payloads[u]))
    images = download_images(list(payloads), session=session, delay=0.15)
    assert [img.mode for img in images] == ["RGB"] * 4
    assert [img.size for img in images] == [(70, 114), (70, 114), (80, 85), (20, 20)]
    assert images[0].getpixel((0, 0)) == (255, 0, 0)
    assert images[2].getpixel((0, 0)) == (0, 0, 255)
    assert images[3].getpixel((0, 0)) == (255, 255, 255)  # P transparent -> fond blanc
    assert [c[0] for c in session.calls] == list(payloads)
    # Delai de politesse : entre deux images seulement, jamais avant la premiere.
    assert sleep_calls["scraper"] == [0.15, 0.15, 0.15]


def test_download_images_zero_delay_never_sleeps(sleep_calls: dict[str, list[float]]) -> None:
    session = FakeSession(lambda _u: FakeResponse(content=png_bytes(4, 4, "red")))
    download_images(["https://x/1", "https://x/2"], session=session, delay=0)
    assert sleep_calls["scraper"] == []


def test_download_images_rejects_non_image_payload() -> None:
    session = FakeSession(lambda _u: FakeResponse(content=b"<html>403 page</html>"))
    with pytest.raises(ScraperError):
        download_images(["https://x/1.jpg"], session=session, delay=0)


def test_download_images_persistent_network_error_surfaces_as_http_error() -> None:
    session = FakeSession(lambda _u: requests.exceptions.ChunkedEncodingError("cut"))
    with pytest.raises(requests.HTTPError):
        download_images(["https://x/1.jpg"], session=session, delay=0, max_retries=1)
    assert len(session.calls) == 2


# --- Stitching ----------------------------------------------------------------------


def test_stitch_mixed_widths_normalizes_to_modal_width() -> None:
    # Cas reel Tower of God ep.1 : 3 morceaux 700x1140 + 1 morceau final 800x850.
    images = [Image.new("RGB", (700, 1140), "white") for _ in range(3)]
    images.append(Image.new("RGB", (800, 850), "black"))
    strip = stitch_webtoon_pages(images)
    expected_last_h = round(850 * 700 / 800)  # 743.75 -> 744
    assert expected_last_h == 744
    assert strip.mode == "RGB"
    assert strip.width == 700
    assert strip.height == 3 * 1140 + expected_last_h == 4164
    # Le dernier morceau (noir) est bien en bas, redimensionne.
    assert strip.getpixel((350, strip.height - 1)) == (0, 0, 0)
    assert strip.getpixel((350, 3 * 1140 + 5)) == (0, 0, 0)
    assert strip.getpixel((350, 0)) == (255, 255, 255)
    assert strip.getpixel((350, 3 * 1140 - 1)) == (255, 255, 255)


def test_stitch_explicit_target_width() -> None:
    images = [Image.new("RGB", (100, 50), "red"), Image.new("RGB", (100, 50), "blue")]
    strip = stitch_webtoon_pages(images, target_width=50)
    assert strip.size == (50, 50)
    assert strip.getpixel((10, 5)) == (255, 0, 0)
    assert strip.getpixel((10, 45)) == (0, 0, 255)


def test_stitch_preserves_order_and_pixels_without_resize() -> None:
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    images = [Image.new("RGB", (20, 10), c) for c in colors]
    strip = stitch_webtoon_pages(images)
    arr = np.asarray(strip)
    assert arr.shape == (30, 20, 3)
    for i, color in enumerate(colors):
        block = arr[i * 10 : (i + 1) * 10]
        assert (block == np.array(color, dtype=np.uint8)).all()


def test_stitch_converts_modes_to_rgb() -> None:
    images = [Image.new("L", (10, 10), 200), Image.new("RGBA", (10, 10), (0, 0, 0, 0))]
    strip = stitch_webtoon_pages(images)
    assert strip.mode == "RGB"
    assert strip.getpixel((0, 0)) == (200, 200, 200)
    assert strip.getpixel((0, 15)) == (255, 255, 255)  # transparent -> fond blanc


def test_stitch_empty_list_raises() -> None:
    with pytest.raises(ValueError):
        stitch_webtoon_pages([])


# --- Pipeline complet & sauvegarde --------------------------------------------------


def _chapter_router(payloads: dict[str, bytes]) -> Callable[[str], FakeResponse]:
    def router(url: str) -> FakeResponse:
        if url == PAGE_URL:
            return FakeResponse(text=CHAPTER_HTML, url=FINAL_URL)
        return FakeResponse(content=payloads[url])

    return router


def test_scrape_chapter_end_to_end_with_fake_session(tmp_path: Path) -> None:
    payloads = {
        EXPECTED_URLS[0]: png_bytes(70, 114, "red"),
        EXPECTED_URLS[1]: png_bytes(70, 114, "green"),
        EXPECTED_URLS[2]: png_bytes(80, 85, "blue"),
    }
    session = FakeSession(_chapter_router(payloads))
    strip, meta = scrape_chapter(PAGE_URL, session=session, delay=0)

    assert meta.chunk_sizes == [(70, 114), (70, 114), (80, 85)]
    assert strip.size == (70, 2 * 114 + round(85 * 70 / 80))
    assert [c[0] for c in session.calls] == [PAGE_URL, *EXPECTED_URLS]
    assert session.headers["Referer"] == WEBTOONS_REFERER
    assert session.closed == 0  # session fournie par l'appelant : conservee ouverte

    out = save_strip(strip, tmp_path / "nested" / "strip.png")
    assert out.exists()
    with Image.open(out) as reloaded:
        assert reloaded.size == strip.size


def test_scrape_chapter_closes_session_it_created(monkeypatch: pytest.MonkeyPatch) -> None:
    payloads = {u: png_bytes(10, 10, "red") for u in EXPECTED_URLS}
    session = FakeSession(_chapter_router(payloads))
    monkeypatch.setattr(http_mod, "build_session", lambda headers=None: session)
    strip, meta = scrape_chapter(PAGE_URL, delay=0)  # session=None
    assert strip.size == (10, 30)
    assert meta.n_chunks == 3
    assert session.closed == 1  # fermee une seule fois, par scrape_chapter


def test_save_strip_refuses_jpeg_too_tall(tmp_path: Path) -> None:
    class TallStub:
        height = 70_000
        width = 800

    with pytest.raises(ValueError):
        save_strip(TallStub(), tmp_path / "strip.jpg")  # type: ignore[arg-type]


def test_save_strip_refuses_webp_too_tall_with_clear_message(tmp_path: Path) -> None:
    class TallStub:
        height = 20_000
        width = 800

    with pytest.raises(ValueError, match=r"\.webp.*16383.*\.png"):
        save_strip(TallStub(), tmp_path / "strip.webp")  # type: ignore[arg-type]
    assert not (tmp_path / "strip.webp").exists()


def test_save_strip_png_has_no_dimension_limit(tmp_path: Path) -> None:
    strip = Image.new("RGB", (4, 17_000), "white")  # > limite WebP, PNG accepte
    out = save_strip(strip, tmp_path / "tall.png")
    with Image.open(out) as reloaded:
        assert reloaded.size == (4, 17_000)


# --- Rapport ASCII ------------------------------------------------------------------


def test_ascii_safe_replaces_non_ascii() -> None:
    assert ascii_safe("Tower of God — “Ep”") == "Tower of God ? ?Ep?"
    assert ascii_safe("plain ascii") == "plain ascii"
    ascii_safe("하늘").encode("ascii")  # ne doit pas lever


def test_format_summary_is_ascii_and_complete() -> None:
    meta = ChapterMeta(
        url=PAGE_URL, final_url=FINAL_URL,
        series_title="Tower of God —", episode_title="[Season 1] Ep. 0",
        title_no=95, episode_no=1, image_urls=EXPECTED_URLS,
    )
    strip = Image.new("RGB", (700, 4164), "white")
    text = format_summary(meta, strip, Path("output") / "strip.png", 3.14159)
    text.encode("ascii")  # ne doit pas lever
    assert "Series     : Tower of God ?" in text
    assert "Episode    : [Season 1] Ep. 0" in text
    assert "title_no=95 episode_no=1" in text
    assert "Chunks     : 3" in text
    assert "Strip size : 700x4164 px (mode RGB)" in text
    assert "strip.png" in text
    assert "Elapsed    : 3.1s" in text
    # Lignes optionnelles omises quand non fournies.
    short = format_summary(meta, strip)
    assert "Saved to" not in short and "Elapsed" not in short


# --- Series : URL d'episodes et decouverte de la liste -----------------------------------
def test_episode_url_and_series_list_url() -> None:
    from src.modules.scraper import episode_url, series_list_url

    viewer = "https://www.webtoons.com/en/action/serie/ep-0-intro/viewer?title_no=3915&episode_no=1"
    listing = "https://www.webtoons.com/en/action/serie/list?title_no=3915"
    assert episode_url(viewer, 7) == "https://www.webtoons.com/en/action/serie/ep-0-intro/viewer?title_no=3915&episode_no=7"
    assert episode_url(listing, 2) == "https://www.webtoons.com/en/action/serie/episode/viewer?title_no=3915&episode_no=2"
    assert series_list_url(viewer) == listing and series_list_url(listing) == listing
    assert series_list_url(listing + "&page=3#top") == listing
    with pytest.raises(ValueError):
        episode_url(viewer, 0)


def test_parse_and_discover_episodes_paginated() -> None:
    from src.modules.scraper import discover_episodes, parse_episode_links

    def page(numbers: list[int]) -> str:
        items = "".join(
            f'<li><a href="/en/action/serie/ep-{n}/viewer?title_no=3915&amp;episode_no={n}"><span class="tx">#{n}</span></a></li>'
            for n in numbers
        )
        return f'<html><body><ul id="_listUl">{items}</ul><a href="/en/action/serie/list?title_no=3915&amp;page=2">2</a></body></html>'

    base = "https://www.webtoons.com/en/action/serie/list?title_no=3915"
    links = parse_episode_links(page([4, 3]), base)
    assert links == {
        4: "https://www.webtoons.com/en/action/serie/ep-4/viewer?title_no=3915&episode_no=4",
        3: "https://www.webtoons.com/en/action/serie/ep-3/viewer?title_no=3915&episode_no=3",
    }
    pages = {base: page([4, 3]), base + "&page=2": page([2, 1]), base + "&page=3": page([2, 1])}  # page 3 = repetition
    session = FakeSession(lambda url: FakeResponse(text=pages.get(url, "<html></html>"), url=url))
    episodes = discover_episodes("https://www.webtoons.com/fr/action/serie/ep-1/viewer?title_no=3915&episode_no=1", session=session)
    assert list(episodes) == [1, 2, 3, 4] and episodes[1].endswith("episode_no=1")
    assert [u for u, _ in session.calls] == [base, base + "&page=2", base + "&page=3"]  # arret des que rien de neuf
    assert discover_episodes(base, session=FakeSession(lambda url: FakeResponse(text="<html></html>", url=url))) == {}
