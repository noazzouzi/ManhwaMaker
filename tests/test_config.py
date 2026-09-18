"""Tests de la configuration transverse : langues par défaut et clé Gemini, URL Webtoons."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.modules import scraper
from src.modules.analyzer import DEFAULT_LANGUAGE
from src.modules.scraper import fetch_chapter_image_urls, normalize_webtoon_url
from src.utils.config import (
    DEFAULT_NARRATION_LANGUAGE,
    DEFAULT_SITE_LANGUAGE,
    gemini_api_keys,
    gemini_key_hint,
    load_dotenv,
    load_gemini_api_key,
)

FR_URL = "https://www.webtoons.com/fr/fantasy/tower-of-god/season-1-ep-1/viewer?title_no=95&episode_no=1"
EN_URL = "https://www.webtoons.com/en/fantasy/tower-of-god/season-1-ep-1/viewer?title_no=95&episode_no=1"
HTML = (
    '<html><body><div id="_imageList">'
    '<img class="_images" data-url="https://cdn.example/1.jpg?type=q90" '
    'src="https://webtoons-static.pstatic.net/image/bg_transparency.png">'
    "</div></body></html>"
)


def test_defaults_are_english() -> None:
    assert DEFAULT_SITE_LANGUAGE == "en"
    assert DEFAULT_NARRATION_LANGUAGE == "en"
    assert DEFAULT_LANGUAGE == "en"


def test_load_gemini_api_key_priority(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    missing = tmp_path / "missing.key"
    assert load_gemini_api_key(key_file=missing) is None

    key_file = tmp_path / ".gemini_key"
    key_file.write_bytes(b"\xef\xbb\xbf  file-key \r\n")  # BOM + espaces + CRLF
    assert load_gemini_api_key(key_file=key_file) == "file-key"

    empty = tmp_path / "empty.key"
    empty.write_text("   \n", encoding="utf-8")
    assert load_gemini_api_key(key_file=empty) is None

    monkeypatch.setenv("GOOGLE_API_KEY", " google-key ")
    assert load_gemini_api_key(key_file=key_file) == "google-key"
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    assert load_gemini_api_key(key_file=key_file) == "gemini-key"
    assert load_gemini_api_key(" explicit-key ", key_file=key_file) == "explicit-key"
    assert load_gemini_api_key("   ", key_file=key_file) == "gemini-key"

    hint = gemini_key_hint()
    hint.encode("ascii")
    assert "GEMINI_API_KEYS" in hint and "GEMINI_API_KEY" in hint and ".gemini_key" in hint


def test_gemini_api_keys_sources_dedup_and_dotenv(tmp_path, monkeypatch) -> None:
    for var in ("GEMINI_API_KEYS", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    missing = tmp_path / "missing.key"
    assert gemini_api_keys(key_file=missing) == []
    # Fichier .gemini_key multi-cles (lignes, virgules, doublons).
    key_file = tmp_path / ".gemini_key"
    key_file.write_text("k1\nk2, k3\n\nk1\n", encoding="utf-8")
    assert gemini_api_keys(key_file=key_file) == ["k1", "k2", "k3"]
    assert load_gemini_api_key(key_file=key_file) == "k1"
    # Une cle unique dans l'environnement prime sur le fichier ; GEMINI_API_KEYS prime sur tout.
    monkeypatch.setenv("GEMINI_API_KEY", "single")
    assert gemini_api_keys(key_file=key_file) == ["single"]
    monkeypatch.setenv("GEMINI_API_KEYS", " a ; b,c c ")
    assert gemini_api_keys(key_file=key_file) == ["a", "b", "c"]  # separateurs mixtes, doublon retire
    # Explicite : chaine ou liste, prioritaire.
    assert gemini_api_keys("x, y", key_file=key_file) == ["x", "y"]
    assert gemini_api_keys(["y", " ", "z", "y"], key_file=key_file) == ["y", "z"]
    # .env : charge sans ecraser l'environnement existant.
    dotenv = tmp_path / ".env"
    dotenv.write_text('# cles\nexport GEMINI_API_KEYS="d1,d2"  \nOTHER=\'v\'\nBROKEN LINE\nEMPTY=\nCOMMENTED=val # note\n', encoding="utf-8")
    parsed = load_dotenv(dotenv)
    assert parsed == {"GEMINI_API_KEYS": "d1,d2", "OTHER": "v", "EMPTY": "", "COMMENTED": "val"}
    assert gemini_api_keys(key_file=key_file, dotenv=dotenv) == ["a", "b", "c"]  # l'environnement prime
    monkeypatch.delenv("GEMINI_API_KEYS")
    assert gemini_api_keys(key_file=key_file, dotenv=dotenv) == ["d1", "d2"]
    assert load_dotenv(tmp_path / "nowhere.env") == {}


def test_normalize_webtoon_url() -> None:
    assert normalize_webtoon_url(FR_URL) == EN_URL
    assert normalize_webtoon_url(EN_URL) == EN_URL
    assert normalize_webtoon_url(FR_URL, "ES") == FR_URL.replace("/fr/", "/es/")
    assert normalize_webtoon_url(
        "https://www.webtoons.com/zh-hant/x/y/z/viewer?title_no=1&episode_no=2"
    ) == "https://www.webtoons.com/en/x/y/z/viewer?title_no=1&episode_no=2"
    # Desactivation explicite.
    assert normalize_webtoon_url(FR_URL, language=None) == FR_URL
    assert normalize_webtoon_url(FR_URL, language="") == FR_URL
    # URL sans segment de langue reconnaissable : inchangee.
    assert normalize_webtoon_url("https://www.webtoons.com/") == "https://www.webtoons.com/"
    assert normalize_webtoon_url("https://x/viewer?title_no=1") == "https://x/viewer?title_no=1"
    assert normalize_webtoon_url("https://www.webtoons.com/fantasy/x") == "https://www.webtoons.com/fantasy/x"


def test_fetch_chapter_normalizes_language_by_default(monkeypatch) -> None:
    calls: list[str] = []

    def fake_get(_http, url, **_kwargs):
        calls.append(url)
        return SimpleNamespace(url=url, text=HTML)

    monkeypatch.setattr(scraper, "get_with_retry", fake_get)
    session = SimpleNamespace(headers={})

    meta = fetch_chapter_image_urls(FR_URL, session=session)
    assert calls == [EN_URL]
    assert meta.url == EN_URL and meta.final_url == EN_URL
    assert meta.image_urls == ["https://cdn.example/1.jpg?type=q90"]

    fetch_chapter_image_urls(FR_URL, session=session, language=None)
    assert calls[-1] == FR_URL
    fetch_chapter_image_urls(FR_URL, session=session, language="es")
    assert calls[-1] == FR_URL.replace("/fr/", "/es/")
    with pytest.raises(scraper.ScraperError):
        monkeypatch.setattr(
            scraper, "get_with_retry", lambda _h, url, **_k: SimpleNamespace(url=url, text="<html></html>")
        )
        fetch_chapter_image_urls(EN_URL, session=session)
