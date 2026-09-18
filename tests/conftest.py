"""Fixtures globales : aucun test ne dort réellement ni ne lit le ``.env`` de la machine."""

from __future__ import annotations

import pytest

from src.modules import analyzer as analyzer_mod
from src.utils import config as config_mod


@pytest.fixture(autouse=True)
def _no_real_sleep_and_isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    # Le delai force entre lots (BATCH_DELAY_S) et les backoffs ne doivent pas ralentir les tests ;
    # les tests qui verifient les attentes remplacent eux-memes ``_sleep`` par un enregistreur.
    monkeypatch.setattr(analyzer_mod, "_sleep", lambda s: None)
    # Les cles de la machine (.env, GEMINI_API_KEYS) ne doivent pas influencer les tests.
    monkeypatch.setattr(config_mod, "DOTENV_FILE", tmp_path / "absent.env")
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.delenv("GEMINI_MODEL_CASCADE", raising=False)
