"""Memoire de serie : ce qu'un chapitre transmet au suivant (``src.modules.series_memory``).

Aucun appel reseau ni Gemini : tout se joue sur des fichiers ecrits dans ``tmp_path``.
Le defaut que ces tests protegent est concret : sans memoire, le heros change de nom
d'un episode a l'autre et le spectateur ne suit plus.
"""

from __future__ import annotations

import json

import pytest

from src.models.chapter import ChapterMeta
from src.models.scene import ChapterAnalysis, CharacterCard, Scene
from src.modules.series_memory import (
    CHARACTERS_FILE,
    MAX_SHEET_CARDS,
    MAX_WHO_CHARS,
    chapter_tail,
    load_series_context,
    merge_cards,
    save_chapter_sheet,
    series_key,
)


def _meta(episode: int | None, title_no: int | None = 9674) -> ChapterMeta:
    return ChapterMeta(
        url=f"https://www.webtoons.com/en/x/y/viewer?title_no={title_no}&episode_no={episode}",
        final_url="https://www.webtoons.com/en/x/y/viewer",
        series_title="Mirror World", episode_title=f"Ep. {episode}",
        title_no=title_no, episode_no=episode, image_urls=["a"],
    )


def _analysis(*, characters: list[dict], last: str = "He draws the blade.") -> ChapterAnalysis:
    return ChapterAnalysis(
        series_title="Mirror World", episode_title="Ep.", model="m", language="en", n_panels=2,
        scenes=[
            Scene(index=0, panel_ids=[0], narration="The gate opens.", emotion="calm"),
            Scene(index=1, panel_ids=[1], narration=last, emotion="tension"),
            Scene(index=2, panel_ids=[2], narration="Credits roll.", emotion="neutral", is_filler=True),
        ],
        characters=[CharacterCard(**c) for c in characters],
    )


def _write(root, episode: int, characters: list[dict], *, last: str = "He draws the blade.", title_no: int = 9674):
    out_dir = root / f"chapter_t{title_no}_ep{episode}"   # une serie par dossier, comme en vrai
    out_dir.mkdir(parents=True, exist_ok=True)
    return save_chapter_sheet(_analysis(characters=characters, last=last), out_dir, _meta(episode, title_no))


# --- Ecriture -----------------------------------------------------------------------
def test_sheet_is_written_next_to_scenes_json(tmp_path) -> None:
    path = _write(tmp_path, 1, [{"name": "Eden", "also_called": ["the revenant"], "who": "The hero."}])
    assert path is not None and path.name == CHARACTERS_FILE
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["series"] == "t9674" and data["episode_no"] == 1
    assert data["characters"] == [{"name": "Eden", "also_called": ["the revenant"], "who": "The hero."}]
    # Le rappel de fin ignore le remplissage : c'est le dernier paragraphe narre.
    assert data["tail"] == "He draws the blade."
    assert not list(path.parent.glob("*.tmp"))          # ecriture atomique, rien ne traine


def test_a_chapter_without_a_number_writes_nothing(tmp_path) -> None:
    """Sans numero d'episode, impossible de dire quels chapitres precedent celui-ci :
    ecrire la fiche exposerait un chapitre a se relire lui-meme."""
    out_dir = tmp_path / "loose"
    out_dir.mkdir()
    assert save_chapter_sheet(_analysis(characters=[]), out_dir, _meta(None)) is None
    assert save_chapter_sheet(_analysis(characters=[]), out_dir, None) is None
    assert save_chapter_sheet(_analysis(characters=[]), out_dir, _meta(1, title_no=None)) is not None
    assert not (out_dir / CHARACTERS_FILE).with_suffix(".json.tmp").exists()


def test_series_key_falls_back_on_the_title(tmp_path) -> None:
    assert series_key(_meta(1)) == "t9674"
    assert series_key(_meta(1, title_no=None)) == "s-mirror-world"   # repli sur le titre
    assert series_key(None) == ""


def test_the_who_and_the_tail_are_trimmed(tmp_path) -> None:
    long_who = "x" * (MAX_WHO_CHARS + 80)
    path = _write(tmp_path, 1, [{"name": "Eden", "also_called": [], "who": long_who}], last="word " * 200)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data["characters"][0]["who"]) <= MAX_WHO_CHARS + 3   # « ... »
    assert data["characters"][0]["who"].endswith("...")
    assert len(data["tail"]) <= 323
    assert chapter_tail(_analysis(characters=[], last="")) == "The gate opens."


# --- Fusion -------------------------------------------------------------------------
def test_merge_keeps_the_first_spelling_and_unions_the_aliases() -> None:
    """Le premier chapitre ou un personnage apparait fixe son nom : c'est celui que le
    modele devra reemployer dans tous les suivants."""
    sheets = [
        {"episode_no": 1, "characters": [{"name": "Eden", "also_called": ["the revenant"], "who": "The hero."}]},
        {"episode_no": 2, "characters": [{"name": "eden", "also_called": ["the boy"], "who": "Someone else."}]},
        {"episode_no": 3, "characters": [{"name": "Hugh", "also_called": [], "who": "The rival."}]},
    ]
    cards = merge_cards(sheets)
    assert [c.name for c in cards] == ["Eden", "Hugh"]
    assert cards[0].who == "The hero."                       # la premiere definition tient
    assert cards[0].also_called == ["the revenant", "the boy"]


def test_a_character_returning_under_a_known_alias_joins_his_own_card() -> None:
    """« the revenant » etait un alias d'Eden : un chapitre qui l'emploie comme nom ne doit
    pas creer un second personnage."""
    sheets = [
        {"episode_no": 1, "characters": [{"name": "Eden", "also_called": ["the revenant"], "who": "The hero."}]},
        {"episode_no": 2, "characters": [{"name": "the revenant", "also_called": ["Ed"], "who": "?"}]},
    ]
    cards = merge_cards(sheets)
    assert [c.name for c in cards] == ["Eden"]
    assert "Ed" in cards[0].also_called


# --- Lecture ------------------------------------------------------------------------
def test_current_and_later_episodes_are_excluded(tmp_path) -> None:
    """Couvre ``--redo`` : relancer le chapitre 2 ne doit pas lui faire relire sa propre
    fiche, sinon le modele se cite lui-meme au lieu de repartir du chapitre 1."""
    _write(tmp_path, 1, [{"name": "Eden", "also_called": [], "who": "The hero."}])
    _write(tmp_path, 2, [{"name": "Hugh", "also_called": [], "who": "The rival."}])
    _write(tmp_path, 3, [{"name": "Nora", "also_called": [], "who": "The captain."}])

    cards, _ = load_series_context(tmp_path, key="t9674", before_episode=2)
    assert [c.name for c in cards] == ["Eden"]
    cards, _ = load_series_context(tmp_path, key="t9674", before_episode=1)
    assert cards == []                                        # premier episode : rien a heriter
    cards, _ = load_series_context(tmp_path, key="t9674", before_episode=4)
    assert [c.name for c in cards] == ["Eden", "Hugh", "Nora"]


def test_tail_comes_from_the_highest_earlier_episode(tmp_path) -> None:
    """Les fiches peuvent etre ecrites dans le desordre (lot parallele, reprise) : c'est le
    numero d'episode qui fait foi, pas la date du fichier."""
    _write(tmp_path, 3, [{"name": "C", "also_called": [], "who": "?"}], last="Three ends.")
    _write(tmp_path, 1, [{"name": "A", "also_called": [], "who": "?"}], last="One ends.")
    _write(tmp_path, 2, [{"name": "B", "also_called": [], "who": "?"}], last="Two ends.")
    cards, tail = load_series_context(tmp_path, key="t9674", before_episode=4)
    assert tail == "Three ends."
    assert [c.name for c in cards] == ["A", "B", "C"]         # fusionnes dans l'ordre du recit
    assert load_series_context(tmp_path, key="t9674", before_episode=3)[1] == "Two ends."


def test_foreign_series_and_corrupt_files_are_ignored(tmp_path, caplog) -> None:
    """Le dossier de sortie est partage par toutes les series, et un lot tue en plein vol
    peut laisser un fichier tronque : ni l'un ni l'autre ne doit faire echouer un chapitre."""
    _write(tmp_path, 1, [{"name": "Eden", "also_called": [], "who": "The hero."}])
    _write(tmp_path, 1, [{"name": "Intrus", "also_called": [], "who": "Autre serie."}], title_no=42)

    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / CHARACTERS_FILE).write_text("{ oops", encoding="utf-8")
    shapeless = tmp_path / "shapeless"
    shapeless.mkdir()
    (shapeless / CHARACTERS_FILE).write_text('{"series": "t9674", "characters": "nope"}', encoding="utf-8")

    import logging

    with caplog.at_level(logging.WARNING, logger="src.modules.series_memory"):
        cards, _ = load_series_context(tmp_path, key="t9674", before_episode=9)
    assert [c.name for c in cards] == ["Eden"]
    assert "illisible" in caplog.text and "mal formee" in caplog.text


def test_a_missing_folder_is_not_an_error(tmp_path) -> None:
    assert load_series_context(tmp_path / "nowhere", key="t9674", before_episode=2) == ([], "")
    assert load_series_context(tmp_path, key="", before_episode=2) == ([], "")
    assert load_series_context(tmp_path, key="t9674", before_episode=None) == ([], "")


def test_the_sheet_is_capped_on_the_most_recurring_characters(tmp_path) -> None:
    """Au-dela du plafond il faut choisir. Un personnage vu dans dix chapitres compte plus
    qu'un figurant vu une fois : c'est justement son nom qui doit rester stable."""
    recurring = [{"name": "Eden", "also_called": [], "who": "The hero."}]
    for episode in range(1, 9):
        extras = [{"name": f"Extra{episode}-{k}", "also_called": [], "who": "?"} for k in range(5)]
        _write(tmp_path, episode, recurring + extras)

    cards, _ = load_series_context(tmp_path, key="t9674", before_episode=99)
    assert len(cards) == MAX_SHEET_CARDS
    assert cards[0].name == "Eden"                            # le recurrent passe en tete
    assert sum(1 for c in cards if c.name.startswith("Extra")) == MAX_SHEET_CARDS - 1


def test_the_sheet_renders_for_the_prompt(tmp_path) -> None:
    """Bout en bout : ce qui est ecrit sur le disque doit arriver lisible dans le prompt."""
    from src.modules.analyzer import character_sheet

    _write(tmp_path, 1, [{"name": "Eden", "also_called": ["the revenant"], "who": "The hero."}])
    cards, tail = load_series_context(tmp_path, key="t9674", before_episode=2)
    rendered = character_sheet(cards)
    assert "- Eden (also called the revenant): The hero." in rendered
    assert tail == "He draws the blade."
    rendered.encode("ascii")


def test_concurrent_chapters_never_touch_the_same_file(tmp_path) -> None:
    """La fusion se fait a la lecture : chaque chapitre n'ecrit que sa propre fiche, donc
    deux chapitres en parallele ne peuvent pas s'ecraser l'un l'autre."""
    paths = {_write(tmp_path, episode, [{"name": f"P{episode}", "also_called": [], "who": "?"}])
             for episode in (1, 2, 3)}
    assert len(paths) == 3
    assert len({p.parent for p in paths}) == 3
