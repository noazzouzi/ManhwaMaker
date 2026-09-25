"""Fiabilite des reponses Gemini : plafond de jetons, troncature, reparation JSON, numerotation.

Ces cas viennent tous d'echecs OBSERVES sur de vrais appels pendant la mise au point du
master prompt : reponse coupee en plein JSON avec ``finish_reason=STOP``, cles sans
guillemets, valeur parasite, et une reponse citant 125 numeros pour un chapitre qui en
compte 38.
"""

from __future__ import annotations

import json

import pytest
from google.genai import errors, types

from src.models.scene import BeatBatch, KeyframeChoice, RecapParagraph
from src.modules.analyzer import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    TRUNCATION_RETRIES,
    GeminiAnalyzer,
    INLINE_PAYLOAD_LADDER,
    InvalidResponseError,
    RUNAWAY_INT,
    TruncatedResponseError,
    _extract_json,
    _repair_json,
    normalize_keyframes,
    normalize_recap,
    response_data,
)
from src.utils import gemini_manager as gm
from tests.gemini_fakes import FakeClient, make_response, panels


def _response(text: str, *, finish: str = "STOP"):
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=[types.Part(text=text)]),
                finish_reason=finish,
            )
        ]
    )


# --- Plafond de jetons ----------------------------------------------------------------
def test_every_config_sets_the_output_ceiling() -> None:
    """Sans plafond, une reponse longue est coupee alors que finish_reason annonce STOP."""
    analyzer = GeminiAnalyzer(client=object())
    configs = [
        analyzer.config_for("x", BeatBatch),
        analyzer.beats_config,
        analyzer.script_config([]),
        analyzer.keyframes_config,
    ]
    assert [c.max_output_tokens for c in configs] == [DEFAULT_MAX_OUTPUT_TOKENS] * 4


def test_output_ceiling_is_configurable_and_validated() -> None:
    assert GeminiAnalyzer(client=object(), max_output_tokens=4096).config_for("x", BeatBatch).max_output_tokens == 4096
    with pytest.raises(ValueError, match="max_output_tokens"):
        GeminiAnalyzer(client=object(), max_output_tokens=0)


# --- Troncature -----------------------------------------------------------------------
def test_truncated_by_max_tokens_is_named_as_such() -> None:
    """Le piege : la reponse arrive AVEC du texte, donc sans ce test elle passerait pour du JSON invalide."""
    with pytest.raises(TruncatedResponseError, match="MAX_TOKENS"):
        response_data(_response('{"paragraphs":[{"text":"abc"', finish="MAX_TOKENS"), BeatBatch)


def test_truncated_with_finish_stop_is_detected_structurally() -> None:
    """Cas reellement observe : JSON coupe net et finish_reason=STOP malgre tout."""
    cut = '{"paragraphs":[' + ",".join(f'{{"id":{i},"text":"phrase numero {i} du recap"}}' for i in range(12))
    assert len(cut) > 200
    with pytest.raises(TruncatedResponseError):
        response_data(_response(cut), BeatBatch)


@pytest.mark.parametrize("junk", ["not json", "garbage", "desole, je ne peux pas"])
def test_short_junk_is_not_mistaken_for_a_truncation(junk: str) -> None:
    with pytest.raises(InvalidResponseError) as excinfo:
        response_data(_response(junk), BeatBatch)
    assert not isinstance(excinfo.value, TruncatedResponseError)


def test_truncation_is_retried_only_once() -> None:
    """Reessayer une troncature la reproduit : on ne brule pas 4 unites de quota pour rien."""
    cut = '{"beats":[' + ",".join(f'{{"panel_ids":[{i}],"summary":"beat {i} du chapitre"}}' for i in range(12))

    analyzer = GeminiAnalyzer(client=FakeClient(lambda contents, call_no, config: make_response(cut)), max_retries=3)
    with pytest.raises(TruncatedResponseError):
        analyzer.extract_beats(panels(2))
    assert analyzer.n_calls == TRUNCATION_RETRIES + 1 == 2


# --- Reparation JSON ------------------------------------------------------------------
def test_repair_never_touches_the_inside_of_strings() -> None:
    """Une narration peut contenir deux-points, guillemets echappes et nombres negatifs."""
    payload = json.dumps({"text": 'Hugh sneers, "Listen: the gate opens at dawn." Then, quiet: gone. -1, -2.'})
    assert _repair_json(payload) == payload
    assert _extract_json(payload)["text"] == json.loads(payload)["text"]


@pytest.mark.parametrize("valid", ['{"a":[1,-1,2],"b":3}', '{"scores":[10,-5,7]}', '{"a":[1,-1],"b":"x"}'])
def test_repair_leaves_a_valid_array_of_negatives_alone(valid: str) -> None:
    """Le garde-fou du retrait de valeur parasite : dans un tableau, ,-1, est legitime."""
    assert _repair_json(valid) == valid


@pytest.mark.parametrize(
    ("broken", "expected"),
    [
        ('{"paragraphs":[{plan_id: 0,"text":"x: y"}]}', {"paragraphs": [{"plan_id": 0, "text": "x: y"}]}),
        ('{"a":[1,2,],}', {"a": [1, 2]}),
        ('{"w":1,-1,"text":"a, -2, b"}', {"w": 1, "text": "a, -2, b"}),
    ],
)
def test_broken_json_is_repaired_locally(broken: str, expected: dict) -> None:
    assert _extract_json(broken) == expected


def test_repaired_json_costs_no_second_call() -> None:
    """Tout l'interet de la reparation : ne pas payer une unite de quota pour une virgule."""
    payload = '{"beats":[{panel_ids: [0, 1], "summary": "Le heros avance.", "characters": [], "dialogue": [],},]}'
    analyzer = GeminiAnalyzer(client=FakeClient(lambda contents, call_no, config: make_response(payload)))
    beats = analyzer.extract_beats(panels(2))
    assert analyzer.n_calls == 1
    assert [b.panel_ids for b in beats] == [[0, 1]]


def test_a_runaway_integer_does_not_kill_the_chapter() -> None:
    """Vu en production : une reponse contenant un entier de 5 234 chiffres.

    Python refuse de convertir au-dela de 4 300 chiffres et ``json.loads`` leve alors un
    ``ValueError`` ordinaire, pas un ``JSONDecodeError``. Cette seule difference a fait
    echouer un chapitre entier - apres avoir deja depense une quarantaine d'appels.
    """
    huge = "9" * 5234
    parsed = _extract_json('{"beats":[{"panel_ids":[0,' + huge + '],"summary":"x"}]}')
    assert parsed["beats"][0]["panel_ids"] == [0, RUNAWAY_INT]
    # Le reste du document est intact : rien d'autre n'a ete touche.
    assert parsed["beats"][0]["summary"] == "x"


def test_ordinary_integers_are_untouched() -> None:
    parsed = _extract_json('{"a":0,"b":-7,"c":183,"d":9007199254740991,"e":1.5}')
    assert parsed == {"a": 0, "b": -7, "c": 183, "d": 9007199254740991, "e": 1.5}


def test_a_runaway_integer_costs_no_second_call() -> None:
    """Meme logique que la reparation locale : un nombre aberrant ne doit pas rebruler une
    unite de quota. Le numero absurde est ensuite ecarte comme un numero invente."""
    huge = "8" * 5000
    payload = '{"beats":[{"panel_ids":[0,1,' + huge + '],"summary":"Le heros avance.","characters":[],"dialogue":[]}]}'
    analyzer = GeminiAnalyzer(client=FakeClient(lambda contents, call_no, config: make_response(payload)))
    beats = analyzer.extract_beats(panels(2))
    assert analyzer.n_calls == 1
    assert [b.panel_ids for b in beats] == [[0, 1]]        # le -1 est tombe a la validation


# --- Tenir en une seule requete ---------------------------------------------------------
def test_a_light_chapter_keeps_the_best_quality() -> None:
    analyzer = GeminiAnalyzer(client=FakeClient())
    assert analyzer.fit_for_single_call(panels(4), limit=10**9) is True
    assert (analyzer.max_image_width, analyzer.jpeg_quality) == INLINE_PAYLOAD_LADDER[0]


def test_the_ladder_stops_at_the_first_setting_that_fits() -> None:
    """Comprimer un peu plus vaut mieux que basculer en deux etapes : la, le redacteur ne
    voit plus aucune image et ne fait que reformuler des resumes deja aplatis."""
    analyzer = GeminiAnalyzer(client=FakeClient())
    analyzer.max_image_width, analyzer.jpeg_quality = INLINE_PAYLOAD_LADDER[0]
    au_premier = analyzer.payload_bytes(panels(6))

    assert analyzer.fit_for_single_call(panels(6), limit=au_premier - 1) is True
    assert (analyzer.max_image_width, analyzer.jpeg_quality) != INLINE_PAYLOAD_LADDER[0]
    # Le palier retenu tient vraiment sous le plafond : c'est le seul contrat qui compte.
    assert analyzer.payload_bytes(panels(6)) <= au_premier - 1


def test_a_chapter_that_never_fits_falls_back_to_two_steps(caplog) -> None:
    import logging

    analyzer = GeminiAnalyzer(client=FakeClient())
    with caplog.at_level(logging.WARNING, logger="src.modules.analyzer"):
        assert analyzer.fit_for_single_call(panels(4), limit=1) is False
    assert "deux etapes" in caplog.text
    assert analyzer.fit_for_single_call([], limit=1) is True     # rien a envoyer, rien a decider


def test_the_quality_setting_reaches_the_images_actually_sent() -> None:
    """Mesurer un poids qui ne serait pas celui des images envoyees rendrait tout le
    mecanisme faux : la mesure et l'envoi doivent suivre le meme reglage."""
    soigne = GeminiAnalyzer(client=FakeClient(), jpeg_quality=92)
    brut = GeminiAnalyzer(client=FakeClient(), jpeg_quality=40)
    assert soigne.payload_bytes(panels(4)) > brut.payload_bytes(panels(4))

    def taille(analyzer):
        parts = analyzer.build_single_call_contents(panels(4))
        return sum(len(p.inline_data.data) for p in parts if p.inline_data is not None)

    assert taille(soigne) > taille(brut)
    assert taille(brut) == brut.payload_bytes(panels(4))


def test_jpeg_quality_is_validated() -> None:
    for mauvais in (0, 96, -1):
        with pytest.raises(ValueError, match="jpeg_quality"):
            GeminiAnalyzer(client=FakeClient(), jpeg_quality=mauvais)


# --- Numerotation des cases -----------------------------------------------------------
def _paragraph(ids: list[int]) -> RecapParagraph:
    return RecapParagraph(text="Le heros avance dans la nuit.", emotion="calm", key_panel_ids=ids, action_heavy_ids=[])


def test_a_whole_wrong_numbering_is_rejected() -> None:
    """Cas observe : des numeros de case confondus avec des numeros de beat."""
    wrong = [_paragraph([100, 101]), _paragraph([102, 103]), _paragraph([104, 105])]
    with pytest.raises(InvalidResponseError, match="Numerotation"):
        normalize_recap(wrong, panels(8))


def test_a_single_invented_id_is_still_only_dropped() -> None:
    """La tolerance utile reste : une coquille isolee ne doit pas couter un appel."""
    kept = normalize_recap([_paragraph([0, 999]), _paragraph([4, 999])], panels(8))
    assert [ids for _, ids, _ in kept] == [[0], [4]]


def test_keyframe_group_rejects_a_fully_wrong_numbering() -> None:
    heights = {i: 100 for i in range(8)}
    choices = [KeyframeChoice(paragraph_index=i, key_panel_ids=[500 + i], action_heavy_ids=[]) for i in range(4)]
    group = [(i, "texte", [i]) for i in range(4)]
    with pytest.raises(InvalidResponseError, match="Numerotation"):
        normalize_keyframes(choices, group, heights, set())


def test_keyframe_group_still_tolerates_an_empty_choice() -> None:
    """Un groupe sans aucune case cle est un repli legitime de l'etape 2, pas une erreur."""
    heights = {i: 100 for i in range(8)}
    choices = [KeyframeChoice(paragraph_index=0, key_panel_ids=[], action_heavy_ids=[])]
    result = normalize_keyframes(choices, [(0, "texte", [0, 1])], heights, set())
    assert result[0]


# --- Cascade sur un plafond refuse ----------------------------------------------------
def _client_error(code: int, message: str) -> errors.ClientError:
    return errors.ClientError(code, {"error": {"code": code, "message": message}})


def test_a_refused_output_ceiling_moves_to_the_next_model() -> None:
    """Sinon ce 400 serait fatal et ferait perdre le chapitre, alors que le modele suivant l'accepte."""
    exc = _client_error(400, "Invalid value at 'generation_config.max_output_tokens' (32768)")
    assert gm.is_output_limit_error(exc)
    assert gm.is_model_unavailable_error(exc)
    assert not gm.is_invalid_key_error(exc)


def test_an_ordinary_400_stays_fatal() -> None:
    other = _client_error(400, "Request contains an invalid argument")
    assert not gm.is_output_limit_error(other)
    assert not gm.is_model_unavailable_error(other)
