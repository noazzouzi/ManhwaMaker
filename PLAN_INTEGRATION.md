# Plan d'intégration

Base : 257 tests collectés (pas 249), 20 fichiers, un seul conftest. Toutes les lignes citées viennent du repérage ; les ancres clés ont été relues dans le code (timeline_builder.py:667, format_factory.py:83/105, slicer.py:73/75/79, analyzer.py:116-143, 801-814, 878-887, 1387-1399).

---

## 1. Ordre d'exécution

1. **Couverture des cases (LONG)** — active `expand_panel_ids` en format long. 5 lignes, zéro appel Gemini, +86 % de cases à l'écran mesuré sur 8 chapitres. Effort S. **En tête** : c'est le seul changement visible à l'écran le soir même, et il ne dépend de rien.
2. **Outillage de test immédiat** — extraire `tests/gemini_fakes.py`, créer `tests/test_main.py`. Effort S. Ne dépend d'aucune décision ; `src/main.py` (468 lignes, 0 test) est traversé par tout le reste.
3. **Fiabilité Gemini** — `max_output_tokens`, troncature nommée, réparation JSON sensible aux chaînes, rejet des ids hors plage. Effort M. **Avant le prompt** : le master prompt allonge la sortie (plan + characters), c'est exactement le régime qui tronque en silence aujourd'hui.
4. **Découpe (slicer)** — supprime le plafond de 3 morceaux, vise la hauteur du cadre. Effort M. **Avant la ré-analyse** : la re-découpe renumérote les cases, donc tout `scenes.json` existant devient faux.
5. **Prompt maître** — un seul gabarit, réponse `{plan, paragraphs, characters}`, plus aucun pilotage de longueur. Effort L. Dépend de la fiabilité (3) et doit tourner sur le nouveau découpage (4).
6. **Mémoire de série (fiche personnages)** — `characters.json` par chapitre, fusion à la lecture, barrière d'ordre du batch. Effort M. Dépend de (5) : sans lui `analysis.characters` reste vide. Exception : `CharacterCard` + `src/modules/series_memory.py` ne dépendent de rien et peuvent être écrits dès l'étape 2.
7. **Reste des tests** — écrits au fil des zones, pas avant : ils figent des interfaces qui ne sont pas encore décidées.

**Arbitrages tranchés** (les repérages se contredisent) :
- *Découpe d'abord ou couverture d'abord ?* → **Couverture d'abord.** Le drapeau ne dépend pas des numéros de case ; seul le comptage 560 → 1042 sera à rejouer après la re-découpe. La découpe reste avant le prompt, qui lui travaille sur les numéros.
- *Le gain de la couverture est-il réel ?* → **Oui.** La zone « tests » conclut au gain nul, mais sur la fixture synthétique de 4 cases avec une scène de 5,3 s. La zone « couverture » mesure 8 chapitres réels, scènes de 17 à 29 s : `int(durée // 2,5)` autorise 7 à 11 cases. La mesure sur chapitres réels l'emporte.
- *Où vit le rejet des ids hors plage ?* → **Dans `normalize_recap`, avec seuil sur les ids DISTINCTS** (zone fiabilité), pas « tout id > dernier index » (zone tests). Le second casse `tests/test_analyzer.py:959` et fait payer un appel Gemini pour un seul id inventé.
- *`parse_recap` renvoie-t-il l'objet ou la liste ?* → **L'objet** (`RecapDraft`/`ScriptDraft`), sinon `plan` et `characters` n'atteignent jamais `write_recap`. `tests/test_analyzer.py:296` est à réécrire.

---

## 2. Le détail, zone par zone

### 2.1 Couverture des cases — activer `expand_panel_ids` en LONG

| fichier | ligne | ce qu'il y a | ce qu'on met | risque |
|---|---|---|---|---|
| src/models/format_profile.py | 143 (ajout après), doc 129-137 | `PacingRules` : min_clip_s, max_clip_s, max_subshots_per_panel, min_shot_s. L'intention « élargir » est portée en douce par max_clip_s | `expand_to_unused_panels: bool = False` + entrée dans le bloc Attributes | Aucun. Défaut **False** : un `PacingRules()` construit à la main ne change pas de comportement |
| src/modules/format_factory.py | 83 | `PacingRules(min_clip_s=2.5, max_clip_s=None, max_subshots_per_panel=1)` | ajouter `expand_to_unused_panels=True` | C'est LE changement de comportement. Le commentaire l.76 (« strictement inchangé ») devient faux |
| src/modules/format_factory.py | 105-107 | `PacingRules(min_clip_s=None, max_clip_s=SHORT_MAX_CLIP_S, max_subshots_per_panel=6, min_shot_s=0.45)` | ajouter `expand_to_unused_panels=True` | **Oubli = régression muette du SHORT** (22 plans sur 3 cases clés) et aucun test ne l'attrape |
| src/modules/timeline_builder.py | 667 | `if profile.pacing.max_clip_s is not None and montable:` | `if profile.pacing.expand_to_unused_panels and montable:` | Aucun appel Gemini, aucune image chargée : la fonction ne manipule que des entiers |
| src/modules/timeline_builder.py | 671-674 | log « Format court : %d cases disponibles… » | `"Format %s : %d cases montees au lieu de %d"`, avec `profile.name` | Nul. Aucun test ne lit ce message |
| src/modules/timeline_builder.py | 653-654, 160-166, 8-10 | docstring : « en mode long c'est voulu » | réécrire : l'élargissement est la règle des deux formats | Nul (documentation), mais la phrase renversée doit partir |
| src/modules/pacing.py | 189-193 | `make_pacing` lit `max_clip_s` | **ne rien toucher** | Y toucher ferait basculer le LONG sur `ShortPacing` : plans < 1 s, plancher 2,5 s perdu, sous-titres incohérents |

**Tests à ajouter**
- `test_both_profiles_expand_to_unused_panels` (LONG et SHORT à True) — le garde-fou contre l'oubli de la l.105.
- `test_long_expands_panels_without_inheriting_short_pacing` (`max_clip_s is None`, drapeau True, `isinstance(..., LongPacing)`).
- `test_long_timeline_shows_the_panels_left_out_by_the_analysis` (scène 0 clés [0,1] → clips [0,1,2] avec un audio allongé).
- `test_widening_never_puts_a_panel_below_min_clip_s`.
- `test_expansion_can_be_switched_off_by_profile` (`pacing={"expand_to_unused_panels": False}`).
- `test_long_widening_never_shows_a_filler_scene_panel` (case 2 de la scène filler ne doit pas apparaître).
- `test_long_widening_never_crops_never_upscales_and_caps_the_zoom`.

**Tests qui vont casser** (2, confirmés en simulant le changement)
- `tests/test_timeline.py:74` l.78 : `[0,1,3]` devient `[0,2,3]`. Cause : scène 0 → [0,1,2] puis `limit_panels_for_duration` (5,3 s // 2,5 = 2) garde 0 (1000 px) et 2 (400 px). Les durées l.82-84 tombent ensuite. Remède : allonger l'audio de la scène 0 ou poser `expand_to_unused_panels=False` sur ce test.
- `tests/test_timeline.py:104` l.116 : `[0,1]` devient `[0,2]`. Même cause, même remède.
- `tests/test_timeline.py:386` est à revérifier (3 clips, mais durées changées).
- Survivent : `test_format.py:229`, `test_timeline.py:244`, `test_capcut.py:79`, `test_preview.py:78`.

---

### 2.2 Fiabilité Gemini

| fichier | ligne | ce qu'il y a | ce qu'on met | risque |
|---|---|---|---|---|
| src/modules/analyzer.py | 124-128 | bloc de constantes s'arrêtant à `DEFAULT_TIMEOUT_MS`. Zéro occurrence de `max_output_tokens` dans src/ | `DEFAULT_MAX_OUTPUT_TOKENS: int = 32_768` (plus gros appel unique observé : 3485 sortie + 8123 réflexion = 11 608) et `TRUNCATION_RETRIES: int = 1` | Aucun tant qu'inerte |
| src/modules/analyzer.py | 1387-1399 | `config_for` ne passe jamais `max_output_tokens` ; les 4 configs (1401, 1405, 1412, 1421) en héritent | `self.max_output_tokens` au constructeur (signature 1337, affectation 1363) + passé dans le `GenerateContentConfig` | Un modèle de la cascade plafonnant plus bas répond 400. `is_invalid_key_error` (gemini_manager.py:121-128) ne reconnaît pas ce 400 → `GeminiManagerError` fatal (l.472) = chapitre perdu. Prévoir le classement en erreur transitoire |
| src/modules/thumbnail/analyzer.py | 176-183 | 2ᵉ construction de config, sans plafond | ajouter `max_output_tokens` | Aucun test ne lit cette config. Ne pas toucher image_backends.py:148 |
| src/modules/analyzer.py | 230-239 | 3 exceptions, rien ne distingue tronqué de mal formé | `class TruncatedResponseError(InvalidResponseError)` après 234, + `__all__` | Aucun **si** c'est bien une sous-classe : `_is_retryable` (1259-1261) et les tests 288-290/306 continuent de passer |
| src/modules/analyzer.py | 881-887 | `finish_reason` n'est lu **que** si le texte est vide ; une réponse tronquée avec du texte ressort en « JSON invalide » | sortir `finish = _finish_reason(response)` avant le `if not text` ; si `MAX_TOKENS`, lever `TruncatedResponseError` | `tests/test_analyzer.py:324-325` attend `match="MAX_TOKENS"` sur une réponse vide : **ne pas retirer** la chaîne `MAX_TOKENS` du message de la branche texte vide |
| src/modules/analyzer.py | 801-814 | `_extract_json` : fence, `json.loads`, repli `find('{')`/`rfind('}')`, sinon `InvalidResponseError`. Aucune réparation | (1) `_json_spans(text)` qui suit l'état « dans une chaîne » + échappements, et n'applique les corrections (clés nues, virgules traînantes, valeur parasite) **que hors chaîne** ; (2) détection structurelle de troncature : contient `{`/`[`, > 200 caractères, dernier caractère non blanc ni `}` ni `]` → `TruncatedResponseError` | Les deux bornes sont indispensables : sans elles `'not json'` et `'garbage'` (tests 288-290, 554-582) basculent en troncature et le décompte d'appels des tests change |
| src/modules/analyzer.py | 1541-1570 | `_generate_via_manager` réessaie `max_retries + 1` fois sur `InvalidResponseError`, contenus identiques → jusqu'à 4 unités de quota | compteur `truncations` ; au-delà de `TRUNCATION_RETRIES`, on relance l'exception | Le chemin client injecté (1506-1539) a sa **propre** boucle : y poser le même compteur, sinon tests et production divergent |
| src/modules/analyzer.py | 702-715 | ids inconnus retirés un à un (warning l.706) ; paragraphe vidé recasé sur la plus grande case libre (709-715) → 125 ids pour 38 beats produit un recap plausible, jamais rejeté | avant la boucle : `hors_plage` = ids distincts absents de `heights`. Lever `InvalidResponseError` si (a) `len(hors_plage) > max(2, 0.2 * len(panels))` ou (b) plus de la moitié des paragraphes n'ont aucun id valide. Message nommant le plus grand id valide et le plus grand reçu | `tests/test_analyzer.py:959-968` (8 cases, ids [0,999] et [4,999]) : **1 seul id distinct hors plage**, chaque paragraphe garde un id valide → les deux conditions sont fausses, test inchangé. Un seuil « > 50 % des ids » le casserait |
| src/modules/analyzer.py | 1184-1191 | `normalize_keyframes` : même tolérance silencieuse | condition (a) seulement | **Ne pas** appliquer (b) : le repli « aucune case clé » (1194-1196) est légitime et fréquent en étape 2 |

**Tests à ajouter** — `test_config_for_sets_the_output_ceiling` (les 4 configs) ; `test_truncated_response_is_named_as_such` (MAX_TOKENS **et** STOP) ; `test_truncation_is_retried_only_once` (exactement 2 appels avec `max_retries=3`) ; `test_json_repair_never_touches_the_inside_of_strings` (narration identique octet pour octet, charge utile contenant `Hugh sneers, "Listen: the gate opens at dawn."`) ; `test_invalid_json_is_repaired_without_a_second_call` (1 seul appel) ; `test_out_of_range_ids_are_rejected_then_retried` ; `test_a_single_invented_id_is_still_only_dropped` ; `test_keyframe_group_rejects_a_fully_wrong_numbering`.

**Tests qui vont casser** — aucun. Trois restent verts mais fragiles aux seuils, à relire à chaque ajustement : `test_analyzer.py:554-582` (décomptes d'appels exacts), `:959-968` (un id hors plage), `:447-467` (repli keyframes).

---

### 2.3 Découpe — supprimer le plafond de morceaux

| fichier | ligne | ce qu'il y a | ce qu'on met | risque |
|---|---|---|---|---|
| src/modules/slicer.py | 70-79 | `SPLIT_MAX_HEIGHT=1200` (73), `SPLIT_MAX_PIECES=3` (75), `SPLIT_SEARCH_RATIO=0.15` (79) — aucune relation avec le cadre | supprimer les trois. Ajouter `DEFAULT_FRAME_HEIGHT=1080`, `SPLIT_MAX_PIECE_RATIO=1.6`, `SPLIT_BOUNDARY_BONUS=0.20`, `SPLIT_GRID_PX=24`, `SPLIT_CHANGE_THRESHOLD=24`, `SPLIT_BORDER_COVER=0.50` | Seuls slicer.py, `tests/test_slicer.py:280-282` et la doc citent les anciens noms. Aucun module de src/ ne les importe |
| src/modules/slicer.py | 107-150 | `compute_row_variance` est le seul signal : on ne sait pas voir une bordure collée bord à bord | `compute_row_change(strip, threshold=…)` : fraction des colonnes dont l'écart vertical dépasse le seuil. **Impératif** : calcul par blocs `_VARIANCE_BLOCK_ROWS` avec recouvrement d'UNE ligne (comme 146-149) — sur ep1 la bande fait 219 443 × 900, un `np.diff` non bloqué réclame 395 Mo | Le gris vient de `cv2.COLOR_RGB2GRAY` (image_utils.py:140), le bac à sable utilisait PIL : ±1 niveau pour un seuil de 24. Les coutures JPEG produisent aussi un pic pleine largeur |
| src/modules/slicer.py | 213-260 | `split_tall_range` : `n = min(3, ceil(h/1200))`, coupes égales déplacées vers la variance minimale à ±15 % (no-op mesuré : 0/60 segments) | `find_borders(...)` (candidats à `row_change >= 0.50`, forces 0,5/0,75/1,0, groupage 20 px, fusion 25 px) + `split_segment_to_frame(...)` (programmation dynamique maximisant `somme(coverage) - lam*n + bonus*force`, hauteurs bornées à [min_piece, 1,6×frame_height], repli division égale) | DP en O(|P| × 1,6·H/24) : 0,9 s pour 102 segments contre quelques µs. Les seuils 0,50/0,60/0,70, le bonus 0,20, la grille 24 px sont calés sur **un seul chapitre** |
| src/modules/slicer.py | 240-241 | `if max_height <= 0 or max_pieces < 2 or height <= max_height…` : seuil explicite à 1200 | plus de seuil : sous `frame_height` la couverture est linéaire, couper coûte `lam` et ne rapporte rien. Bascule effective vers 1 660 px | **Seul endroit moins découpant qu'avant** : un segment de 1 400 px reste entier à 0,77×. La couverture moyenne gagne, la couverture cumulée perd. Jugé par la mesure, pas à l'œil |
| src/modules/slicer.py | 297-309 | `split_max_height` (307), `split_max_pieces` (308) | `frame_height: int = DEFAULT_FRAME_HEIGHT`, `frame_width: int = 1920`. `frame_height=0` = ne jamais sous-découper | Rupture d'API : 10 tests passent `split_max_height=0` → `TypeError`. Mécanique, à faire d'un bloc |
| src/modules/slicer.py | 350-353 | `ValueError` sur `split_max_height < 0` et `split_max_pieces < 1` | mêmes messages sur `frame_height < 0` et `frame_width < 1` | `tests/test_slicer.py:435-436` doit passer `frame_height=-1` / `frame_width=0`, sinon `pytest.raises(ValueError)` reçoit un `TypeError` |
| src/modules/slicer.py | 366 | `row_variance = compute_row_variance(rgb)` | ajouter `row_change`, **seulement si `frame_height > 0`** | Une passe complète de plus sur 219 k lignes. Le court-circuit protège `test_render_debug_overlay_keeps_min_width_for_full_chapters` (800 × 70 000) |
| src/modules/slicer.py | 378-416 | boucle unique : padding, filtre, `split_tall_range`, création des Panel | deux temps : (1) segments paddés retenus (padding 383-384 et filtre 386-392 inchangés) ; (2) itération de Dinkelbach sur **tous** les segments du chapitre (8 tours max, lam initial 0,40, arrêt à 1e-5) ; (3) création des Panel, logique 397-416 identique | Vraie restructuration. Le lambda rend le découpage d'un segment dépendant du reste du chapitre : les tests sur bandes synthétiques deviennent sensibles au contenu global |
| src/modules/slicer.py | 394 | `pieces = split_tall_range(...)` | morceaux issus de la DP globale. Étiquettes `['top', 'middle'×(n-2), 'bottom']` : `Panel.part` (panel.py:53) accepte déjà `middle` répété | Plus de plafond à 3 : max observé 4-5. `analyzer.py:526-531` et `report.py:82-85` traitent `middle` sans le supposer unique |
| src/modules/slicer.py | 21-24 et 418 | docstring « 2 ou 3 blocs… ligne la plus calme », log idem | réécrire les deux + `__all__` (743-747) + PRD.md:98-99, 317 ; RESUME.md:377, 390, 888, 1232 | Aucun risque d'exécution. Laisser les .md produirait deux descriptions contradictoires de l'étape 2 — exactement ce qui a brouillé les prompts |
| src/pipeline.py | 256 | `panels = slice_panels(strip)` — aucun argument, le profil n'atteint jamais l'étape 2 | `profile = options.profile()` puis `frame_height=profile.framing.height if profile.framing.fit == "contain" else 0`, `frame_width=profile.framing.width` | **Point dur.** Test sur `fit`, pas sur le nom du profil. Le cache (l.250) ne teste que l'existence de panels.json : écrire un `slice_params.json` à côté et re-découper s'il diffère. **Ne rien ajouter dans panels.json** (`test_slicer.py:600` fige le jeu de clés) |
| src/modules/slicer.py | 59-60 | `GIANT_PANEL_HEIGHT = 1500` | ne rien changer, mais le savoir : 7 morceaux sur 187 restent au-dessus (contre 25 sur 183) | Borner à 1500 supprimerait tout `scroll_vertical` (−0,3 % de couverture) : option **non retenue**, à décider après un premier rendu |

**Tests à ajouter** — `test_compute_row_change_detects_full_width_borders` ; `test_compute_row_change_block_boundary_is_seamless` ; `test_split_segment_to_frame_targets_frame_height` (pavage exact, morceaux dans [180, 1728]) ; `test_split_segment_to_frame_snaps_to_a_real_border` (coupe à < 25 px du saut) ; `test_split_segment_to_frame_keeps_short_segments_whole` ; `test_slice_panels_frame_height_zero_disables_splitting` (monkeypatch qui lève, pour verrouiller le court-circuit) ; `test_slice_panels_labels_more_than_three_pieces` ; `test_slice_panels_every_piece_respects_min_panel_height` ; `tests/test_pipeline_slicing.py::test_stage_scrape_slice_passes_long_frame_height` et `::test_stage_scrape_slice_keeps_short_profile_unsliced`.

**Tests qui vont casser** — `test_slicer.py:279` (ImportError dès la l.280, à réécrire intégralement) ; `:304` (l.322 TypeError, géométrie figée l.311/320/325-330) ; `:261` (TypeError l.265/270 ; sinon 1830 px donne toujours 2 morceaux, bon canari) ; `:426` (l.435-436 `ValueError` → `TypeError`) ; `:341`, `:377`, `:480`, `:584`, `:677`, `:690` (tous `split_max_height=0` → TypeError, rupture purement mécanique). Survit : `test_analyzer.py:713` (aller-retour disque uniquement).

---

### 2.4 Prompt maître

| fichier | ligne | ce qu'il y a | ce qu'on met | risque |
|---|---|---|---|---|
| src/modules/analyzer.py | 365-433 | deux gabarits divergents : `SCRIPT_SYSTEM_INSTRUCTION_TEMPLATE` (365-395) et `SINGLE_CALL_SYSTEM_INSTRUCTION_TEMPLATE` (397-433) | `MASTER_PROMPT_TEMPLATE` + `INPUT_BLOCK_IMAGES` + `INPUT_BLOCK_BEATS`, et `master_instruction(*, mode, language, meta, n_ids, first_id, last_id, max_key, memory)` qui formate **en deux passes** : d'abord le bloc, puis le corps. Texte ASCII pur (0 caractère > 127 sur 11 301) | Une passe unique laisse `{n_ids}/{first_id}/{last_id}` littéraux — c'est arrivé en bac à sable (prompt_envoye.txt, l.11-12, parti tel quel chez Gemini). Étendre le test ASCII `:227` au nouveau gabarit |
| src/modules/analyzer.py | 116-123, 143 | `SCRIPT_WORDS_PER_PANEL=11`, `MIN_WORDS=250`, `MAX_WORDS=1500`, `MIN_PARAGRAPHS=4`, `MAX_PARAGRAPHS=40`, `MIN_LENGTH_RATIO=0.65` | supprimer les six. Une seule nouvelle : `PEAK_MAX_SPAN: int = 4` (plafond d'étendue, pas un budget de mots) | `SCRIPT_MIN_LENGTH_RATIO` est dans `__all__` (1955) et importé par `test_analyzer.py:921`. Aucun module de src/ ne les importe |
| src/modules/analyzer.py | 573-581, 731-740 | `single_call_targets`, `target_script_words`, `target_paragraphs` | supprimer les trois + leurs entrées `__all__` (1957, 1989, 1990). Le nombre de paragraphes dérive des UNITS | Les garder « au cas où » donnerait l'illusion que la longueur est pilotée |
| src/modules/analyzer.py | 584-596 | header : ligne chapitre + « Script language… Target length… » (591-594) + « All N panels… » | header réduit à `All {n} panels of the chapter follow, in reading order:`. La ligne chapitre alimente `{chapter_line}`. Signature `build_single_call_header(panels)` | `test_analyzer.py:832` casse (attendu). Vérifier que **les deux** chemins passent bien `meta`, sinon le titre de série disparaît |
| src/modules/analyzer.py | 599-609 | footer répétant la cible (603-608) | footer sans cible, mais **conserver mot pour mot** le fragment `numbered {first} to {last}` | Le footer est le dernier rappel avant génération ; y retirer toute contrainte rouvre le défaut des 125 ids |
| src/modules/analyzer.py | 743-765 | `build_script_prompt` : cible 749-752, lignes de beats 754-763, consigne finale 764 | supprimer 749-752 et la ligne chapitre. **Conserver à l'identique** le format des lignes de beats. Remplacer 764 par le rappel de plage + format objet | Le bac à sable écrivait `[FILLER]`, le code écrit `[FILLER - do not narrate]` ; le regex des tests tolère les deux, mais **ne pas y toucher** : le prompt divergerait de ce qui a été mesuré |
| src/models/scene.py | 111-151 | `ParagraphDraft`, `ScriptDraft`, `RecapParagraph`, `RecapDraft` | ajouter `PlanAct`, `PlanUnit`, `ChapterPlan` (aucun champ avec défaut), `CharacterCard`. `plan_id: int = -1` et `weight: int = 2` sur les deux paragraphes ; `plan` et `characters` sur les deux drafts. **Garder les deux classes distinctes** | `test_analyzer.py:722` prend `next(iter($defs))` ; les `$defs` sont triés alphabétiquement, le premier devient `ChapterPlan` → tous ses champs doivent être requis. **Jamais** de `Field(min_length=…)`. Les défauts sur weight/plan_id sont indispensables pour relire un `analysis_checkpoint.json` ancien (299-304). Les faux clients aiguillent sur `is ScriptDraft` / `is RecapDraft` (151-157, 807) |
| src/models/scene.py | 174-199 | `ChapterAnalysis` : scenes (198), beats (199) | `characters: list[CharacterCard] = Field(default_factory=list)` et, optionnellement, `plan`. **Ne pas** ajouter `weight` à `Scene` | `test_analyzer.py:705` fige les clés exactes de `scenes[0]` : `weight` sur `Scene` le casse, `characters` sur `ChapterAnalysis` non |
| src/modules/analyzer.py | 890-903 | `_validate_list(data, model_cls, list_key, fix_item)` | ajouter `fix_root: Callable[[dict], None] | None = None`, appelé sur le dict complet : injecte un plan vide et une liste characters vide plutôt que de lever | Sans lui, une réponse sans `plan` déclenche un réessai = un appel Gemini de plus. Un plan vide accepté doit être journalisé en WARNING |
| src/modules/analyzer.py | 919-929, 952-963 | `_fix_paragraph`, `_fix_choice`, `_fix_recap` ; `parse_script`/`parse_recap` renvoient la liste | normaliser `weight` (borné [1,3], défaut 2) et `plan_id` (défaut -1) ; `_fix_master` pour plan et characters ; `parse_*` renvoient l'**objet** validé, appelants 1603 et 1758 ajustés | `parse_script`/`parse_recap` sont dans `__all__` (1999, 1960) et importés par `test_analyzer.py:296` : le test l.295 casse. Repli possible : ajouter `parse_master_*` à côté |
| src/modules/analyzer.py | 677-728 | `normalize_recap` : rejet des ids hors plage (701-706), repli, reconstruction d'un `ParagraphDraft` neuf l.718 | logique conservée. Seule la l.718 change : recopier `weight` et `plan_id`. Le contrôle global des ids hors plage vient de la zone fiabilité | `_enforce_reading_order` (637-674) reste indispensable : le master prompt ne garantit pas l'ordre croissant entre paragraphes |
| src/modules/analyzer.py | 1749-1818 | `write_recap` : `too_short` (1762-1766), rappel de longueur (1776-1779, 1785-1786), comparaison de longueurs (1792-1798) | supprimer tout le volet longueur. Régénération conservée, déclenchée seulement par `violations` ou `intro`. Récupérer `characters` et `plan` | Seul mécanisme qui empêchait 478 mots pour un chapitre entier. Garder le `logger.info` 1814-1817 comme trace. Une reprise plus pauvre remplacera désormais une bonne première réponse — acceptable, la reprise n'étant plus déclenchée que par une faute de style |
| src/modules/analyzer.py | 1599-1632 | `write_script` : `script_config(beats)` (1602), `normalize_script(parse_script(r), beats)` (1603), régénération, conventions (1629) | même structure, config sans cible, parse objet, weight/plan_id conservés, retour élargi. **Ignorer** les `key_panel_ids` du mode BEATS : l'étape 2 (1648) voit les images et reste l'autorité | Supprimer l'étape 2 gagnerait des appels mais choisirait des cases jamais vues : régression directe du montage |
| src/modules/analyzer.py | 1101-1147, 202-217 | CTA du modèle toujours retiré (1135-1144) puis phrase canonique insérée (1146-1152) | helpers tous conservés. Un seul changement : si le modèle a écrit **exactement un** CTA bien placé (indice ≥ 2, pas le dernier), on le garde. `--cta` / `--no-cta` priment toujours | Le comptage doit porter sur **tout** le script, pas paragraphe par paragraphe. Vérifié : `test_analyzer.py:360` reste vert (son CTA est au paragraphe 1) |
| src/modules/analyzer.py | 173-186, 190-199 | 13 motifs visuels, 8 motifs d'intro | élargir `FORBIDDEN_PATTERNS` : « the protagonist », « our hero », « the narrator », « artwork », « illustration », « <X> unfolds », « the screen », « the interface », « a notification », « a system message », « a display ». `GENERIC_INTRO_PATTERNS` inchangé | Chaque motif ajouté peut déclencher **un appel Gemini de plus**. Faux positifs : « the interface », « a display » dans une série où le Système est un objet du récit. Garder un seul réessai |
| src/modules/analyzer.py | 1320-1385, 1405-1419, 1875-1892 | `script_config` / `single_call_config` formatent les anciens gabarits ; `_chapter_analysis` sans personnages | appeler `master_instruction(mode=…)`, passer le schéma élargi à `config_for` (1387). `__init__` gagne `memory: ChapterMemory | None = None`. `_chapter_analysis` stocke `characters` | La sortie grossit → dépend du plafond `max_output_tokens` (zone fiabilité) |
| src/pipeline.py | 292-314 | `stage_analyze` construit l'analyzer (294-298) sans mémoire | charger le `scenes.json` de l'épisode précédent et passer `memory=` | Voir zone mémoire de série |
| src/modules/analyzer.py | 1934-2015 | `__all__` exporte les six symboles de longueur | les retirer ; ajouter les nouveaux | Python n'applique `__all__` qu'à `import *` : l'erreur ne se verrait qu'à la lecture |

**Tests à ajouter** — `test_master_prompt_fills_both_input_blocks` (dont `'{' not in rendered`, qui attrape la régression du bac à sable) ; `test_master_prompt_has_no_length_target` ; `test_master_response_carries_plan_weight_and_characters` ; `test_master_response_without_plan_or_characters_is_accepted` (1 seul appel) ; `test_weight_and_plan_id_are_coerced` ; `test_out_of_range_ids_are_rejected` ; `test_model_cta_is_kept_when_well_placed` ; `test_characters_round_trip_through_scenes_json` ; `test_memory_block_reuses_previous_characters` ; `test_checkpoint_from_the_old_paragraph_format_still_loads` ; `test_widened_forbidden_patterns` (dont le non-déclenchement sur « The System grants him a class. »).

**Tests qui vont casser** — `:227` (l.246 et l.262 : `master_prompt_v7.txt` ne contient pas « STRICTLY FORBIDDEN ») ; `:266` (à supprimer) ; `:808` (l.832 « Target length » et l.834 « HOOK », absent du v7) ; `:837` (première moitié à supprimer, `normalize_recap` l.845-865 à conserver ailleurs) ; `:919` (test entier supprimé) ; `:700` **seulement si** on ajoute `weight` à `Scene` — donc ne pas le faire. Hors pytest : `tests/test_analyzer_local.py:104` fabrique une réponse sans plan/characters/weight, à mettre à jour en même temps.

---

### 2.5 Mémoire de série (fiche personnages)

| fichier | ligne | ce qu'il y a | ce qu'on met | risque |
|---|---|---|---|---|
| src/models/scene.py | 172-173 | aucun modèle de personnage ; seul `BeatDraft.characters` (94), jamais persisté | `CharacterCard(name, also_called, who)` + `__all__` (233-249) | Ne pas le réutiliser tel quel comme response_schema Gemini : l'en-tête du module (9-13) interdit les défauts dans les schémas envoyés |
| src/models/scene.py | 199 | `ChapterAnalysis` sans personnages | `characters: list[CharacterCard] = Field(default_factory=list)` et `tail: str = ""` | Champs avec défaut : les scenes.json existants se relisent, `test_analyzer.py:700` reste vert |
| src/modules/series_memory.py | nouveau | n'existe pas ; la seule mémoire est `AnalysisCheckpoint` (242-340), interne à UN chapitre | `CHARACTERS_FILE`, `MAX_SHEET_CARDS=20`, `MAX_WHO_CHARS=160`, `series_key`, `save_chapter_sheet` (tmp + `os.replace`, motif de batch_processor.py:164-169), `load_series_context` (glob, tolérance OSError/ValueError comme 117-124, filtre `episode_no < N`, fusion par `casefold`), `format_sheet`, `format_tail` | Le glob parcourt tout `output/` : trivial aujourd'hui, à surveiller sur 200 épisodes. `MAX_SHEET_CARDS` est arbitraire. Zéro appel Gemini |
| src/pipeline.py | 152-153 | pas de réglage de mémoire | `series_memory: bool = True`, `series_root: str | None = None` ; `--no-series-memory` dans main.py | `run --out <ailleurs>` sort le chapitre du dossier partagé — d'où `series_root` |
| src/pipeline.py | 285-298 | branche de réutilisation (287) puis construction de l'analyzer (294-298), sans contexte | charger cartes + tail avant la branche, passer `character_sheet=` et `previous_tail=` | `meta.episode_no` est `int | None` (chapter.py:289) : si None, **désactiver** la mémoire plutôt que risquer un chapitre qui se lit lui-même |
| src/pipeline.py | 287-289, 318-319 | `save_analysis` n'écrit que scenes.json ; la branche de réutilisation n'écrit rien | `save_chapter_sheet` après 318 **et** dans la branche de réutilisation | Sans cela, un chapitre traité avant la fonctionnalité reste invisible et un `--redo tts` ne répare pas la série |
| src/modules/batch_processor.py | 85-94 | `BatchOptions` n'ordonne rien ; l.401 `asyncio.gather` lance tout | `series_order: bool = True`, `series_order_timeout_s: float = 240.0`, validation dans `__post_init__` ; `--no-series-order` | **Le seul vrai arbitrage** : l'analyse d'une série est sérialisée. Le temps de mur d'un lot de 5 passe d'une attente Gemini à cinq |
| src/modules/batch_processor.py | 337, 362, 386-396 | un seul `asyncio.Event` partagé (337) ; `sem_chapters` enchaîne sur l'analyse (362-365) ; le try de `one` n'a **aucun** `finally` | `analyzed: dict[str, Event]`, table `predecessors` (via `parse_ids_from_url`, déjà importé l.41), `await asyncio.wait_for(...)` **avant** 362 (hors sémaphore), `set()` juste après 365, `finally: set()` après 396 **et** `set()` sur la branche « déjà fait » (343-345) | **LE point d'interblocage.** Attendre en tenant `sem_chapters` gèlerait le lot. Sans `finally` ni `set()` sur la branche ignorée, un chapitre en échec bloque tout jusqu'au timeout |

**Tests à ajouter** — `tests/test_series_memory.py` : `test_sheet_is_written_next_to_scenes_json` ; `test_merge_keeps_the_first_spelling_and_unions_the_aliases` ; `test_current_and_later_episodes_are_excluded` (couvre `--redo`) ; `test_tail_comes_from_the_highest_earlier_episode` (fiches écrites dans le désordre) ; `test_foreign_series_and_corrupt_files_are_ignored` ; `test_sheet_is_capped_and_rendered_for_the_prompt` ; `test_concurrent_chapters_never_touch_the_same_file`. `tests/test_pipeline_series.py` : `test_stage_analyze_writes_the_sheet_even_when_the_analysis_is_reused` (premier test direct de pipeline.py) ; `test_stage_analyze_injects_the_previous_sheet_into_the_analyzer`. `tests/test_batch.py` : `test_series_order_analyses_episodes_in_ascending_order` ; `test_series_order_never_deadlocks_on_failed_skipped_or_quota_chapters` ; `test_series_order_off_keeps_the_historical_parallelism` ; `test_the_canonical_name_comes_from_the_lowest_episode_whatever_the_finish_order`.

**Tests qui vont casser** — aucun, vérifié fichier par fichier. Les quatre candidats relus : `test_batch.py:124` (signature des Stages inchangée, car `stage_analyze` charge la fiche lui-même), `:168` et `:173` (la barrière ne fait que baisser le pic), `:239` (la barrière précède le test de quota existant 363-364), `test_analyzer.py:700-707` (nouveaux champs avec défaut).

---

### 2.6 Outillage de test (transversal)

| fichier | ligne | ce qu'il y a | ce qu'on met | risque |
|---|---|---|---|---|
| tests/gemini_fakes.py | nouveau (extrait de test_analyzer.py:95-182 et 76-86) | les faux Gemini ne sont importables qu'en exécutant les 1072 lignes de test_analyzer.py | déplacer tel quel sur le modèle de `tests/synthetic_strip.py` ; `tests/__init__.py` existe déjà | Ne rien renommer ; vérifier le même nombre de tests passés avant/après |
| tests/test_main.py | nouveau — src/main.py fait 468 lignes, 0 test | rien n'importe `src.main` ; `_pipeline_options` (50-90), codes de sortie 1 (147/151) et 2 (268) non couverts | `typer.testing.CliRunner` (typer 0.27.2, click 8.5.0, vérifié). Monkeypatch de `src.main.run_pipeline` (importé l.25) et de `src.utils.gemini_manager.GeminiManager` (importé **dans** la fonction l.129 → patcher le module source) | Le conftest autouse (11-19) supprime `GEMINI_API_KEYS` : sans monkeypatch, tout `run` sort en code 1. `_setup(verbose)` (31-43) reconfigure le logger racine pour la session |

**Tests à ajouter** — `test_cli_help_lists_every_command` (zéro monkeypatch, le moins cher, attrape une signature `Annotated` cassée) ; `test_run_maps_the_flags_onto_pipeline_options` ; `test_run_prints_failed_and_exits_1` ; `test_merge_exits_2_when_no_chapter_matches` ; `test_preview_and_capcut_read_timeline_json`.

---

## 3. Ce qui devient du code mort

À supprimer, pas à commenter ni à garder « au cas où » : laissées en place, ces fonctions continueraient d'être testées et donneraient l'illusion que la longueur est pilotée.

**Constantes** (src/modules/analyzer.py)
- `SCRIPT_WORDS_PER_PANEL` (118), `SCRIPT_MIN_WORDS` (119), `SCRIPT_MAX_WORDS` (120)
- `SCRIPT_MIN_PARAGRAPHS` (122), `SCRIPT_MAX_PARAGRAPHS` (123)
- `SCRIPT_MIN_LENGTH_RATIO` (143)

**Fonctions** (src/modules/analyzer.py)
- `single_call_targets` (573-581)
- `target_script_words` (731-734)
- `target_paragraphs` (737-740)

**Gabarits** (src/modules/analyzer.py)
- `SCRIPT_SYSTEM_INSTRUCTION_TEMPLATE` (365-395)
- `SINGLE_CALL_SYSTEM_INSTRUCTION_TEMPLATE` (397-433)

**Blocs de code**
- `write_recap` : lignes 1762-1763, 1766, branche `too_short` 1767-1771, bloc 1776-1779, rappels 1785-1786, comparaison 1792-1798
- header 591-594 et footer 603-608

**Constantes du slicer**
- `SPLIT_MAX_HEIGHT` (73), `SPLIT_MAX_PIECES` (75), `SPLIT_SEARCH_RATIO` (79)
- `split_tall_range` (213-260) remplacée par `find_borders` + `split_segment_to_frame`

**Entrées `__all__` à retirer** : analyzer 1955, 1956, 1957, 1976, 1989, 1990 ; slicer 743-747.

**Documentation à corriger en même temps** (sinon deux descriptions contradictoires coexistent) : PRD.md:98-99 et 317 ; RESUME.md:377, 390, 888, 1232 ; REVUE_PIPELINE.md:161-162 ; docstrings slicer.py:21-24 et timeline_builder.py:8-10, 160-166, 653-654.

**Ne pas réintroduire** : aucun objectif, budget, plancher ou ratio de mots, sous aucune forme. Décision du propriétaire, cinq mécanismes mesurés, tous en échec.

---

## 4. Les pièges

**Ce qui casse le format court**
- Oublier `expand_to_unused_panels=True` à **format_factory.py:105-107**. La porte de timeline_builder.py:667 cesse de lire `max_clip_s` : le SHORT perdrait son élargissement en silence, et **aucun test existant ne l'attrape** (aucun test n'appelle `build_timeline` avec un profil SHORT). D'où `test_both_profiles_expand_to_unused_panels`.
- Toucher `make_pacing` (pacing.py:189-193). Le LONG basculerait sur `ShortPacing`, qui ignore `limit_panels_for_duration` et `min_clip_s` : plans < 1 s, plancher 2,5 s perdu, sous-titres 2-4 mots incohérents. La bonne conception est justement de laisser `make_pacing` lire `max_clip_s` et l'élargissement lire son propre drapeau.
- Passer `frame_height` au slicer en SHORT. Son affichage est un recadrage 9:16 par saillance, pas un `contain` : la loi de couverture n'y vaut rien. **Tester `fit == "contain"`, pas le nom du profil.**

**Contraintes dures**
- LONG : jamais d'agrandissement, jamais de rognage, zoom plafonné à 1,05. Les assertions de largeur (`test_slicer.py:314`) et d'égalité pixel (320-321) ne doivent **pas** être relâchées en adaptant le nombre de morceaux.
- Plancher de 2,5 s garanti par construction, pas par chance : `limit_panels_for_duration` (timeline_builder.py:467) plafonne n à `durée // min_clip_s`, puis `LongPacing` (pacing.py:111) donne `base = min_clip_s`.
- Prompts ASCII : `test_analyzer.py:259-261` impose `prompt.encode("ascii")`. Le master prompt est ASCII pur (vérifié), mais la fiche personnages injectée vient de bulles de webtoon — rien ne garantit qu'elle le soit.
- `Panel.part` est `Literal["top","middle","bottom"] | None` : n > 3 morceaux ne demande aucun changement de modèle, `middle` se répète.
- Ne rien ajouter dans `panels.json` (`test_slicer.py:600` fige le jeu de clés) : les paramètres de découpe vont dans un `slice_params.json` à côté.

**Quota (~20 appels/jour/modèle, 5 RPM, une clé)**
- Chaque motif ajouté à `FORBIDDEN_PATTERNS` peut déclencher la régénération unique : **garder un seul réessai maximum**.
- Rejeter une réponse sans `plan` coûterait un appel : d'où l'injection d'un plan vide par `fix_root` plutôt qu'une exception.
- Seuil de rejet des ids hors plage calibré **haut** et sur les ids **distincts** : un seul id inventé reste un simple retrait. Sinon on paie un appel pour rien, et `test_analyzer.py:959` casse.
- Troncature : un seul réessai (`TRUNCATION_RETRIES = 1`). Une fois le plafond fixé, un `MAX_TOKENS` signifie un emballement que le même prompt reproduira ; réessayer trois fois brûle trois appels.
- Un 400 `INVALID_ARGUMENT` sur `max_output_tokens` est aujourd'hui **fatal** : `is_invalid_key_error` (gemini_manager.py:121-128) ne reconnaît un 400 que s'il mentionne une clé, le reste tombe en `GeminiManagerError` (472) = chapitre perdu. Prévoir le classement en transitoire.
- Charge utile de la découpe : neutre sur ep1 (24,657 → 24,572 Mio), mais un chapitre **juste sous** 16 Mio (analyzer.py:140) qui passerait au-dessus basculerait de 1 à ~40 appels. Juger sur `payload_bytes`, pas sur le nombre de cases.
- La barrière d'ordre du batch **ne consomme aucun appel** ; elle coûte du temps de mur, pas du quota.

**Interblocage**
- `analyzed[url].set()` doit être posé dans un `finally` **et** sur la branche « déjà fait, ignoré » (batch_processor.py:343-345), et l'attente doit rester **hors** de `sem_chapters`. C'est le seul endroit du plan où un bug gèle un lot entier.

**Effets induits déjà mesurés, à surveiller**
- Bruitages : 22 → 37 sur mirror ep1 (+68 %), inchangés sur lazy-lord ep1. À écouter avant publication.
- Fonds floutés : un JPEG par case (capcut_builder.py:171), 100 → 180 fichiers par chapitre.
- Cache d'images de l'aperçu (preview_renderer.py:715, 721-730) sans éviction : ~80 % de cases en plus en mémoire.
- Durée totale de la vidéo inchangée (verrouillée sur l'audio), nombre de transitions inchangé.

---

## 5. Ce qu'on ne saura qu'en lançant

- **Le bloc IMAGES n'a jamais été envoyé à Gemini.** `t8_1.py:46` et `t8_3.py:46` découpent tous deux sur `[INPUT_BLOCK_BEATS]`. Or la production tourne en mode une requête par défaut (pipeline.py:114) — donc sur le bloc images. Tous les gains (verbes 8 → 0, vocabulaire 11 → 0) l'ont été en mode BEATS. → **Le test qui tranche** : un vrai chapitre en mode une requête, puis `ab_mesure.py` rejoué sur la sortie. Rien d'autre ne le dira.
- **Le bac à sable ne reproduit pas la production.** Bac : temperature 0,6, `gemini-3.5-flash-lite`, **sans** response_schema. Production : 0,4, `gemini-3.5-flash`, schéma pydantic. Le schéma peut supprimer le JSON invalide mais aussi changer le style. → **Test** : un appel réel avec le nouveau schéma imbriqué, comparé aux mêmes mesures.
- **`plan` requis ou optionnel côté schéma Gemini ?** Un champ optionnel produit un `anyOf/null` dont la conversion par le SDK google-genai n'est pas vérifiée. → **Test** : un appel réel avant de figer ; en attendant, champ requis + injection d'un plan vide par `fix_root`.
- **Le plan v7 est incohérent.** L'unique sortie disponible donne un cliffhanger `first_id=111 / last_id=37` et `peak_id=118` pour 38 beats. Il vaut comme étape de réflexion imposée, pas comme donnée. → **Test** : `test_the_plan_units_cover_every_story_id_exactly_once` (trou/recouvrement réparé et journalisé, borne hors plage rejetée). Ne pas piloter le montage avec.
- **32 768 jetons de sortie est une déduction**, pas une mesure : 11 608 pour le plus gros appel unique connu (hidden-rank ep9, n_batches=1). Aucun relevé par appel n'existe ; le plafond réel de chaque modèle de la cascade (gemini_manager.py:54-64) est invérifiable hors ligne. → **Test** : un seul appel réel sur le plus gros chapitre, avant de généraliser.
- **Un seul chapitre mesuré pour la découpe.** Tous les chiffres viennent de mirror ep1 (219 443 × 900, 102 segments). REVUE_PIPELINE.md cite des ep2/ep3 très différents (136 et 127 cases). Les seuils 0,50/0,60/0,70, bonus 0,20, grille 24 px, seuil de changement 24 sont calés sur ce seul corpus. → **Test** : rejouer `harness.py` sur ep2 et ep3 avant d'intégrer.
- **Rien n'a été rendu.** Le gain « 5 → 16 coupes sur une vraie bordure » est une mesure de signal. Personne n'a regardé si une bulle est encore tranchée (26 % des coutures aujourd'hui). La couverture 35,79 % → 41,55 % est un modèle, pas une image : la loi du banc plafonne l'échelle à 1,0 alors que le rendu tolère 1,05. → **Test** : un MP4 et un brouillon CapCut sur le nouveau découpage. Rien d'autre.
- **Le repli de la DP n'a jamais été déclenché** sur le corpus (strat_naturel.py:109-112). → **Test** : `test_split_segment_to_frame` avec une hauteur qui interdit tout pavage dans [min_piece, 1,6×frame_height].
- **Le vivier élargi contient des cases que Gemini avait écartées** comme transitions (analyzer.py:419-421). Impossible de dire quelle part des 482 cases réadmises est du dessin et quelle part du remplissage. Indice rassurant : hauteur médiane 887 px contre 962 px pour les clés. Les vrais filler sont négligeables (7 sur 1113, 0,6 %) — **ne pas coder le filtre maintenant**. → **Test** : un visionnage. `test_long_widening_never_shows_a_filler_scene_panel` protège le cas des cartes-titre, pas le jugement éditorial.
- **Le nombre de paragraphes reste non pilotable** (quatre formulations, toujours ~10). Le mécanisme UNITS est censé le remplacer. → **Test** : compter les paragraphes sur trois chapitres réels après intégration. Si c'est encore ~10, accepter — pas réintroduire une consigne.
- **Rien n'a été écouté.** La normalisation typographique des répliques citées destinée à Kokoro `am_puck` n'a jamais été vérifiée à l'oreille, alors que le master prompt augmente délibérément la part de dialogue cité. → **Test** : écouter un chapitre entier.
- **`_extract_json` peut produire un objet partiel parsable.** Le repli `find('{')`/`rfind('}')` pourrait, sur certaines coupures, rendre un script silencieusement amputé. Aucun contre-exemple construit. → **Test** : plusieurs points de coupure (fin de chaîne, fin d'objet, fin de tableau), pas un seul.
- **Règle de fusion des noms** : première orthographe canonique, variantes en `also_called` — ou l'inverse si le modèle corrige une erreur des premiers épisodes. Non départageable. → **Test** : un essai sur les épisodes 1 à 4 réels de Mirror World.
- **`episode_no` peut être `None`** (chapter.py:289) et c'est la seule clé d'ordre. Décision proposée : chapitre sans numéro = mémoire désactivée pour lui. → **Test** : `test_current_and_later_episodes_are_excluded` verrouille le cas numéroté ; le cas `None` demande une décision, pas une mesure.
- **Plafond de 20 fiches et 160 caractères par « who » non mesuré** ; coût estimé 500-800 jetons d'entrée. Effet sur le taux de JSON invalide (1 sur 3) inconnu. → **Test** : comparer le taux d'échec sur 5 chapitres avec et sans fiche injectée.
- **`click` 8.x et `mix_stderr`** : `result.output` peut ne contenir que stdout. `src/main.py` imprime en stdout, les logs vont en stderr. → **Test** : le premier `test_cli_help_lists_every_command` le confirme en une exécution.
- **Aucun `pytest.ini` ni `pyproject.toml`**, ni pytest-asyncio, ni pytest-mock. Les tests de concurrence restent sur `asyncio.run(process_batch(...))` (test_batch.py:162-166). Aucun marqueur asyncio, aucun mocker.