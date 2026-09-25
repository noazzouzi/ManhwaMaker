# Audit des prompts — Claude Opus 5.5

Date : 25/09/2026. Propositions dans `AUDIT_PROMPTS.diff` (19 blocs).
Mesurées puis appliquées, sauf les blocs 13-14 (M4) et 17 (M5) : voir « Résultats des mesures » en fin de rapport.

## Hypothèses

- **Périmètre** : tout ce que Claude lit dans le dossier du projet.
  - Le prompt du script (`src/modules/script_prompt.md`) et le code qui l'assemble et l'envoie (`script_writer.py`, `toonsplit/ai.py`).
  - Le juge du banc d'essai (`eval/script_bench/run_bench.py`).
  - Les prompts de toonsplit.
  - `CLAUDE.md`.
- **Hors périmètre** :
  - Le `CLAUDE.md` global et la mémoire (hors du dossier).
- **Modèle cible** : `claude-opus-5-5`, modèle de production depuis le 25/09 (`script_writer.DEFAULT_MODEL`).
- **Autre fournisseur** : Google Gemini (`google.genai`) dans `analyzer.py`, `gemini_manager.py` et `thumbnail/`.
  - Ces prompts visent Gemini : ils ne sont pas audités contre Claude.
  - Ils servent seulement à retracer d'où viennent certaines règles (M1).
- **Pas d'historique git** : `script_prompt.md` et `script_writer.py` ne sont pas encore commités.
  - L'origine des règles vient de la comparaison avec le prompt Gemini dont le prompt Claude est issu.

## Inventaire

| Surface | Fichier | Rôle |
|---|---|---|
| Prompt système du script | `src/modules/script_prompt.md` | Production, 1 appel par chapitre |
| Ouvertures et message utilisateur | `src/modules/script_writer.py:68-73`, `:213-214` | Production |
| Requête (CLI `claude -p`) | `src/modules/toonsplit/ai.py:498-506` | Commun à tous les appels Claude |
| Relance après rejet | `src/modules/toonsplit/ai.py:359-360` | Commun |
| Juge du banc d'essai | `eval/script_bench/run_bench.py:149-171`, `:184-185` | Mesure |
| Message du banc | `eval/script_bench/run_bench.py:92-93` | Copie du message de production |
| Prompts toonsplit (`SPEC_SYSTEM`, `JUDGE_SYSTEM`) | `src/modules/toonsplit/ai.py:220-262` | Outil de test, hors chaîne |
| Règles du projet | `CLAUDE.md` | Lues par Claude Code |

## Résumé

- 1 constat de confiance haute (groupe 4 : réglage de la requête).
- 7 constats de confiance moyenne :
  - 4 du groupe 1 : texte daté dans les prompts ;
  - 3 du groupe 2 : fichier de règles.
- 4 signalements sans modification proposée.

Les trois constats qui comptent le plus :

1. **L'effort n'est jamais fixé (H1).**
   - La réflexion d'Opus 5.5 ne se désactive pas : l'effort est le seul réglage de sa quantité, donc de la durée et du coût.
   - L'appel ne le précise pas, et `--setting-sources ""` écarte aussi les réglages utilisateur.
   - Les 81 s et 1,04 $ mesurés dépendent donc d'une valeur que rien ne fixe ni ne garantit.
2. **Les mots bannis viennent du prompt Gemini (M1).**
   - La liste reprend celle du prompt Gemini, écrite contre ses tics d'écriture.
   - Elle contredit le prompt lui-même : « epic » est banni, mais c'est aussi une des émotions imposées au champ `emotion`.
3. **Le format JSON est décrit deux fois (M3).**
   - Le prompt le redécrit avec un squelette et un exemple (« Mira », « the healer ») alors que `--json-schema` l'impose déjà.

## Constats

### H1 — Effort non fixé · confiance haute · ajouter

- **Emplacement** : `src/modules/script_writer.py:192`, sans `effort=` :
  ```python
  self.client = client or ClaudeCliJson(model=model, retries=retries, timeout_s=timeout_s)
  ```
  `toonsplit/ai.py:504` n'ajoute `--effort` que s'il est fourni.
- **Motif** : groupe 4, réglage de réflexion calé pour un autre modèle.
- **Pourquoi c'est daté** :
  - Sur Opus 5.5, la réflexion est toujours active et l'effort est le seul levier.
  - Côté API, l'effort par défaut est `medium`, un cran sous le `high` d'Opus 5.
  - La consigne de migration est de le fixer explicitement, puis de comparer les niveaux voisins.
  - Ici, l'effort dépend du défaut du CLI Claude Code, qui peut changer à chaque mise à jour.
- **Action (blocs 1 à 5 en production ; 10, 12, 15 et 16 dans le banc)** :
  - Ajouter la constante `DEFAULT_EFFORT = "medium"` (défaut API d'Opus 5.5).
  - Ajouter le paramètre `effort` à `ClaudeScriptWriter`.
  - Dans le banc d'essai, accepter `modele@effort` pour comparer les efforts (`--models opus@low,opus@medium,opus@high`).
- **À faire avant d'appliquer le bloc de production** :
  - Lancer la comparaison des efforts (voir « Vérification »).
  - Garder `medium` seulement si le juge le place au niveau de l'effort par défaut actuel.

### M1 — Mots bannis hérités de Gemini · confiance moyenne · réécrire

- **Emplacement** : `src/modules/script_prompt.md:47-49`
  > Banned words: massive, powerful, overwhelming, incredible, sheer, utter, colossal, epic, "little did he know". "suddenly", "realizes", "notices", "observes": at most twice in total.
- **Motif** : groupe 1e, interdictions de style sans origine ; groupe 1c, règles qui se contredisent.
- **Pourquoi c'est daté** :
  - La liste reprend `analyzer.py:477-487`, le prompt Gemini (« Banned: massive, powerful, overwhelming, incredible, glowing, sheer, utter, colossal »).
  - Elle a été écrite contre les tics de Gemini, et le prompt Claude a perdu la raison qui l'accompagnait (« Show the thing; never announce that someone perceives it »).
  - Sur un modèle récent, interdire une faute qu'il ne commettait pas peut l'y attirer.
  - « epic » est à la fois banni et imposé comme valeur d'émotion (`{emotions}`, ligne 75).
- **Action (bloc 8)** : réécrire en consigne positive, avec la raison.
  > size and danger come from what happens, not from intensifiers. Show what a character discovers instead of announcing that they perceive it (not "She realizes the letter is forged" but "The seal on the letter is still wet.").
- **Contrôle** : les mesures `BANNED` et `PERCEPTION` du banc (`run_bench.py:197-198`) restent en place. Elles diront si ces mots reviennent.

### M2 — « Lis tout avant d'écrire » · confiance moyenne · réécrire

- **Emplacements** :
  - `src/modules/script_prompt.md:8-9` : « Read the whole chapter before you write a single word ».
  - `src/modules/script_writer.py:214` et sa copie `eval/script_bench/run_bench.py:93` : « Read them all, then write the script. »
- **Motif** : groupe 1b, consigne de planifier avant d'agir.
- **Pourquoi c'est daté** :
  - En un seul appel, le modèle a déjà toutes les planches avant d'écrire.
  - Sur Opus 5.5, la réflexion est toujours active.
  - Ce type de consigne pousse à trop planifier, ce qui coûte de la réflexion facturée sans rien apporter.
- **Action (blocs 6, 7 et 11)** :
  - Garder le but : « Build the script towards the moment everything turns on. »
  - Remplacer la fin du message par « Write the script. », en production et dans le banc (mêmes entrées pour les deux).

### M3 — JSON redécrit alors que le schéma l'impose · confiance moyenne · réécrire

- **Emplacement** : `src/modules/script_prompt.md:71-74` :
  - « One JSON object, nothing else: » ;
  - suivi d'un squelette JSON avec des valeurs d'exemple (« Mira », « the healer »).
- **Motif** : groupe 1b, structure remplacée par une fonction de l'API.
- **Pourquoi c'est daté** :
  - Le format est déjà imposé par `--json-schema` pour Claude (`toonsplit/ai.py:502`) et par `response_schema` pour la variante Gemini du banc.
  - Demander « du JSON et rien d'autre » était le palliatif des modèles sans sortie structurée.
  - Le squelette fige aussi des valeurs d'exemple que le modèle risque d'imiter.
- **Action (bloc 9)** :
  - Remplacer le squelette par « The answer's format is enforced by a schema. What each field means: ».
  - Garder le sens de chaque champ, qui est le contrat.
  - Ajouter le sens de `also_called` et `who`, que seul le squelette donnait (« One sentence. »).
  - Le test `test_script_writer.py:78` passe toujours.

### M4 — Même consigne de lecture dans le juge du banc · confiance moyenne · réécrire

- **Emplacements** :
  - `eval/script_bench/run_bench.py:152` : « Read the whole chapter first, then every script. »
  - `:185` : « Read the panels, then score the scripts. »
- **Motif** : groupe 1b, comme M2.
- **Action (blocs 13 et 14)** :
  - Supprimer la consigne d'ordre de lecture.
  - Garder la description de ce qui est envoyé (scripts, puis planches), qui est une information utile.
- **Limite** : les notes du juge ne seront plus strictement comparables aux notes déjà enregistrées (Opus 8,5, Sonnet 6,5…). À appliquer avec une nouvelle série complète.

### M5 — Plafond chiffré des réponses · confiance moyenne · réécrire

- **Emplacement** : `CLAUDE.md:10` : « 10 puces maximum. Si le sujet en demande plus, c'est qu'il va dans un fichier. »
- **Motif** : groupe 1f, plafond chiffré de la sortie (exemple cité par le guide : « at most five bullets »).
- **Pourquoi c'est daté** :
  - Un plafond chiffré pousse à couper ce qui compte pour tenir le compte.
  - Un but de lecture se suit mieux.
- **Action (bloc 17)** : « Une réponse se lit d'un coup d'œil : seulement ce qui compte pour ma décision. Un sujet plus long va dans un fichier. »
- **À toi de voir** : c'est ta préférence de lecture. Refuse ce bloc si le chiffre te sert de repère.

### M6 — Nombre de tests écrit à la main · confiance moyenne · réécrire

- **Emplacement** : `CLAUDE.md:29` : « 490 tests, ~30 s ».
- **Motif** : groupe 2, précisions qui périment.
- **Pourquoi** :
  - Le nombre change à chaque session et doit être corrigé à la main.
  - Tant qu'il ne l'est pas, il est faux : il affirme quelque chose que `pytest` dit mieux.
- **Action (bloc 18)** : garder la durée et l'option `TOONSPLIT_SLOW`, retirer le nombre.

### M7 — Chiffres d'un modèle abandonné · confiance moyenne · supprimer

- **Emplacement** : `CLAUDE.md:59` : « Opus 5 au banc d'essai : 2 à 3,5 min, ~1,45 $. »
- **Motif** : groupe 2, historique et nom de modèle figé.
- **Pourquoi** : Opus 5 n'est plus utilisé. La règle utile est la mesure d'Opus 5.5, qui reste.
- **Action (bloc 19)** : supprimer la phrase.

## Signalements (aucune modification proposée)

- **S1 — toonsplit reste sur Opus 5** (`src/modules/toonsplit/ai.py:425`, `CLAUDE_DEFAULT_MODEL = "claude-opus-5"`).
  - Ses prompts ont été réglés sur Opus 5.
  - Changer de modèle vide le cache toonsplit (le modèle fait partie de la clé) et relance tous les appels.
  - Outil hors chaîne de production : à décider si toonsplit revient dans la chaîne.
- **S2 — « Return only the JSON object. »** (`toonsplit/ai.py:251`, `:262`).
  - Même motif que M3.
  - Le modifier impose d'incrémenter `SPEC_VERSION` / `JUDGE_VERSION`, sinon le cache sert d'anciennes réponses. Tous les blocs seraient alors recalculés.
  - Même décision que S1.
- **S3 — Refus du modèle non distingués** (`toonsplit/ai.py:541-548`).
  - Opus 5.5 peut refuser une requête (catégories `cyber`, `bio`, `reasoning_extraction`).
  - Par le CLI, un refus arrive sans doute comme une réponse non JSON. Il serait alors relancé une fois (~1 $) avant le repli Gemini.
  - Risque faible pour du manhwa, et la forme exacte du refus côté CLI n'a pas été vérifiée.
- **S4 — Hors grille : le héros est supposé masculin** (`script_writer.py:70`, « what he stands to lose »).
  - Ce n'est pas un motif daté.
  - Mais sur une série à héroïne, la consigne d'ouverture parle d'un « he ».

## Ce qui est propre

- Pas de préremplissage de réponse.
- Pas de réglage de température côté Claude (seulement Gemini).
- Pas de `budget_tokens` ni de réflexion désactivée.
- Pas de `tool_choice` forcé.
- Pas de consigne demandant de reproduire la réflexion (le champ `reason` du juge toonsplit est un verdict, pas un raisonnement).
- Pas de modification de l'historique : chaque appel est une session d'un seul échange, et la relance repart d'une session neuve.
- Un seul appel au modèle par chapitre, et il est justifié (écriture du script) ; les contrôles de cases sont faits dans le code (`normalize_recap`).
- Le coût est suivi à chaque appel (`cost_usd`, journal du pipeline).
- La relance après rejet (`ai.py:359`) sert à une validation de sens (numéros de cases), pas à réparer du JSON : elle reste.
- Les bandeaux « Panel N » sur les planches sont un contrat de données, pas une aide à la lecture d'image : ils restent.
- Les règles du « quatrième mur », la limite « Our hero » et les limites de citations viennent de décisions produit, avec leur raison : elles restent.

## Vérification

Les tests liés passent sur une copie avec tout le diff appliqué : `test_script_writer.py` et `test_toonsplit_ai.py`, 30/30.

Une suppression dans un prompt reste une hypothèse. Pour la vérifier, sur les deux chapitres du banc d'essai (dossiers présents) :

```powershell
.\.venv\Scripts\python.exe eval\script_bench\run_bench.py the-lazy-lord-masters-the-sword_ep1 the-unparalleled-hidden-rank-equipment_ep2 --models opus,opus@low,opus@medium,opus@high --judge
```

- **Coût** : environ 4 scripts × 2 chapitres × ~1 $, plus 2 appels du juge, soit **~10 $ au tarif API**, décomptés de l'abonnement.
- **Ordre conseillé** :
  1. Blocs du banc 10, 12, 15 et 16 seuls, avec l'ancien prompt : choisir l'effort.
  2. Puis M1, M2 et M3 ensemble, avec l'effort retenu, et relancer le banc.
  3. Garder si le juge ne baisse pas et si `BANNED` / `PERCEPTION` restent bas.
- **Si un bloc fait baisser la note** : remettre la consigne sous sa forme la plus courte, pas l'ancienne version longue.

## Résultats des mesures (25/09, 23:19-23:25)

Opus 5.5, 2 chapitres (Lazy Lord ch. 1, Hidden Rank ch. 2), un script par variante, juge du banc inchangé.
Détail et scripts : `output/script_bench_audit/index.html` et `summary.json`. Coût total : **11,34 $** au tarif API.

| Variante | Note globale (ch. 1 / ch. 2) | Moyenne | Erreurs relevées | Durée (s) | Coût ($) |
|---|---|---|---|---|---|
| Prompt actuel, effort par défaut du CLI (production) | 7 / 7 | 7,0 | 5 + 0 | 81 / 58 | 0,95 / 0,90 |
| Prompt actuel, `low` | 5 / 5 | 5,0 | 5 + 4 | 35 / 28 | 0,85 / 0,83 |
| Prompt actuel, `medium` | 6 / 8 | 7,0 | 3 + 2 | 84 / 68 | 0,97 / 0,92 |
| Prompt actuel, `high` | 8 / 6 | 7,0 | 3 + 3 | 175 / 87 | 1,16 / 0,97 |
| Prompt allégé (M1-M3), `medium` | 8 / 7 | **7,5** | 5 + 2 | 72 / 57 | 0,93 / 0,90 |

**Effort :**
- `medium` coûte et dure comme le défaut actuel : le CLI tourne sans doute déjà en `medium`. Le fixer protège d'un changement du CLI.
- `low` donne des scripts deux fois plus courts, avec plus d'erreurs.
- `high` est jusqu'à 2× plus lent et coûte 13 % de plus, sans gain constant.

**Prompt allégé :**
- 2e aux deux chapitres, meilleure moyenne, 3 à 15 % plus rapide.
- Les mots surveillés reviennent à peine : 1 « realizes » et 1 « powerful », au sens propre (« one of the most powerful classes »).
- Les alertes « 4e mur » (« pages », « panels ») sont des objets de l'histoire (le journal intime, les fenêtres du Système), dans toutes les variantes.

**Limite :**
- Une seule rédaction par variante.
- Le défaut et `medium` (même réglage probable) diffèrent d'un point dans un sens puis dans l'autre : le bruit du juge est d'environ ±1.
- Conclusion : aucune régression, amélioration non prouvée.

**Appliqué :**
- H1 (production et banc), M1, M2, M3, M6, M7.
- Tests : 490 passent.

**Non appliqué :**
- M4 (juge) : le changer romprait la comparaison avec ces notes.
- M5 (« 10 puces maximum ») : ta préférence.
