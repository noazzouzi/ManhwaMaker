# PRD — Auto-Manhwa Recap Generator

## 1. Vision

CLI Python qui transforme l'URL d'un chapitre **Webtoons.com** en un **projet CapCut (.draft)** prêt à
être ouvert, relu et exporté en **1920x1080 @ 60 FPS** pour YouTube.

```
URL chapitre Webtoons
        │
        ▼
┌──────────────────┐   bande verticale    ┌──────────────────┐   cases + métadonnées
│ 1. Scraper &     │ ───────────────────▶ │ 2. Smart Slicer  │ ─────────────────────┐
│    Stitcher      │   (PIL, en mémoire)  │    (OpenCV)      │                      │
└──────────────────┘                      └──────────────────┘                      ▼
                                                                          ┌──────────────────┐
                                                                          │ 3. VLM Analyzer  │
                                                                          │    (Gemini)      │
                                                                          └──────────────────┘
                                                                                    │ scènes JSON
                                                                                    ▼
┌──────────────────┐   .wav + durées      ┌──────────────────┐
│ 5. CapCut Draft  │ ◀─────────────────── │ 4. Kokoro TTS    │
│    Builder       │                      │    (local)       │
└──────────────────┘                      └──────────────────┘
        │
        ▼
 output/<projet>/draft_content.json (+ assets)
```

- **Entrée** : `https://www.webtoons.com/<lang>/<genre>/<titre>/<episode>/viewer?title_no=…&episode_no=…`
- **Sortie** : dossier projet CapCut + assets (cases PNG, voix WAV, sous-titres) dans `output/`.
- **Langue par défaut : anglais** partout (chapitres lus sur `webtoons.com/en/`, narration
  générée, voix off), chaque module acceptant une surcharge explicite.

## 2. Stack technique

| Rôle | Outil |
|---|---|
| Langage | Python 3.11+ |
| CLI | Typer |
| Scraping | `requests` + `beautifulsoup4` (headers `User-Agent` + `Referer` obligatoires) |
| Image | Pillow (stitch), OpenCV + NumPy (slicing) |
| VLM / LLM | `google-genai` (Gemini 3.5 Flash par défaut, cascade 3.6 → 3.7 → 3.8 → 2.5 Flash → pro-latest ; multi-clés) |
| TTS | Kokoro (`KPipeline`, 100 % local) + `soundfile` / `pydub` |
| Schémas | Pydantic |
| Montage | Générateur JSON `draft_content` compatible CapCut (pyCapCut) |

## 3. Architecture des modules

Arborescence :

```
ManhwaMaker/
├── PRD.md
├── Consignes.md
├── requirements.txt
├── .env.example      # GEMINI_API_KEYS=cle1,cle2 (copier en .env), GEMINI_MODEL_CASCADE
├── config/           # pronunciations.json, sfx/ (bruitages), bgm/ (musiques par ambiance)
├── src/
│   ├── main.py       # CLI Typer (run / batch / preview / capcut)
│   ├── pipeline.py   # étapes (scrape+slice, analyze, tts, montage) avec réutilisation des sorties
│   ├── modules/      # scraper, slicer, analyzer, tts_engine, timeline_builder, capcut_builder,
│   │                 # preview_renderer, batch_processor (lots parallèles, batch_status.json)
│   ├── models/       # schémas Pydantic (Panel, Scene, SceneAudio, Timeline, …)
│   └── utils/        # config (.env, clés), gemini_manager (multi-clés, cascade, RPM), http,
│                     # image_utils, audio_assets, report
├── tests/            # tests unitaires + scripts de validation locale
├── batch_status.json # suivi d'un lot (généré, ignoré par git)
└── output/           # artefacts générés (ignoré par git)
```

### Module 1 — Scraper & Stitcher (`src/modules/scraper.py`)

- Normalise l'URL vers la version anglaise du site (`/fr/…` → `/en/…`) sauf demande contraire
  (`--site-language`, ou `language=None` en librairie).
- Télécharge la page du chapitre puis toutes les images du viewer (`img._images`, attribut `data-url`).
- **Headers HTTP obligatoires** sur chaque requête (page + images) :
  `User-Agent` (navigateur desktop) et `Referer: https://www.webtoons.com/`. Sans eux : HTTP 403.
- Téléchargement séquentiel avec retries/backoff, images conservées **en mémoire** (PIL).
- `stitch_webtoon_pages(images)` : normalise la largeur, convertit en RGB, puis concatène
  verticalement les morceaux (1280 px de haut chacun) en **une seule bande continue** via `np.vstack`.
- Sortie : `PIL.Image` (bande complète) + métadonnées (titre, n° épisode, nb de morceaux).

### Module 2 — Smart Slicer (`src/modules/slicer.py`)

- Entrée : bande continue (PIL ou `np.ndarray`).
- Calcule pour chaque **ligne de pixels** la variance en niveaux de gris ; une ligne de variance
  `< variance_threshold` est une ligne de « gouttière » (fond blanc ou noir uniforme).
- Un run de lignes de gouttière d'au moins `min_gap` px sépare deux cases.
- Chaque case est étendue de `margin_padding` px de chaque côté (bornée à l'image).
- **Paramètres par défaut** : `variance_threshold = 15.0`, `min_gap = 20`, `margin_padding = 15`.
- **Fusion des petites cases** : une suite de cases voisines de moins de 250 px chacune
  (bulles de chat, lignes de texte), séparées d'au plus 150 px, devient une seule case.
- **Filtre micro-cases** : `MIN_PANEL_HEIGHT = 180` px, toute case plus petite (bruitage,
  onomatopée, « tick », objet isolé) est éliminée.
- **Sous-découpage des cases très hautes** : une case (padding inclus) de plus de
  `SPLIT_MAX_HEIGHT = 1200 px` est coupée **strictement horizontalement** en 2 ou 3 blocs
  (`ceil(hauteur / 1200)`, 3 au plus) qui conservent **100 % de la largeur d'origine**
  (`part` = `top` / `middle` / `bottom`, même `source_index`). Chaque coupe est placée sur la
  ligne de plus faible variance dans une fenêtre de ± 15 % de la hauteur d'un bloc autour de la
  coupe idéale, pour ne pas trancher un visage ou une bulle. Aucun algorithme ne rogne jamais une
  case en largeur.
- **Filtre cases géantes** : toute case de hauteur `> 1500 px` reçoit `"type": "scroll_vertical"` ;
  les autres sont `"type": "static"`. Ce type n'est plus qu'un libellé : le montage n'applique
  aucun défilement (voir Module 5).
- Sortie : liste de `Panel` (Pydantic) `{index, y_start, y_end, height, width, type, part,
  source_index, image}`.

### Module 3 — Gemini VLM Analyzer (`src/modules/analyzer.py`)

- Envoie les cases par **lots de 10 à 15 images max** à Gemini (évite les timeouts) :
  `batch_size = 12` par défaut, lots équilibrés (20 cases → 2 lots de 10), jamais plus de 15.
- Chaque case est envoyée en JPEG numéroté (« Case N »), largeur bornée à 1024 px ; une case
  géante (`scroll_vertical`) est découpée en tranches de 2048 px max (4 au plus) envoyées de haut
  en bas sous le même numéro, pour rester lisible sans exploser les tokens image.
- **Mode « une requête » (par défaut depuis le 2026-09-12)** : toutes les cases du chapitre
  partent dans **un seul appel** qui renvoie directement les paragraphes du script avec leurs
  cases clés et leurs cases `action_heavy` (schéma `RecapDraft`). Mesuré sur un chapitre de
  176 cases : 196 000 jetons d'entrée et 13,8 Mo d'images, contre 241 000 jetons répartis sur
  18 appels en mode deux étapes — les images n'étant plus envoyées deux fois. Garde-fous :
  - **bascule automatique** vers le mode deux étapes si les images dépassent
    `MAX_INLINE_PAYLOAD_BYTES` (16 Mo, marge sous la limite de 20 Mo de Gemini) ;
  - **numéros de cases** validés contre la liste réelle, jamais réutilisés d'un paragraphe à
    l'autre, bornés à 4 par paragraphe, repli sur la plus grande case libre ;
  - **longueur** : le mode a tendance à trop résumer (478 mots pour 1441 visés au premier essai),
    donc un script sous `SCRIPT_MIN_LENGTH_RATIO` (65 %) de la cible est régénéré une fois avec
    un rappel chiffré ; la reprise n'est gardée que si elle est plus fournie ;
  - formules visuelles interdites, accroche obligatoire et appel à l'abonnement unique :
    mêmes contrôles qu'en mode deux étapes.
  - `--multi-call` force l'ancien mode (utile pour retrouver la traçabilité des *beats*, que le
    mode une requête ne produit pas).
- **Génération en deux étapes** (`--multi-call`, schémas Pydantic en `response_schema`) :
  - *Étape 1a, beats* : par lot, Gemini extrait des beats factuels `{panel_ids, summary,
    characters, dialogue, is_filler}` ; chaque case du lot appartient à exactement un beat
    (normalisation en partition consécutive), contexte glissant des 3 derniers beats.
  - *Étape 1b, script global* : à partir de tous les beats (texte seul), Gemini rédige le
    script complet du récap en paragraphes `{text, beat_ids, emotion}` : storytelling purement
    rétrospectif, présent, 3ᵉ personne, **interdiction stricte des formules visuelles**
    (« dans cette case », « on voit », « ici », « la scène montre »...). **Accroche obligatoire**
    dès la première phrase (urgence, menace, mystère, fait choc tiré de l'histoire), **intro
    générique interdite** (« In this chapter », « Welcome back », « Let's dive in »...). **Un seul
    appel à l'abonnement**, canonique par langue (`--cta` pour le personnaliser, `--no-cta` pour
    l'omettre), placé vers 40 % du script, jamais dans les deux premiers paragraphes ni en
    dernière phrase ; ceux écrits par le modèle sont retirés. Formules interdites et intro
    générique sont détectées par expressions régulières, le script est régénéré une fois avec
    rappel, puis nettoyé localement en dernier recours. Les beats de remplissage ne sont jamais
    narrés.
  - *Étape 2, cases clés* : pour chaque paragraphe, Gemini choisit parmi les cases de ses beats
    (images) 1 à 4 **cases clés** ; les cases de transition (texte seul, bruitages, fragments)
    ne sont pas retenues et ne sont pas montées. Repli sur la plus grande case candidate.
    Gemini marque aussi, parmi ces cases clés, celles qui montrent un **impact décisif**
    (`action_heavy_ids` : coup porté, explosion, rugissement, révélation) ; elles reçoivent un
    zoom d'impact au montage. Les numéros hors cases clés sont ignorés.
  - `emotion` parmi `neutral, calm, tension, action, sad, happy, humor, romance, fear, mystery, epic`.
- Tous les appels passent par le **gestionnaire Gemini** (`src/utils/gemini_manager.py`, voir
  Module 6) : rotation de clés, backoff 2/4/8/16 s sur 429, cascade de modèles, limite RPM
  globale. Les réponses invalides (JSON) sont réessayées par l'analyzer ; prompt bloqué par la
  sécurité = erreur définitive avec la raison. Un **délai forcé de 2,5 s** (`BATCH_DELAY_S`)
  sépare deux envois d'un même chapitre (lots d'images, script, groupes de cases clés) pour
  étaler la consommation de jetons par minute (TPM).
- Modèle préféré `gemini-3.5-flash` (`GEMINI_MODEL` ou `--model`), suivi de la cascade de
  secours ; budget de réflexion réglable (`thinking_budget`). Modèles servis sur ce compte au
  2026-09-10 : 2.5-flash / -lite / -pro, 3-flash-preview, 3.1-flash-lite, 3.1-pro-preview,
  3.5-flash / -lite, 3.6 / 3.7 / 3.8-flash, flash-latest, pro-latest ; **`gemini-2.0-flash`,
  `gemini-1.5-flash` et `gemini-1.5-pro` ne sont plus servis** (404).
- **Quotas** : une analyse en **une requête** consomme 1 à 2 appels par chapitre (2 quand le
  garde-fou de longueur se déclenche), soit **10 à 20 chapitres par jour, par modèle et par
  clé** sur l'offre gratuite au lieu d'un seul. Une analyse en deux étapes consomme 12 à 17
  requêtes par chapitre. Sur l'offre
  gratuite, le quota **journalier** par modèle et par clé est très bas (≈ 20 requêtes/jour),
  soit un chapitre par jour, par modèle et par clé : plusieurs clés dans `GEMINI_API_KEYS`
  et/ou la facturation sont nécessaires pour un usage réel (coût constaté ~0,05 $ par
  chapitre). Toutes clés et tous modèles épuisés → `QuotaExhaustedError` immédiate avec la
  marche à suivre.
- **Point de reprise** (`analysis_checkpoint.json` dans le dossier du chapitre) : chaque appel
  réussi (lot de beats, script, groupe de cases clés) y est consigné ; une analyse coupée par le
  quota repart de là au prochain `run`, y compris avec un autre `--model`. Le fichier n'est
  valable que pour la même découpe, la même taille de lot et la même langue, et il est supprimé
  quand `scenes.json` est écrit (`--force` l'ignore).
- Sortie : `ChapterAnalysis` (scènes = paragraphes du script avec leurs cases clés et leurs
  beats, liste des beats, titres, modèle, tokens entrée / sortie / réflexion) → `scenes.json`.

### Module 4 — Kokoro TTS Engine (`src/modules/tts_engine.py`)

- Voix off **en anglais par défaut** (voix Kokoro américaine masculine `am_puck`, pipeline `a`),
  alignée sur la langue de la narration ; autres langues via `LANGUAGE_VOICES` (fr → `ff_siwis`).
- `KPipeline` local (modèle `hexgrad/Kokoro-82M`, ~330 Mo téléchargés une fois depuis Hugging
  Face) ; dictionnaire de **remplacement phonétique** `config/pronunciations.json` appliqué avant
  synthèse (mots entiers, insensible à la casse ; valeur `/phonèmes/` transmise telle quelle).
- Nettoyage du texte : guillemets typographiques, caractères invisibles, ponctuation finale.
- Un `.wav` 24 kHz 16 bits par scène narrative (le remplissage `is_filler` est exclu) ; chaque
  phrase est synthétisée séparément puis assemblée avec **0,4 s de pause entre les phrases**, et
  **0,18 s de silence** en fin de segment (rythme resserré entre les scènes, sans chevauchement),
  plus `voiceover_full.wav` pour écoute. Les deux durées sont réglables (`--sentence-gap`,
  `--padding`) et `--redo tts` les applique sans relancer l'analyse Gemini.
- `voiceover.json` : durée **exacte** de chaque segment (échantillons / fréquence), parole seule
  et totale, texte réellement synthétisé, pour caler la timeline CapCut.

### Module 5 — Timeline, CapCut Draft Builder et aperçu

- **Timeline** (`src/modules/timeline_builder.py`, `timeline.json`) : plan de montage commun au
  brouillon CapCut et à l'aperçu. Chaque scène narrée occupe **exactement** la durée de son
  segment audio ; ses cases clés se partagent ce temps au prorata de leur hauteur (poids
  minimal 400 px). **Rythme** : chaque case reste au moins 2,5 s à l'écran ; si une scène a trop
  de cases pour sa durée, les plus petites sont écartées plutôt que d'enchaîner trop vite.
  - **Mouvement** par case : `ken_burns` (100 % → 105 %) ou `punch_in` (105 % atteint en
    0,2 s puis maintenu) pour les cases `action_heavy`. **Jamais de zoom au-delà de 105 %**
    (`MAX_ZOOM`) de la taille native, jamais de défilement, jamais de recadrage : une case
    haute a été coupée en blocs par le slicer et chaque bloc est montré entier.
  - **Sous-titres** : narration découpée en **blocs de 2 à 4 mots** (jamais à cheval sur deux
    phrases, groupes équilibrés), répartis sur la durée de parole au prorata des caractères
    (≥ 0,25 s par bloc).
  - **Design sonore** : un **bruitage court** au début de chaque case des scènes d'émotion
    `action` (cycle impact → swoosh → roar → swoosh, −12 dB) et un `impact` au début de chaque
    case `action_heavy` des autres scènes (Gemini tague souvent les combats `tension` ou `epic`),
    lu dans `config/sfx/<kind>.wav|mp3` ;
    à défaut, des bruitages de substitution sont synthétisés (`src/utils/audio_assets.py`) à
    remplacer par de vrais effets sous les mêmes noms (`--sfx-dir`, `--no-sfx`). **Musique par
    ambiance** : les émotions sont ramenées à trois ambiances (`calm` : neutral/calm/happy/
    romance/humor ; `tense` : tension/mystery/fear/sad ; `action` : action/epic) ; à chaque
    changement d'ambiance la piste change avec un **fondu enchaîné de 1,5 s**, musiques lues dans
    `config/bgm/<mood>.wav|mp3` (`default` en repli, `--bgm-dir`) ; `--bgm fichier` garde une
    musique unique ; `--no-bgm` la coupe. Niveau **−22 dB** sous la voix Kokoro. Si le dossier
    ne contient aucune musique, trois **boucles d'ambiance de substitution** (nappe calme,
    bourdon tendu, pulsation d'action, ~24 s) sont synthétisées pour qu'un montage n'ait jamais
    de silence musical ; elles sont à remplacer par de vraies musiques (droits) sous les mêmes
    noms.
- **CapCut** (`src/modules/capcut_builder.py`, via `pycapcut`) : séquence 1920x1080 @ 60 FPS.
  - **V1** : fond **pré-rendu** en JPEG (case étirée pour couvrir le cadre, flou gaussien,
    −30 % de luminosité), produit par le même code que l'aperçu, dans `backgrounds/`.
  - **V2** : case PNG **à sa résolution native**, centrée dans le cadre 1920x1080 (échelle
    CapCut = 1 / ajustement « contain ») ; une case plus grande que le cadre est réduite pour y
    tenir, jamais agrandie ; le fond flouté de V1 comble les bandes 16:9 de part et d'autre.
    Images clés d'échelle uniquement : Ken Burns 100 % → 105 % sur la durée du clip, ou
    **punch-in 105 % en 0,2 s** puis maintien (cases `action_heavy`) ; jamais de position ni de
    recadrage.
  - **A1** : un segment WAV par scène ; **A2 / A2b** : musique par ambiance à **−22 dB**, liée
    par chemin absolu (`.wav` / `.mp3`), **bouclée** en segments contigus (`loop_ranges`) quand
    la vidéo dure plus longtemps que la musique, sur deux pistes alternées ; fondu d'entrée sur
    le premier segment de chaque ambiance et de sortie sur le dernier (`add_fade`) pour le fondu
    enchaîné ; **A3** : bruitages (un segment par transition, −12 dB).
  - **T1** : un segment texte par bloc de 2-4 mots, police grasse CapCut (`Rubik_Bold`, sinon
    `Montserrat`, `Anton`, `BebasNeue`), blanc, taille 9, **contour noir fin** (largeur CapCut
    20 ≈ 2 px en 1080p ; la valeur 40 par défaut de CapCut donne ~4 px), sans bandeau, centré
    en bas (`transform_y = −0,78`).
  - Temps en microsecondes entières calculés de proche en proche (segments contigus, sans
    chevauchement). Brouillon écrit dans `capcut/<nom>/` et copié dans le dossier des projets
    CapCut s'il est détecté (`%LOCALAPPDATA%\CapCut\User Data\Projects\com.lveditor.draft`) ;
    si le projet homonyme est ouvert dans CapCut (verrouillé), la copie prend le nom `<nom> (2)`.
  - À confirmer dans CapCut : disponibilité des polices `Rubik_Bold` / `Montserrat`, rendu des
    fondus et largeur exacte du contour.
- **Aperçu** (`src/modules/preview_renderer.py`) : rendu ffmpeg (binaire embarqué par
  `imageio-ffmpeg`) de la même timeline, H.264 + AAC, par défaut la première minute
  (`--preview-seconds`), pour contrôler le rendu sans ouvrir CapCut : même placement natif
  centré (réduction seulement si la case dépasse le cadre, jamais d'agrandissement), même Ken
  Burns et punch-in bornés à 105 %, sous-titres en police grasse système (Montserrat / Rubik si
  installées, sinon Arial Bold) avec contour noir de 2 px, mixage voix + musique (boucle,
  fondus, −22 dB) + bruitages.

### Module 6 — Production : gestionnaire Gemini multi-clés et orchestrateur de lots

- **Gestionnaire Gemini** (`src/utils/gemini_manager.py`, classe `GeminiManager`, un seul
  exemplaire partagé par tous les chapitres, thread-safe) :
  - **Clés** : `GEMINI_API_KEYS=cle1,cle2,...` dans `.env` (chargé par `src/utils/config.py`,
    sans dépendance) ou dans l'environnement ; sinon `GEMINI_API_KEY` / `GOOGLE_API_KEY` ou
    `.gemini_key` (plusieurs clés possibles). La clé active est conservée d'un appel à l'autre ;
    rotation vers la suivante quand elle est **invalide / expirée** (400 `API_KEY_INVALID`, 401,
    403) ou quand son **quota journalier (RPD)** est atteint pour le modèle courant. Les clés ne
    sont jamais journalisées (`...` + 4 derniers caractères).
  - **Rate limit** : sur 429 par minute (`RESOURCE_EXHAUSTED`), 5xx ou coupure réseau, réessai
    avec **backoff exponentiel 2 s, 4 s, 8 s, 16 s** (le délai suggéré par l'API est honoré s'il
    est plus long, plafond 65 s), **4 essais au plus par clé** puis rotation. Si **toutes** les
    clés sont limitées par minute sur le modèle courant (cas d'une clé unique), le gestionnaire
    **patiente 30 s et réessaie le même modèle** (2 fois au plus) au lieu de cascader : le quota
    journalier du modèle est encore disponible, seule la minute est saturée.
  - **Cascade de modèles** : quand toutes les clés ont échoué sur le modèle courant, bascule sur
    le suivant de `GEMINI_MODEL_CASCADE` (défaut : 3.5-flash → 3.6-flash → 3.7-flash →
    3.8-flash → 2.5-flash → 3.5-flash-lite → 3.1-flash-lite → 2.5-flash-lite → pro-latest ; un
    modèle retiré / 404 est sauté). **Chaque modèle a son propre quota journalier** : la
    longueur de la cascade détermine le nombre de chapitres traitables par jour et par clé, les
    Flash-Lite servant de réserve (qualité moindre sur l'analyse d'images). Tout épuisé →
    `QuotaExhaustedError`.
  - **RPM global** : fenêtre glissante de 60 s bornant les requêtes démarrées par minute,
    toutes clés et tous chapitres confondus (`--max-gemini-rpm`, défaut 10 ; **5 sur l'offre
    gratuite**, qui limite à 5 requêtes/min et par modèle).
  - **Mesures** : temps de réponse de chaque appel (médiane, p95, max, cumul) et compteurs
    par modèle (`ok` / `échec` / secondes), repris dans le rapport et dans `batch_status.json`.
- **Orchestrateur de lots** (`src/modules/batch_processor.py`, commande `batch`) :
  - cibles : `--url-list fichier` (une URL par ligne), ou une URL de série / d'épisode avec
    `--start-chapter` / `--end-chapter` ; les épisodes sont **découverts sur la page de liste**
    Webtoons (pagination `&page=N`, URL canoniques) et, à défaut, dérivés de `episode_no`
    (Webtoons redirige vers le slug canonique) ; `--dry-run` liste les cibles ;
  - le pipeline est découpé en étapes (`stage_scrape_slice`, `stage_analyze`, `stage_tts`,
    `stage_montage` dans `src/pipeline.py`) exécutées dans des threads et orchestrées par
    `asyncio.gather` sous des **sémaphores** : `--max-chapters` (défaut 5, chapitres scrapés +
    analysés simultanément, I/O), `--max-tts-workers` (défaut 2, synthèses Kokoro simultanées,
    CPU / VRAM), `--max-render-workers` (défaut 1, montage + rendu ffmpeg) ; la charge réseau
    (Webtoons / Gemini) est ainsi isolée de la charge locale ;
  - **`batch_status.json`** (racine du projet, `--status-file`) : par URL, `status`
    (`pending` / `processing` / `done` / `failed`), étape courante, dossier de sortie, épisode,
    modèle, **durée de chaque étape** (`timings`), durée totale, erreur, nombre de tentatives,
    plus un bloc `run` (durée du lot, parallélisme, médianes) et l'état du gestionnaire Gemini.
    `python -m src.main stats` relit ce fichier et affiche les temps sans rien relancer. Une
    exécution suivante **saute les chapitres `done`**, reprend les `failed` (sauf
    `--no-retry-failed`) et les `processing` interrompus ; les étapes déjà calculées sur disque
    sont réutilisées. Après un `QuotaExhaustedError`, les chapitres restants sont marqués
    `failed` sans nouvel appel Gemini (à relancer quand le quota est rétabli).
- Dimensionnement (offre gratuite) : ≈ 15 appels par chapitre et ≈ 20 appels/jour/modèle/clé ;
  20 chapitres/jour demandent donc plusieurs clés (× 6 modèles de la cascade) ou la facturation.

## 4. Garde-fous

1. Headers HTTP `User-Agent` + `Referer` systématiques (anti-403 Webtoons).
2. Batching Gemini (10–15 images) + retries.
3. Remplacement phonétique avant toute synthèse vocale.
4. Cases de plus de 1200 px coupées horizontalement en 2 ou 3 blocs de pleine largeur ; aucune
   case n'est rognée en largeur, agrandie au-delà de 105 % ni défilée.
5. Aucune écriture disque intermédiaire obligatoire : stitch et slicing en mémoire.

## 5. Utilisation (phase 1)

Environnement : Python 3.12 dans `.venv` (`.\.venv\Scripts\python.exe`), dépendances de
`requirements.txt` sauf Kokoro/torch (installés à la phase 3). Toutes les commandes se lancent
depuis la racine du projet.

```powershell
# Tests unitaires (aucun accès réseau)
.\.venv\Scripts\python.exe -m pytest tests -q

# Module 1 : télécharger + assembler un chapitre en une bande PNG
.\.venv\Scripts\python.exe -m src.modules.scraper "<url du viewer Webtoons>" --out output/strip.png

# Module 2 : valider le découpage (cases PNG + panels.json + debug_overlay.png)
.\.venv\Scripts\python.exe tests/test_slicer_local.py --synthetic
.\.venv\Scripts\python.exe tests/test_slicer_local.py --image output/strip.png
.\.venv\Scripts\python.exe tests/test_slicer_local.py --url "<url du viewer Webtoons>" --out output/slicer_test
```

Options du script de découpe : `--threshold`, `--min-gap`, `--padding`, `--giant`, `--min-height`.
Le script affiche un tableau ASCII des cases et termine par `PANELS=<n> SCROLL_VERTICAL=<m>`
(code de sortie 1 si aucune case).

### Phase 2 : Module 3 (Analyzer Gemini)

```powershell
# Clé API : fichier .gemini_key à la racine (ignoré par git), ou $env:GEMINI_API_KEY

# Analyser un dossier de cases produit par le slicer -> scenes.json
.\.venv\Scripts\python.exe tests/test_analyzer_local.py --panels-dir output/slicer_test

# Pipeline complet URL -> cases -> scènes (cases écrites dans output/analyzer_test)
.\.venv\Scripts\python.exe tests/test_analyzer_local.py --url "<url du viewer Webtoons>"

# Validation hors ligne sans clé (client factice, une scène par case)
.\.venv\Scripts\python.exe tests/test_analyzer_local.py --panels-dir output/slicer_test --dry-run
```

Options : `--model`, `--batch-size` (1-15), `--language` (défaut `en`), `--thinking-budget N`,
`--max-panels N` pour limiter le coût d'un essai, `--out` pour le fichier de sortie. Le script
termine par `SCENES=<n> PANELS=<m> BATCHES=<b> TOKENS=<in>+<out>+<réflexion>` (code 2 si la clé
API est absente ou les arguments invalides).

### Phase 3 : Module 4 (Kokoro TTS)

```powershell
# Dépendances (une fois) : kokoro + torch CPU + soundfile
.\.venv\Scripts\python.exe -m pip install kokoro soundfile

# Synthèse de toutes les scènes narratives d'un chapitre -> audio/ (WAV + voiceover.json + voiceover_full.wav)
.\.venv\Scripts\python.exe tests/test_tts_local.py --scenes output/max_level_newbie_ep1/scenes.json

# Validation hors ligne sans modèle (un bip par mot)
.\.venv\Scripts\python.exe tests/test_tts_local.py --scenes output/max_level_newbie_ep1/scenes.json --dry-run
```

Options : `--voice` (ex. `am_michael`), `--speed`, `--padding` (défaut 0,18 s), `--pronunciations`,
`--include-filler`, `--max-scenes N`. Le rapport HTML est régénéré avec un lecteur audio par scène.
Le script termine par `SEGMENTS=<n> TOTAL=<secondes>s`.

### Phase 4 : pipeline complet (CLI)

```powershell
# Toutes les etapes : cases, scenes, voix off, timeline, brouillon CapCut, apercu 60 s
.\.venv\Scripts\python.exe -m src.main run "<url du viewer Webtoons>"
# Options utiles : --out DIR, --voice af_heart, --speed 1.1, --preview-seconds 0 (complet),
#                  --no-preview, --no-capcut, --capcut-dir "<dossier des projets CapCut>", --force
# Son : --bgm-dir config/bgm (calm.mp3 / tense.mp3 / action.mp3 / default.mp3 ; vide = boucles de substitution),
#       --bgm musique.mp3 (unique, bouclee), --no-bgm, --sfx-dir config/sfx (swoosh / impact / roar), --no-sfx

# Re-rendre seulement l'apercu ou le brouillon a partir d'un chapitre deja traite
.\.venv\Scripts\python.exe -m src.main preview output/<chapitre> --seconds 60
.\.venv\Scripts\python.exe -m src.main capcut output/<chapitre>
```

### Phase 8 : production (lots de chapitres, plusieurs cles)

```powershell
# .env a la racine : GEMINI_API_KEYS=cle1,cle2,cle3   (voir .env.example)
# Lister les episodes cibles sans rien traiter
.\.venv\Scripts\python.exe -m src.main batch "https://www.webtoons.com/en/action/<serie>/list?title_no=N" --start-chapter 1 --end-chapter 20 --dry-run
# Traiter la plage (5 chapitres en parallele, 10 requetes Gemini/min, 2 syntheses Kokoro)
.\.venv\Scripts\python.exe -m src.main batch "<url serie ou episode>" --start-chapter 1 --end-chapter 20 --max-chapters 5 --max-gemini-rpm 10 --max-tts-workers 2
# Ou depuis une liste d'URL ; relancer la meme commande reprend la ou le lot s'est arrete (batch_status.json)
.\.venv\Scripts\python.exe -m src.main batch --url-list chapitres.txt --no-preview
# Temps mesures du dernier lot (par etape, par chapitre, par seconde de video)
.\.venv\Scripts\python.exe -m src.main stats
# Refaire la voix off apres un changement de reglage, sans redepenser de quota Gemini
.\.venv\Scripts\python.exe -m src.main batch "<url>" --start-chapter 1 --end-chapter 4 --redo tts --sentence-gap 0.15
# Fusionner les chapitres deja traites en UN projet CapCut (compilation)
.\.venv\Scripts\python.exe -m src.main merge --pattern "output/ma-serie_ep*" --preview-seconds 45
# Comparer les 28 voix Kokoro (extraits WAV + rapport HTML avec lecteurs audio)
.\.venv\Scripts\python.exe -m src.main voices --workers 2
```

Sur l'offre gratuite, `--max-gemini-rpm 5` evite les 429 par minute (la limite est de 5
requetes/min et par modele) : au-dela, le gestionnaire passe son temps en backoff.

Les étapes déjà calculées (cases, scènes, voix off) sont réutilisées si leurs fichiers existent
dans le dossier de sortie ; `--force` recalcule tout. Attention : toute modification de la règle
de découpe change la numérotation des cases, un chapitre traité avant doit être refait dans un
nouveau dossier (ou avec `--force`), sinon `scenes.json` référence d'anciens numéros.

## 5 bis. Performances mesurées (2026-09-11)

Série *The Lazy Lord Masters the Sword*, 4 chapitres menés de l'URL au projet CapCut, 2 en
parallèle, `--max-gemini-rpm 5`, aperçu de 30 s, **une seule clé gratuite**, modèle
`gemini-3.5-flash-lite` (les Flash avaient épuisé leur quota journalier).

| Étape | Médiane par chapitre | Remarque |
|---|---|---|
| Scraping + découpe | **39 s** (31 à 46 s) | 112 à 176 cases, 4 chapitres en parallèle |
| Analyse Gemini | **416 s** | dont ~375 s d'attente Gemini (appels + limite 5 req/min) |
| Voix off Kokoro | **155 s** | 110 à 194 s selon la longueur du script |
| Timeline + brouillon CapCut | **4 s** | |
| Aperçu ffmpeg 30 s | **18 s** | ≈ 0,6 s de calcul par seconde d'aperçu |
| **Total par chapitre** | **≈ 625 s (10 min)** | pour **8 min de vidéo montée** |

- 4 chapitres en 17,5 min de temps réel (263 s par chapitre grâce au parallélisme), pour
  31,9 min de vidéo cumulée.
- Répartition du temps : **63 % d'attente Gemini** (25,7 min) contre 37 % de calcul local
  (14,9 min). Sur les 25,7 min d'attente, seules **8 min** sont du temps de réponse réel
  (médiane 4,3 s par appel, p95 18,7 s) : le reste est la limite de **5 requêtes/minute** de
  l'offre gratuite.
- **Avec une clé payante** (limite RPM levée), l'analyse tomberait à ≈ 165 s par chapitre, soit
  **≈ 380 s (6,5 min) par chapitre**, le goulot devenant la voix off Kokoro (CPU).
- **Coût en quota** : ≈ 18 à 24 appels par chapitre (1 lot de 12 cases = 1 appel, + 1 script,
  + 1 appel par groupe de paragraphes). L'offre gratuite donne ≈ 20 à 25 appels par jour, par
  modèle et par clé : avec la cascade de 9 modèles, **≈ 7 à 8 chapitres par jour et par clé**.
  Pour 10 chapitres et plus par jour : plusieurs clés dans `GEMINI_API_KEYS`, ou la facturation.

## 5 ter. Où se gagne le temps (mesures 2026-09-12)

Analyse du profil mesuré, du plus rentable au moins rentable :

| Levier | Gain | État |
|---|---|---|
| Ajouter des clés `GEMINI_API_KEYS` / facturer | supprime les 2/3 d'attente quota | à faire côté compte |
| `--gemini-batch-delay 0` (le limiteur RPM suffit) | ≈ 45 s par chapitre | option livrée |
| `--batch-size 15` au lieu de 12 | ≈ 20 % d'appels Gemini en moins | option livrée |
| Cases clés en parallèle (`--keyframe-workers 4`) | 2 à 7 appels groupés au lieu de sérialisés | **livré, activé par défaut** |
| `--no-preview` si seul le projet CapCut compte | 18 s par chapitre | déjà disponible |
| Scraping découplé de l'analyse | ≈ 39 s par chapitre masqués | **livré** |

Ce qui **ne** se parallélise **pas** :

- **Les lots de beats** (12 à 15 appels par chapitre, l'essentiel du coût) : chaque lot reçoit
  en contexte les 3 derniers beats du lot précédent, pour la continuité des noms et du récit.
  Les paralléliser demanderait de sacrifier ce contexte.
- **Les chapitres entre eux, sur l'offre gratuite** : le limiteur RPM global est partagé, donc
  augmenter `--max-chapters` ne change rien tant que la limite de 5 requêtes/minute s'applique.
  Sur une clé payante, c'est le levier principal.
- **La synthèse vocale au-delà de 2 workers** : mesuré sur 16 cœurs (torch utilise déjà
  8 threads par flux) — 1 worker = 3,2x temps réel, **2 workers = 4,7x (optimum)**,
  4 workers = 4,4x. Au-delà de 2, la sur-souscription fait régresser.

### Comparatif des voix Kokoro

`python -m src.main voices` synthétise le même extrait avec les 28 voix anglaises et écrit
`output/voices/voices.html` (un lecteur audio par voix). Débits mesurés : de **175 mots/min**
(`af_jessica`) à **108 mots/min** (`af_nicole`, très posée). Le choix de la voix change la durée
de la vidéo de près de 40 %. **Voix par défaut depuis le 2026-09-12 : `am_puck`** (américaine
masculine, 161 mots/min), retenue après écoute du comparatif ; `am_fenrir` était la précédente.

### Compilation multi-chapitres

`python -m src.main merge --pattern "output/ma-serie_ep*"` fusionne des chapitres déjà traités
en **un seul projet CapCut** : les timelines sont mises bout à bout (`concat_timelines`) avec
0,6 s de silence entre chapitres, les médias restent dans le dossier de leur chapitre
(chemins absolus, aucune copie), les numéros de scène sont décalés pour rester uniques.

## 6. Roadmap

| Phase | Contenu | Statut |
|---|---|---|
| 1 | Structure, PRD, requirements, **Scraper/Stitcher**, **Smart Slicer**, test local | livrée (2026-09-08) |
| 2 | Analyzer Gemini + schémas Pydantic des scènes | livrée et validée en direct (2026-09-08) : Tower of God ep. 1 (20 cases → 9 scènes) et *I'm the Max-Level Newbie* ep. 0 (117 morceaux, bande 720x109 586, 82 cases dont 24 `scroll_vertical` → 28 scènes dont 1 filler, 7 lots, ~41 000 tokens, 50 s). Sans budget de réflexion : 24 scènes, qualité équivalente, ~15 % de tokens en moins |
| 3 | Kokoro TTS + dictionnaire phonétique | livrée et validée (2026-09-08) : *I'm the Max-Level Newbie* ep. 0, 27 scènes → 357,7 s de voix off `af_heart` en 102 s de calcul CPU (x3,5 temps réel), torch 2.14 CPU, kokoro 0.9.4 |
| 4 | Timeline, CapCut Draft Builder, aperçu ffmpeg, CLI de bout en bout | livrée (2026-09-09) : *I'm the Max-Level Newbie* ep. 0 traité de l'URL à l'aperçu 60 s (82 cases, 31 scènes dont 5 filler, 344 s de voix `am_fenrir`, brouillon 4 pistes) ; ouverture dans CapCut à valider |
| 5 | Qualité : fusion/filtre des micro-cases, script global en deux étapes (beats → script sans formule visuelle → cases clés), rythme ≥ 2,5 s par case et zoom 106 % | livrée (2026-09-09) : même chapitre → 63 cases, 24 beats, 12 paragraphes / 656 mots sans formule visuelle, 36 cases clés, 247 s de voix, 4,2 à 11,8 s par case ; modèle par défaut `gemini-3.5-flash` |
| 6 | Montage : cases hautes coupées en deux (ratio > 1,5) et cadrage automatique sur le tiers supérieur, plus aucun défilement > 2 s, punch-in +15 % / 0,2 s sur les cases `action_heavy` taguées par Gemini, sous-titres en blocs de 2-3 mots (gras, contour noir), bruitages aux transitions des scènes d'action et musique par ambiance avec fondu enchaîné, point de reprise Gemini | livrée (2026-09-09) : même chapitre → 99 cases (36 coupées en deux), 34 beats, 16 paragraphes / 839 mots, 45 cases clés montées dont 37 moitiés et 10 punch-in, 0 défilement, 294 blocs de sous-titres, 14 bruitages, 366,8 s de voix ; analyse reprise à travers 3 modèles (3.6 → 3.7 → 3.8-flash) grâce au point de reprise, quotas gratuits obligent. À faire : déposer de vraies musiques `config/bgm/` et de vrais bruitages `config/sfx/`, valider le brouillon dans CapCut |
| 7 | Anti-pixellisation et musique : cases à résolution native centrées (jamais agrandies), zoom ≤ 105 %, plus aucun recadrage ni défilement, cases > 1200 px coupées horizontalement en 2-3 blocs pleine largeur sur la ligne la plus calme, musique par ambiance toujours présente (boucles de substitution synthétisées si `config/bgm/` est vide), bouclée dans CapCut, −22 dB, sous-titres 2-4 mots avec contour de 2 px | livrée (2026-09-09, tard) : même chapitre → 106 cases (35 coupées, dont 8 en 3 blocs), 52 cases clés montées dont 15 punch-in, 10 cases réduites (≤ 10 %) car plus hautes que le cadre, 222 blocs de sous-titres, 20 bruitages, 11 segments de musique bouclés sur A2/A2b à −22 dB, aperçu contrôlé image par image. Quota Gemini épuisé sur tous les modèles ce jour : script et cases clés **repris de la v3** par recouvrement des plages de lignes, à régénérer avec Gemini (nouveau dossier) dès que le quota est rétabli |
| 11 | Mode « une requête » par défaut : script et cases clés en un seul appel Gemini, bascule automatique si les images dépassent 16 Mo, garde-fou de longueur avec régénération, `--multi-call` pour l'ancien mode | livrée (2026-09-12) : *The Lazy Lord* ep. 5 (131 cases) en **2 appels au lieu de 20**, 19 paragraphes / 1373 mots (cible 1441), 76 cases clés, 0 numéro invalide, 0 doublon, ordre respecté, 0 formule interdite ; analyse 114 s contre 416 s, chapitre complet en 297 s contre 586 s |
| 10 | Parallélisme et confort : cases clés Gemini en parallèle, options `--batch-size` / `--gemini-batch-delay` / `--keyframe-workers` / `--sentence-gap` / `--padding`, `--redo <etape>` (refaire la voix sans redépenser de quota), commande `merge` (compilation multi-chapitres en un projet CapCut), commande `voices` (comparatif des 28 voix Kokoro), pause entre phrases 0,4 → 0,2 s | livrée (2026-09-12) : 4 chapitres de *The Lazy Lord* refaits en 9,1 min sans aucun appel Gemini, fusionnés en un projet de **31,5 min** (249 cases, 84 scènes, 1263 sous-titres) ; parallélisme TTS mesuré (optimum 2 workers) |
| 9 | Mesures de performance : temps de réponse Gemini (médiane / p95 / max, par modèle), durée de chaque étape par chapitre dans `batch_status.json`, commande `stats`, sémaphore de scraping séparé de l'analyse, pause de 30 s au lieu d'une cascade quand une clé unique est limitée par minute, cascade étendue aux Flash-Lite | livrée (2026-09-11) : voir « Performances mesurées » ci-dessous |
| 8 | Production : padding TTS 0,18 s, gestionnaire Gemini multi-clés (`GEMINI_API_KEYS`, rotation sur clé invalide / quota journalier, backoff 2-4-8-16 s, 4 essais par clé, cascade de modèles, RPM global), délai forcé 2,5 s entre lots, orchestrateur `batch` (asyncio + sémaphores `--max-chapters` / `--max-gemini-rpm` / `--max-tts-workers`, découverte des épisodes, `batch_status.json` reprenable) | livrée (2026-09-10) : 189 tests ; validée sur la série test (4 épisodes découverts, plage 1-2 : ep. 1 réutilisé, ep. 2 scrapé puis analyse refusée par le quota journalier épuisé, marqué `failed` et repris à la prochaine exécution). La cascade demandée (2.0-flash → 1.5-flash → 1.5-pro) est remplacée par les modèles réellement servis, configurable par `GEMINI_MODEL_CASCADE` |
