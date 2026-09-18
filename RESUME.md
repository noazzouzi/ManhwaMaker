# RESUME.md — Auto-Manhwa Recap Generator

> Document de transmission destiné à un agent IA qui reprendrait le projet.
> Il décrit l'application telle qu'elle est **réellement implémentée** au 2026-09-12,
> pas telle qu'elle était spécifiée. Quand l'implémentation s'écarte du cahier des
> charges initial (`Consignes.md`), l'écart est signalé explicitement.
>
> Documents voisins : `Consignes.md` (cahier des charges d'origine, court, partiellement
> périmé), `PRD.md` (document produit détaillé et tenu à jour, ~40 Ko).

---

## 0. Carte d'identité

| | |
|---|---|
| **Nom** | Auto-Manhwa Recap Generator |
| **Nature** | CLI Python mono-utilisateur, exécution locale |
| **Entrée** | URL d'un chapitre ou d'une série sur `webtoons.com` |
| **Sortie** | Projet CapCut prêt à ouvrir (`draft_content.json`) + aperçu MP4 1920x1080 @ 60 fps |
| **But métier** | Produire des vidéos récapitulatives de manhwa pour YouTube, en série |
| **Racine** | `C:\Users\Nouamane\Documents\ManhwaMaker` |
| **Python** | 3.12.10, venv `.\.venv\Scripts\python.exe` |
| **Plateforme** | Windows 11, PowerShell 5.1 |
| **Versionné par git** | **Non** (pas de dépôt git dans le dossier de travail) |
| **Tests** | 242 tests `pytest`, tous passants, ~19 s |
| **Lignes de code** | ~7 800 en `src/`, ~5 100 en `tests/` |

---

## 1. Ce que fait l'application

Une commande, un chapitre :

```powershell
.\.venv\Scripts\python.exe -m src.main run "https://www.webtoons.com/en/action/<serie>/<ep>/viewer?title_no=7358&episode_no=1"
```

Enchaînement réalisé :

1. **Scraping** — télécharge les morceaux d'image du chapitre et les recompose en une
   bande verticale continue.
2. **Découpe** — détecte les gouttières et découpe la bande en *cases* (panels) PNG.
3. **Analyse** — envoie **toutes** les cases à Gemini en **une seule requête** ; reçoit
   le script narré complet, découpé en paragraphes, chacun accompagné des numéros de
   cases à montrer.
4. **Voix off** — synthétise chaque paragraphe avec Kokoro TTS en local (24 kHz).
5. **Timeline** — construit le plan de montage : la durée d'affichage de chaque case est
   calée **exactement** sur la durée du segment audio correspondant.
6. **Projet CapCut** — écrit un brouillon CapCut (6 pistes) et le copie dans le dossier
   projets de CapCut.
7. **Aperçu** — rend un MP4 via ffmpeg pour contrôler le résultat sans ouvrir CapCut.

Et en série :

```powershell
.\.venv\Scripts\python.exe -m src.main batch "<url de la serie>" --start-chapter 1 --end-chapter 10
.\.venv\Scripts\python.exe -m src.main merge --pattern "output/<serie>_ep*"   # compilation en 1 projet
```

---

## 2. Environnement d'exécution et conventions de code

Ces conventions ne sont pas cosmétiques : les enfreindre casse l'exécution sur cette
machine.

### 2.1 Shell

- **PowerShell 5.1**, pas de `&&` ni de `||` (erreur de parseur). Chaîner avec
  `A; if ($?) { B }`.
- Pas d'opérateurs `?:`, `??`, `?.`.
- Les *here-strings* avec guillemets imbriqués cassent : pour tout script Python
  jetable, **écrire un fichier** dans le dossier scratchpad plutôt que `python -c "..."`.

### 2.2 Encodage

- La console Windows est en **cp1252**. Tout `print` destiné à l'utilisateur passe par
  `_ascii()` (`src/main.py:46`) ou `.encode("ascii", "replace")`.
- Les sorties de log ne doivent contenir **aucun caractère non ASCII**.
- Les fichiers source sont en UTF-8 ; les **docstrings et commentaires sont en
  français**, les **identifiants en anglais**. Conserver ce mélange.

### 2.3 Variables d'environnement utiles

| Variable | Rôle |
|---|---|
| `GEMINI_API_KEYS` | Plusieurs clés séparées par des virgules → rotation automatique |
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | Clé unique (repli) |
| `GEMINI_MODEL` | Modèle préféré (la cascade suit) |
| `GEMINI_MODEL_CASCADE` | Remplace toute la cascade de secours |
| `HF_HUB_DISABLE_SYMLINKS=1` | **Obligatoire sur cette machine** : sans ça, le cache Hugging Face de Kokoro échoue en `WinError 1314` (privilège de lien symbolique) |
| `THUMBNAIL_IMAGE_BACKEND` | `gemini` (défaut), `stability` ou `local` |
| `STABILITY_API_KEY` / `LOCAL_SD_URL` | seulement pour les backends correspondants |

### 2.4 Secrets

- La clé Gemini vit dans `.gemini_key` (racine, gitignoré) ou dans `.env`
  (`GEMINI_API_KEYS=...`). `.env.example` documente le format.
- **Une clé ne doit jamais être imprimée.** Toute journalisation passe par
  `mask_key()` (`src/utils/gemini_manager.py:102`) qui ne garde que les 4 derniers
  caractères : `...oywg`.

### 2.5 Langue par défaut

L'**anglais** est la langue par défaut partout : URL Webtoons (`/en/`), narration
générée, voix off. Voir `DEFAULT_SITE_LANGUAGE` et `DEFAULT_NARRATION_LANGUAGE`
(`src/utils/config.py`).

---

## 3. Arborescence

```
ManhwaMaker/
├── src/
│   ├── main.py                    CLI Typer : 7 commandes           (321 l.)
│   ├── pipeline.py                étapes + orchestration 1 chapitre (502 l.)
│   ├── models/                    contrats Pydantic
│   │   ├── chapter.py             ChapterMeta                        (50 l.)
│   │   ├── panel.py               Panel (+ pixels NumPy)             (89 l.)
│   │   ├── scene.py               Scene, Beat, Recap*, ChapterAnalysis (187 l.)
│   │   ├── audio.py               SceneAudio, VoiceoverManifest      (63 l.)
│   │   ├── thumbnail.py           ThumbnailDraft, ThumbnailBrief     (79 l.)
│   │   └── timeline.py            PanelClip, AudioClip, SfxClip, BgmClip, SubtitleCue,
│   │                              CueAnimation, ClipTransition, VfxClip, Timeline (260 l.)
│   ├── modules/
│   │   ├── scraper.py             Module 1 — scraping + assemblage  (597 l.)
│   │   ├── slicer.py              Module 2 — découpe en cases       (667 l.)
│   │   ├── analyzer.py            Module 3 — Gemini VLM            (1763 l.)
│   │   ├── tts_engine.py          Module 4 — Kokoro TTS             (473 l.)
│   │   ├── timeline_builder.py    Module 5a — plan de montage       (495 l.)
│   │   ├── capcut_builder.py      Module 5b — brouillon CapCut      (307 l.)
│   │   ├── preview_renderer.py    Module 5c — rendu ffmpeg          (408 l.)
│   │   ├── batch_processor.py     Module 6 — orchestrateur de lots  (513 l.)
│   │   ├── thumbnail/             Module 7 — miniatures YouTube
│   │   │   ├── analyzer.py        etape 1 : scene, accroche, fleche (200 l.)
│   │   │   ├── image_backends.py  etape 2 : Gemini / Stability / local (270 l.)
│   │   │   ├── compositor.py      etape 3 : compositing Pillow      (330 l.)
│   │   │   └── pipeline.py        enchainement + frontiere d'erreur (180 l.)
│   │   └── voice_lab.py           banc d'essai des voix Kokoro      (207 l.)
│   └── utils/
│       ├── config.py              racine, langues, clés API         (114 l.)
│       ├── gemini_manager.py      rotation clés / cascade / RPM     (435 l.)
│       ├── http.py                session requests + retries        (185 l.)
│       ├── image_utils.py         conversions PIL/NumPy             (130 l.)
│       ├── audio_assets.py        SFX et BGM (+ synthèse de repli)  (237 l.)
│       └── report.py              rapport HTML de contrôle          (173 l.)
├── tests/                         19 fichiers, 209 tests
│   ├── conftest.py                neutralise les sleeps, isole .env
│   ├── synthetic_strip.py         générateur de bandes de test
│   └── test_*_local.py            tests nécessitant le réseau/modèles (marqués)
├── config/
│   ├── pronunciations.json        dictionnaire phonétique
│   ├── sfx/{swoosh,impact,roar}.wav     ⚠ placeholders synthétisés
│   ├── bgm/{calm,tense,action}.wav      ⚠ placeholders synthétisés
│   ├── fonts/{Bangers-Regular,Montserrat-Bold,Rubik-Bold}.ttf   Google Fonts, SIL OFL
│   └── thumbnail/arrow_yellow.png 600x400 RGBA, dessin géométrique généré
├── output/<serie>_ep<N>/          un dossier par chapitre traité
├── PRD.md                         document produit détaillé
├── Consignes.md                   cahier des charges d'origine
├── RESUME.md                      ce fichier
├── requirements.txt
├── .env.example
├── .gemini_key                    ⚠ secret, gitignoré
└── batch_status.json              suivi du dernier lot (gitignoré)
```

---

## 4. Flux de données

### 4.1 Les quatre étapes

Le pipeline est découpé en étapes **réutilisables et indépendamment relançables**
(`src/pipeline.py`). C'est ce découpage qui permet à l'orchestrateur de lots de les
placer sous des sémaphores différents.

| Étape | Fonction | Nature | Artefacts écrits |
|---|---|---|---|
| 1-2 | `stage_scrape_slice` | I/O réseau | `chapter.json`, `panels.json`, `panel_NNN.png`, `debug_overlay.png` |
| 3 | `stage_analyze` | réseau + quota | `scenes.json` (+ `analysis_checkpoint.json` en mode deux étapes) |
| 4 | `stage_tts` | CPU | `audio/scene_NNN.wav`, `audio/voiceover.json`, `audio/voiceover_full.wav/.mp3`, `scenes_report.html` |
| 5-7 | `stage_montage` | CPU | `timeline.json`, `capcut/<nom>/draft_content.json`, `preview_<N>s.mp4` |

**Règle de réutilisation :** chaque étape relit son artefact s'il existe au lieu de
recalculer. `--redo <étape>` force le recalcul à partir de cette étape ; `--force`
recalcule tout, scraping compris. Conséquence pratique importante : `--redo tts`
**ne consomme aucun quota Gemini**.

Rangs des étapes (`STAGE_RANKS`) : `analyze` = 1, `tts` = 2, `montage` = 3.

### 4.2 Dossier de sortie d'un chapitre

```
output/<serie>_ep<N>/
├── chapter.json           ChapterMeta (titres, URL, liste des morceaux)
├── panels.json            métadonnées des cases (sans pixels)
├── panel_000.png …        une image par case, résolution native
├── debug_overlay.png      bande entière avec les découpes tracées (~20 Mo)
├── scenes.json            ChapterAnalysis : script + cases clés + tokens
├── scenes_report.html     contrôle visuel : vignettes + texte + lecteurs audio
├── audio/
│   ├── scene_000.wav …    un WAV par paragraphe
│   ├── voiceover.json     VoiceoverManifest (durées exactes)
│   └── voiceover_full.wav/.mp3
├── backgrounds/           fonds floutés pré-rendus 1920x1080 (JPEG)
├── timeline.json          Timeline : le plan de montage complet
├── capcut/<nom du projet>/draft_content.json + draft_meta_info.json
└── preview_60s.mp4
```

### 4.3 Transformation logique

```
URL
 └─ ChapterMeta + bande verticale (PIL/NumPy)
     └─ list[Panel]                      (découpe par variance)
         └─ ChapterAnalysis              (1 appel Gemini)
             ├─ scenes[].narration       → texte narré
             ├─ scenes[].panel_ids       → cases clés à montrer
             └─ scenes[].action_heavy_ids→ cases à punch-in
                 └─ VoiceoverManifest    (durées réelles de la voix)
                     └─ Timeline         (durée image = durée audio)
                         ├─ draft_content.json  (CapCut)
                         └─ preview.mp4         (ffmpeg)
```

---

## 5. Contrats de données (Pydantic)

### 5.1 `ChapterMeta` — `src/models/chapter.py`

`url`, `final_url`, `series_title`, `episode_title`, `title_no`, `episode_no`,
`image_urls`, `chunk_sizes`. Méthode `summary()` renvoie une ligne **ASCII pure** pour
les logs (les titres Webtoons contiennent des guillemets typographiques et parfois des
alphabets non latins).

### 5.2 `Panel` — `src/models/panel.py`

```python
index: int            # ordre de lecture, 0 = haut de la bande
y_start, y_end: int   # bornes dans la bande d'origine (padding inclus)
height, width: int
type: "static" | "scroll_vertical"
part: "top" | "middle" | "bottom" | None   # bloc d'une case très haute découpée
source_index: int | None                   # segment d'origine partagé par les blocs
image: np.ndarray     # (H, W, 3) uint8 — EXCLU de model_dump
```

Particularités à connaître : `image` est exclu des exports (`Field(exclude=True)`),
`__eq__` compare métadonnées **et** pixels via `np.array_equal`, et le modèle n'est
**pas hachable** (`__hash__ = None`).

### 5.3 `ChapterAnalysis` — `src/models/scene.py`

C'est le cœur du contrat entre l'IA et le montage.

```python
series_title, episode_title, source_url: str
model: str                 # modèle Gemini effectivement utilisé
language: str
n_panels: int              # cases soumises
n_batches: int             # nombre d'appels API (1 en mode une requête)
scenes: list[Scene]
beats: list[Beat]          # vide en mode une requête
prompt_tokens, output_tokens, thinking_tokens: int
```

Propriétés dérivées : `n_scenes`, `n_key_panels`, `script_words`, `n_filler`,
`story_scenes()` (hors remplissage), `covered_panel_ids()`.

`Scene` :

```python
index: int                 # ordre de lecture
panel_ids: list[int]       # SEULEMENT les cases clés retenues (1 à 4)
narration: str             # le paragraphe narré
emotion: Emotion
is_filler: bool
beat_ids: list[int]        # vide en mode une requête
action_heavy_ids: list[int]# sous-ensemble de panel_ids → punch-in
```

⚠ **Piège** : `panel_ids` ne contient **pas** toutes les cases de la scène, seulement
celles qui seront affichées. Les cases de transition ne sont montées nulle part. C'est
un changement par rapport à la première version du projet.

`Emotion` est un `Literal` de 11 valeurs : `neutral`, `calm`, `tension`, `action`,
`sad`, `happy`, `humor`, `romance`, `fear`, `mystery`, `epic`. Un alias renvoyé par le
modèle (`joy`, `battle`, `suspense`…) est ramené à la valeur canonique par
`coerce_emotion()` via `_EMOTION_ALIASES`.

**Schémas de réponse Gemini** (volontairement minimaux : pas de valeur par défaut ni de
contrainte, pour rester compatibles avec la conversion JSON-Schema du SDK) :

| Schéma | Usage |
|---|---|
| `RecapDraft` / `RecapParagraph` | **mode une requête** (défaut) : `text`, `emotion`, `key_panel_ids`, `action_heavy_ids` |
| `BeatBatch` / `BeatDraft` | mode deux étapes, étape 1a |
| `ScriptDraft` / `ParagraphDraft` | mode deux étapes, étape 1b |
| `KeyframeBatch` / `KeyframeChoice` | mode deux étapes, étape 2 |
| `SceneBatch` / `SceneDraft` | legacy, première génération |

### 5.4 `VoiceoverManifest` — `src/models/audio.py`

`language`, `lang_code`, `voice`, `speed`, `padding_s`, `sentence_gap_s`,
`sample_rate`, `model`, `items: list[SceneAudio]`, `total_duration_s`, `full_file`.

`SceneAudio` porte `duration_s` (parole + silence de fin, calculée sur le **nombre exact
d'échantillons**) et `speech_s` (parole seule). `duration_s` pilote la timeline ;
`speech_s` sert à répartir les sous-titres.

### 5.5 `Timeline` — `src/models/timeline.py`

```python
width=1920, height=1080, fps=60
series_title, episode_title: str
panels_dir, audio_dir: str         # chemins absolus
clips: list[PanelClip]             # V2 (et V1 pour le fond)
audio: list[AudioClip]             # A1
subtitles: list[SubtitleCue]       # T1
sfx: list[SfxClip]                 # A3
bgm: list[BgmClip]                 # A2
total_duration_s: float
bgm_file: str | None               # compatibilité (converti en bgm[])
bgm_gain_db: float = -22.0
```

`Motion` = `"ken_burns" | "punch_in" | "scroll_vertical"`. Le troisième n'est plus
produit : il reste accepté pour relire d'anciennes timelines et se rend comme
`ken_burns`.

---

## 6. Modules

### 6.1 Module 1 — Scraper (`src/modules/scraper.py`)

**Rôle** : URL → bande verticale continue + `ChapterMeta`.

Chaîne : `fetch_chapter_image_urls` → `download_images` → `stitch_webtoon_pages`,
enrobées par `scrape_chapter`.

Faits d'implémentation importants :

- Les images sont dans `div#_imageList img._images`, attribut **`data-url`** (pas `src` :
  le `src` est un pixel transparent de remplissage, repéré par
  `PLACEHOLDER_SRC_MARKER = "bg_transparency"`).
- **Sans `Referer: https://www.webtoons.com/` et sans `User-Agent`, Webtoons répond 403.**
  Garanti par `ensure_mandatory_headers()` (`src/utils/http.py`).
- Les morceaux téléchargés **n'ont pas tous la même largeur** : `stitch_webtoon_pages`
  normalise avant le `np.vstack`.
- Téléchargement séquentiel, en mémoire, avec délai de politesse et retries.
- `discover_episodes` / `parse_episode_links` / `episode_url` / `series_list_url`
  permettent de résoudre une plage d'épisodes depuis l'URL de la liste d'une série.
- `normalize_webtoon_url` force le segment de langue (`/en/` par défaut).
- CLI autonome : `python -m src.modules.scraper <url> --out output/strip.png`.

### 6.2 Module 2 — Smart Slicer (`src/modules/slicer.py`)

**Rôle** : bande → `list[Panel]`.

Algorithme :

1. Conversion en niveaux de gris, puis **variance par ligne** (`compute_row_variance`,
   traitée par blocs de 4096 lignes pour borner la mémoire).
2. `gutter_mask = variance < 15.0` ; les suites d'au moins **20 lignes** consécutives
   forment une gouttière (`find_gutters`, vectorisé avec `np.diff`).
3. Les segments de contenu sont le complément des gouttières.
4. Chaque segment est élargi de **15 px** de chaque côté, les segments de moins de
   **180 px** sont écartés (micro-cases, onomatopées).
5. Deux cases voisines faisant chacune moins de **250 px**, séparées de moins de
   **150 px**, sont fusionnées (`merge_small_segments`) : traite les suites de bulles de
   chat.
6. Une case de plus de **1200 px** est **sous-découpée horizontalement** en 2 ou 3 blocs
   (`split_tall_range`) qui gardent **100 % de la largeur d'origine** ; la coupe est
   placée sur la ligne de plus faible variance dans une fenêtre de ±15 % autour de la
   coupe idéale, pour ne pas trancher un visage ou une bulle.
7. Une case restant plus haute que **1500 px** reçoit le type `scroll_vertical`.

| Constante | Valeur |
|---|---|
| `DEFAULT_VARIANCE_THRESHOLD` | 15.0 |
| `DEFAULT_MIN_GAP` | 20 px |
| `DEFAULT_MARGIN_PADDING` | 15 px |
| `MIN_PANEL_HEIGHT` | 180 px |
| `MERGE_SMALL_BELOW` / `MERGE_MAX_GAP` | 250 px / 150 px |
| `SPLIT_MAX_HEIGHT` / `SPLIT_MAX_PIECES` / `SPLIT_SEARCH_RATIO` | 1200 px / 3 / 0.15 |
| `GIANT_PANEL_HEIGHT` | 1500 px |

⚠ **Contrainte produit forte** : *aucun* rognage en largeur, jamais. Une version
antérieure recadrait vers le tiers supérieur des cases ; cela a été jugé inacceptable
(pixellisation) et retiré. Ne pas réintroduire de crop.

`render_debug_overlay` produit une planche de contrôle (découpes tracées et numérotées),
réduite et éventuellement répartie en colonnes si la bande dépasse 8000 px.

### 6.3 Module 3 — Gemini VLM Analyzer (`src/modules/analyzer.py`)

Le module le plus gros et le plus chargé en règles. Deux modes coexistent.

#### Mode « une requête » — **comportement par défaut**

`GeminiAnalyzer.analyze_panels_single_call(panels, meta)`.

Toutes les cases du chapitre partent dans **un seul appel** qui renvoie directement les
paragraphes du script, chacun avec ses cases clés et ses cases d'impact. Les images ne
sont envoyées qu'une fois.

Séquence interne :

1. `payload_bytes(panels)` — si les images dépassent `MAX_INLINE_PAYLOAD_BYTES`
   (**16 Mo**, marge sous la limite dure de 20 Mo de Gemini), bascule **automatique**
   sur le mode deux étapes (décidé dans `stage_analyze`, `src/pipeline.py:263`).
2. `single_call_targets(panels)` → `(mots, paragraphes)` : 11 mots par case borné à
   [250, 1500] ; paragraphes = cases / 7 borné à [4, 40].
3. `build_single_call_contents` : en-tête, puis pour chaque case une légende
   `Panel N` + l'image, puis un pied de requête qui **répète la cible de longueur**
   (c'est le message le plus proche de la génération, donc le plus suivi).
4. `parse_recap` → `_fix_recap` → validation Pydantic.
5. `normalize_recap` — la passe de sécurité (détaillée ci-dessous).
6. `enforce_script_conventions` — formules interdites, intro générique, CTA unique.
7. Garde-fou de longueur : si le script fait moins de `SCRIPT_MIN_LENGTH_RATIO`
   (**65 %**) de la cible, une régénération est tentée **une fois** avec un rappel
   numérique ; la reprise n'est gardée que si elle est plus longue.

`normalize_recap` (`src/modules/analyzer.py:677`) applique, dans l'ordre :

- rejet des numéros de case inconnus ou déjà pris par un paragraphe précédent ;
- bornage à `MAX_KEY_PANELS_PER_PARAGRAPH` = **4**, en gardant les cases les plus hautes ;
- repli sur la plus grande case encore libre si un paragraphe n'a aucune case valide ;
- suppression des paragraphes au texte vide ;
- **rétablissement de l'ordre de lecture** par `_enforce_reading_order` (voir 6.3.1).

#### 6.3.1 `_enforce_reading_order` — remise en ordre à coût minimal

Le modèle place parfois une case en avance : « case 8 » au paragraphe 2 alors que le
paragraphe 3 raconte les cases 4 à 7. À l'écran, cela produit un faux retour en arrière.

Un simple plancher croissant serait trop brutal : il rejetterait les cases 4 à 7 (quatre
cases) au lieu de la seule case 8 fautive. L'implémentation calcule donc la **plus longue
sous-suite strictement croissante** (`_longest_increasing`, algorithme des piles en
O(n log n)) sur la liste aplatie des cases, et ne retire que le complément — le minimum
de cases nécessaire.

Un paragraphe vidé par cette passe est recasé dans la fenêtre libre entre ses voisins
(la case la plus haute disponible). Si cette fenêtre est vide, on garde la case retirée
et donc le recul ponctuel : mieux vaut un petit saut qu'un paragraphe narré sans image à
l'écran.

Mesure sur 10 chapitres réels : 2 chapitres sur 10 présentaient un recul, d'amplitude 2
et 4 cases, sur 666 cases clés. Après correction, 0.

#### Mode « deux étapes » — historique, activé par `--multi-call`

Conservé parce qu'il est le repli automatique des chapitres trop lourds et parce qu'il
seul produit la liste des *beats* dans `scenes.json` (traçabilité).

- **1a. Beats** — les cases partent par lots de `DEFAULT_BATCH_SIZE` = 12 (maximum 15) ;
  pour chaque lot, Gemini extrait des beats factuels (cases couvertes, résumé,
  personnages, répliques, remplissage), avec un contexte glissant des
  `CONTEXT_BEATS` = 3 derniers beats du lot précédent. `normalize_partition` garantit que
  **chaque case du lot appartient à exactement un beat**. Étape intrinsèquement
  séquentielle (le contexte dépend du lot précédent).
- **1b. Script** — à partir de tous les beats en **texte seul** (donc sans limite de
  lot), Gemini rédige le script complet.
- **2. Cases clés** — pour chaque paragraphe, Gemini choisit parmi les cases de ses beats
  (au plus `MAX_CANDIDATES_PER_PARAGRAPH` = 8 candidates, en vignettes 640 px) 1 à 4
  cases clés. Ces groupes étant **indépendants**, ils sont appelés en parallèle
  (`DEFAULT_KEYFRAME_WORKERS` = 4 via `ThreadPoolExecutor`) ; la normalisation qui les
  départage reste séquentielle, donc le résultat est identique au séquentiel.

`AnalysisCheckpoint` (`src/modules/analyzer.py:242`) persiste les lots de beats, le
script et les groupes de cases clés dans `analysis_checkpoint.json`, pour qu'une analyse
coupée par le quota reparte des appels déjà réussis. Les groupes de cases clés sont
stockés dans un **dictionnaire indexé**, pas une liste : une liste ne pouvait sauvegarder
qu'un préfixe contigu et perdait les succès parallèles. L'ancien format liste est relu
par compatibilité.

#### Règles éditoriales imposées au modèle

Encodées dans les prompts **et** vérifiées en post-traitement :

1. **Aucune formule visuelle.** 13 motifs interdits (`FORBIDDEN_PATTERNS`), anglais et
   français : « in this panel », « we see », « the image shows », « is shown », « the
   camera », « close-up », « zoom in », « here, », « sur cette case », « on voit »,
   « ici, », « l'image montre »… Détection par `find_forbidden`, régénération **une
   fois** avec rappel, puis nettoyage local en dernier recours (`scrub_forbidden`).
2. **Aucune intro générique.** `GENERIC_INTRO_PATTERNS` bloque « In this chapter »,
   « Welcome back », « Today we », « Let's dive in », « This recap covers »… La première
   phrase doit être une **accroche tirée de l'histoire** (urgence, menace, mystère,
   fait choquant).
3. **Exactement un appel à l'abonnement**, formulé naturellement en une phrase, placé
   dans un paragraphe du milieu — jamais dans les deux premiers, jamais en phrase de
   clôture. `CTA_POSITION = 0.4`, détection par `CTA_PATTERN`.
4. Présent, 3ᵉ personne, storytelling purement rétrospectif, noms des personnages lus
   dans les dialogues, pas de puces ni de titres ni d'emoji.
5. Les cases de remplissage (carte-titre, crédits, publicité, note d'auteur) ne sont ni
   narrées ni choisies comme cases clés.

#### Transport et encodage des images

- Les images sont réduites à `DEFAULT_MAX_IMAGE_WIDTH` = 1024 px de large (ratio
  conservé), encodées en JPEG qualité 88.
- Une case plus haute que `DEFAULT_MAX_SLICE_HEIGHT` = 2048 px est découpée en tranches
  consécutives (au plus `MAX_SLICES_PER_PANEL` = 4) envoyées **sous le même numéro**.
- Vignettes de l'étape 2 : 640 px de large, tranches de 1280 px, 2 au maximum — divise
  les tokens image par environ trois.
- `panel_caption` signale les blocs d'une case découpée : « top part of a tall drawing,
  continued in panel N ».
- `temperature` = 0.4, `timeout` = 120 s.

#### Appels

Par défaut le transport passe par `GeminiManager` (§6.8). Un client injecté
(`GeminiAnalyzer(client=...)`, utilisé par les tests) garde une boucle de réessai locale.
`self.api_seconds` accumule le temps passé chez Gemini sous un `threading.Lock`.
`BATCH_DELAY_S` = 2.5 s sépare deux envois d'un même chapitre pour étaler les jetons par
minute ; cette pause est redondante avec le limiteur RPM et peut être mise à 0
(`--gemini-batch-delay 0`) quand `--max-gemini-rpm` est réglé sur la limite réelle.

### 6.4 Module 4 — Kokoro TTS (`src/modules/tts_engine.py`)

**Rôle** : `ChapterAnalysis` → WAV + durées exactes.

- Modèle `hexgrad/Kokoro-82M`, **100 % local**, sur torch CPU, sortie 24 kHz.
  Poids ~330 Mo téléchargés au premier usage puis mis en cache.
- `prepare_text` normalise les guillemets et tirets typographiques, supprime les
  caractères de largeur nulle.
- `apply_pronunciations` applique le **dictionnaire phonétique**
  (`config/pronunciations.json`) aux noms propres **avant toute synthèse** : c'est le
  garde-fou n°3 du cahier des charges. Une valeur entre barres obliques (`/ˈdɛstiə/`) est
  traitée comme une transcription phonémique.
- Chaque **phrase** est synthétisée séparément puis assemblée avec
  `DEFAULT_SENTENCE_GAP_S` = **0,2 s** de silence ; chaque segment de scène reçoit
  `DEFAULT_PADDING_S` = **0,18 s** de silence final. Ces deux valeurs ont été resserrées
  (0,4 s donnait une narration traînante) sans jamais chevaucher la phrase suivante.
- `duration_s` est calculée sur le **nombre exact d'échantillons** : c'est elle qui
  pilote la timeline, donc elle doit rester exacte.
- Le pipeline Kokoro est **injectable** (`KokoroTTS(pipeline=...)`) : les tests
  n'emploient aucun modèle réel.
- Voix par défaut : `LANGUAGE_VOICES["en"] = ("a", "am_puck")` — voix masculine
  américaine, 161 mots/min au banc d'essai. `synthesize(text, voice=...)` et
  `synthesize_text` acceptent une voix par appel.

API : `prepare_text`, `load_pronunciations`, `apply_pronunciations`, `resolve_voice`,
`write_wav`, `read_wav`, `concat_wavs`, `export_mp3`, `build_pipeline`,
`KokoroTTS.{synthesize, synthesize_text, synthesize_scene, synthesize_analysis}`,
`save_manifest`, `load_manifest`, `format_manifest`.

#### Banc d'essai des voix — `src/modules/voice_lab.py`

28 voix anglaises (`ENGLISH_VOICES` = 11 féminines US + 9 masculines US + 4 féminines UK
+ 4 masculines UK). `compare_voices` synthétise le même extrait avec chacune (moteurs
en `threading.local`), `build_voice_report` produit un HTML avec un lecteur `<audio>`,
la durée, le débit en mots/min et le temps de calcul par voix. Commande :
`python -m src.main voices`.

### 6.5 Module 5a — Timeline builder (`src/modules/timeline_builder.py`)

**Rôle** : `ChapterAnalysis` + `VoiceoverManifest` → `Timeline`.

Règles :

- Seules les scènes qui ont un segment audio sont montées (le remplissage n'a pas de
  voix off, il est donc naturellement exclu).
- **La durée d'affichage d'une scène est exactement la durée de son segment audio**,
  silence final inclus. Chaque case clé reçoit d'abord `DEFAULT_MIN_CLIP_S` = 2,5 s, puis
  le reste au prorata des hauteurs. Si la scène a trop de cases pour sa durée, les plus
  petites sont écartées (`limit_panels_for_duration`).
- Mouvement (`motion_for`) : `punch_in` pour les cases `action_heavy`, sinon
  `ken_burns`. **Jamais de scroll.**
- Sous-titres : la narration d'origine est découpée en blocs de 2 à 4 mots
  (`split_subtitle_text`), répartis sur la durée de parole au prorata des caractères. Les
  blocs ne franchissent pas une fin de phrase et sont équilibrés à l'intérieur d'une
  phrase (9 mots → 3+3+3, 5 mots → 3+2) : jamais un bloc d'un seul mot quand la phrase en
  compte plusieurs.
- Design sonore : un bruitage au début de chaque case des scènes `action` selon le cycle
  `impact → swoosh → roar → swoosh`, et un `impact` au début de chaque case `punch_in`
  des autres scènes. Musique par ambiance (`MOOD_BY_EMOTION` projette les 11 émotions sur
  `calm` / `tense` / `action`), bouclée, fondu enchaîné de 1,5 s à chaque changement
  d'ambiance, à **−22 dB**.

`concat_timelines(timelines, gap_s=0.6, …)` fusionne plusieurs chapitres : réécrit tous
les médias en **chemins absolus** (aucun fichier n'est copié), décale les temps et
renumérote les indices de scène. C'est la brique de la commande `merge`. La transition de
la dernière case d'un chapitre est retirée : elle enjamberait le silence inter-chapitre.

#### Dynamisme du montage (`--dynamics`)

Trois niveaux : `none` (montage sobre d'origine), `subtle` (animations de sous-titres
seules), **`punchy` (défaut)**. Décidé ici, une seule fois, puis consommé à l'identique
par CapCut et par l'aperçu — `timeline.json` reste la source de vérité unique.

- **`animation_for_cue`** : une animation d'entrée par bloc de sous-titre, choisie selon
  l'émotion (`弹入` rebond pour l'action, `故障` glitch pour la tension, `渐显` fondu pour
  le calme, `逐字` mot à mot par défaut), plus une boucle (`心跳`, `颤抖`) sur les seules
  scènes intenses. Sa durée vaut **35 % du bloc**, bornée à [0,12 s ; 0,40 s] : nos blocs
  font 2 à 4 mots, souvent moins de 0,5 s, et le défaut CapCut de 0,5 s les consommerait
  entièrement. Un bloc trop court reste fixe.
- **`transition_for`** / **`attach_transitions`** : une transition **uniquement aux
  changements de scène** dont l'une est `action` ou `tension`, jamais à l'intérieur d'une
  scène. Portée par la case **précédente** (convention `pycapcut`). Durée plafonnée à
  0,30 s et à **15 % de la plus courte des deux cases** ; sous 0,20 s, coupe franche.
- **`build_vfx_clips`** : un effet superposé par scène `action` / `epic` / `mystery` d'au
  moins 2 s — effet **intégré CapCut** (`冲刺` lignes de vitesse, `光晕` halo, `下雨`
  pluie) ; un asset local à canal alpha est préféré s'il est fourni.

⚠ **Le point le plus important du dynamisme** : dans le catalogue CapCut, **1131 des 1137
transitions ont `is_overlap`**, c'est-à-dire qu'elles consomment du temps sur les cases
voisines. Si CapCut applique ce recouvrement, l'image prend de l'avance sur la voix.
`Timeline.transition_drift_s` chiffre cette dérive potentielle et elle est journalisée à
chaque construction (~2 à 3 s par chapitre). L'aperçu ffmpeg, lui, joue la transition
**à l'intérieur** de la fin du clip sortant : il ne dérive donc jamais, ce qui en fait la
référence de synchronisation.

### 6.6 Module 5b — CapCut builder (`src/modules/capcut_builder.py`)

**Rôle** : `Timeline` → projet CapCut, via `pycapcut` 0.0.3.

Pistes produites :

| Piste | Contenu |
|---|---|
| **V1** | fond de chaque case, **pré-rendu** en JPEG 1920x1080 (case étirée, flou gaussien, −30 % de luminosité) par le même code que l'aperçu ffmpeg |
| **V2** | la case PNG **à sa résolution native**, centrée ; réduite si elle dépasse le cadre, **jamais agrandie** |
| **A1** | un segment de voix off par scène |
| **A2 / A2b** | musique par ambiance à −22 dB, **bouclée** en segments contigus ; deux pistes alternées pour que les fondus (`add_fade`) se chevauchent |
| **A3** | bruitages |
| **T1** | sous-titres, avec leur animation d'entrée et leur boucle |
| **E1** | effets superposés des scènes spectaculaires (piste d'effets dédiée) |

Points techniques :

- **Convention d'échelle CapCut** : une image importée est ajustée (« contain ») dans le
  cadre à l'échelle 1.0 ; l'échelle native vaut donc `1 / contain`. C'est ce que calcule
  `native_scale` / `panel_placement`.
- Images clés : Ken Burns 100 % → 105 % sur la durée du clip, ou punch-in 105 % atteint
  en 0,2 s puis maintenu. **Jamais au-delà de 105 %** (`MAX_ZOOM`).
- Temps exprimés en **microsecondes entières** (`_us`). `contiguous_ranges`,
  `cue_ranges` et `loop_ranges` garantissent des clips contigus sans trou ni
  chevauchement.
- Sous-titres : `SUBTITLE_FONT_SIZE` = 9.0, `SUBTITLE_TRANSFORM_Y` = −0.78 (centré bas),
  contour noir `SUBTITLE_BORDER_WIDTH` = 20.0 sur l'échelle CapCut 0-100 — soit ≈ 2 px en
  1080p pour la taille 9 (le défaut CapCut de 40 donne ≈ 4 px). Polices candidates :
  `Rubik_Bold`, `Montserrat`, `Anton`, `BebasNeue`.
- `ensure_background` écrit les fonds **à côté du fichier de la case**
  (`panel_path.parent / "backgrounds"`) : une compilation multi-chapitres réutilise ainsi
  les fonds déjà rendus de chaque chapitre.
- `detect_capcut_drafts_dir` cherche
  `%LOCALAPPDATA%\CapCut\User Data\Projects\com.lveditor.draft`.
- `copy_draft` : si CapCut tient le dossier verrouillé (`WinError 32`, projet ouvert), la
  copie se rabat sur `<nom> (2)`. Le pipeline dégrade l'échec en avertissement, le
  brouillon restant disponible dans `<out_dir>/capcut`.
- **La construction est sérialisée entre threads** (`_DRAFT_LOCK`). `pycapcut` sonde
  chaque média avec `pymediainfo.MediaInfo.parse()`, qui n'est pas sûr en concurrence :
  deux constructions simultanées échouaient en `ParseError: syntax error: line 1,
  column 0` ou en « fichier sans piste vidéo ni image » sur des PNG valides. L'étape ne
  durant que quelques secondes par chapitre, la sérialiser ne coûte presque rien et rend
  `--max-render-workers` supérieur à 1 utilisable.
- `_catalog_member` résout les noms d'animations, de transitions et d'effets à l'exécution :
  un nom absent du catalogue `pycapcut` est ignoré avec un avertissement, sans faire
  échouer tout le montage.
- **Piège d'API** : une transition doit être attachée au segment **avant** `add_segment` —
  c'est l'insertion dans le script qui collecte `segment.transition` dans les *materials*.
  L'attacher après produit un brouillon sans aucune transition, **sans lever d'erreur**.
  Un test de non-régression couvre ce point.
- `compensated_ranges(clips, "shift")` décale les cases de la durée cumulée des
  transitions qui les précèdent, pour absorber le recouvrement. **Non vérifié dans
  CapCut** : le défaut reste `"none"`.

### 6.7 Module 5c — Preview renderer (`src/modules/preview_renderer.py`)

**Rôle** : `Timeline` → MP4, pour contrôler le rendu sans ouvrir CapCut.

Reproduit volontairement la composition prévue dans CapCut : fond flouté et assombri
(`BACKGROUND_DARKEN` = 0.7, réduction ×8 puis flou σ = 6), case au premier plan à sa
résolution native centrée, sous-titres blancs en police grasse avec contour noir de 2 px
(mis à l'échelle de la hauteur d'image), voix off, musique bouclée avec fondus, bruitages.

- Images composées avec OpenCV/NumPy, envoyées à l'exécutable ffmpeg fourni par
  `imageio-ffmpeg` (H.264 + AAC) : **aucune installation système requise**.
- `native_scale` ne renvoie jamais un facteur > 1 (pas d'agrandissement) ;
  `zoom_factor` est plafonné à `MAX_ZOOM`.
- `mix_bgm_clips` boucle la musique ; `mix_sfx_clips` superpose les bruitages.
- Police : première trouvée parmi `FONT_CANDIDATES` (Montserrat-Bold,
  Montserrat-ExtraBold, Rubik-Bold, arialbd, segoeuib, bahnschrift, puis DejaVu et Arial
  pour Linux/macOS).
- L'avance d'image tient le clip précédent pendant les silences inter-chapitres d'une
  compilation :
  `while clip_pos < len(clips) - 1 and t >= clips[clip_pos + 1].start_s`.

**Dynamisme rejoué localement** (`PreviewRenderer(timeline, dynamics=True)`) :

- `blend_transition` approxime quatre familles — `whip` (balayage + flou de filé),
  `glitch` (décalage des canaux R/B + bandes déplacées), `flash_black` / `flash_white`,
  `dissolve`. Elle se joue **dans** la fin du clip sortant : la durée totale ne bouge pas,
  donc la synchronisation image / voix reste exacte, contrairement à CapCut.
- `apply_vfx` superpose quatre familles en mélange additif : `speed_lines`, `glow`,
  `rain`, `sparks`.
- `cue_animation_state` / `apply_cue_state` rejouent l'entrée du sous-titre (rebond avec
  dépassement, fondu, agrandissement, glitch, révélation mot à mot par masquage de la
  droite) et la boucle (`heartbeat`, `shake`).

⚠ **Performance** : ces effets doivent rester en opérations `cv2` sur des `uint8`. Une
première version en `float32` NumPy coûtait **38 à 41 ms par image** (rendu ×6 plus lent) ;
en `cv2.add` / `cv2.addWeighted` / `cv2.convertScaleAbs` elle est tombée à **2 à 3 ms**,
soit +26 % sur le rendu complet. Ne pas réintroduire d'arithmétique NumPy pleine image.

### 6.8 Module 6a — Gemini manager (`src/utils/gemini_manager.py`)

Un **seul** `GeminiManager`, thread-safe, partagé par tous les chapitres d'un lot.

**Clés.** Lues par `gemini_api_keys()` dans l'ordre : valeur explicite →
`GEMINI_API_KEYS` (virgules, points-virgules ou retours à la ligne, lu aussi depuis
`.env`) → `GEMINI_API_KEY` puis `GOOGLE_API_KEY` → fichier `.gemini_key`. Dédoublonnées
en conservant l'ordre. La clé active est gardée d'un appel à l'autre ; on passe à la
suivante si elle est invalide/expirée (400 `API_KEY_INVALID`, 401, 403) ou si son quota
journalier est atteint **pour le modèle courant**.

**Réessais.** Sur 429 « par minute », 5xx ou erreur réseau : backoff exponentiel
**2, 4, 8, 16 s** (le délai suggéré par l'API est honoré s'il est plus long, borné à
65 s), **4 essais au plus par clé** (`MAX_ATTEMPTS_PER_KEY`), puis rotation.

**Cascade de modèles.** Quand toutes les clés ont échoué sur le modèle courant, passage
au suivant. Un modèle inconnu du compte (404) est sauté. Cascade par défaut :

```
gemini-3.5-flash, gemini-3.6-flash, gemini-3.7-flash, gemini-3.8-flash,
gemini-2.5-flash, gemini-3.5-flash-lite, gemini-3.1-flash-lite,
gemini-2.5-flash-lite, gemini-pro-latest
```

⚠ `gemini-2.0-flash`, `gemini-1.5-flash` et `gemini-1.5-pro` — les modèles nommés dans le
cahier des charges d'origine — **ne sont plus servis sur ce compte (404, retirés)**. La
cascade a été reconstruite à partir d'une sonde `models.list()` réelle. Chaque modèle a
son propre quota journalier, donc une cascade longue = plus de chapitres par jour et par
clé. Les Flash-Lite ont des quotas plus larges mais une qualité moindre sur l'analyse
d'images : les garder en fin de cascade.

**Garde-fou anti-gâchis de cascade.** Quand **toutes** les clés sont momentanément
limitées à la minute sur le modèle courant, le manager patiente
`RATE_LIMIT_COOLDOWN_S` = 30 s et réessaie **le même modèle**
(`MAX_RATE_LIMIT_ROUNDS` = 2) au lieu de descendre la cascade : le quota journalier du
modèle est encore disponible. Un sentinelle `_RATE_LIMITED` distinct de `_ROTATE` porte
cette distinction.

**RPM global.** `RateLimiter` (fenêtre glissante de 60 s) borne le nombre de requêtes
démarrées par minute, toutes clés et tous chapitres confondus.

**Observabilité.** `status()` expose le modèle courant, la cascade, la clé active
(masquée), l'état de chaque clé, le nombre d'appels, de rotations, de descentes de
cascade, l'attente RPM cumulée, les latences (`n`, `mean`, `p50`, `p95`, `max`, `total`)
et `calls_by_model`.

Classificateurs d'erreurs exportés et testés séparément : `is_daily_quota_error`,
`is_rate_limit_error`, `is_invalid_key_error`, `is_model_unavailable_error`,
`is_transient_error`, `suggested_retry_delay`. Exceptions : `GeminiManagerError`,
`QuotaExhaustedError`, `ModelUnavailableError`.

### 6.9 Module 6b — Batch processor (`src/modules/batch_processor.py`)

Traite plusieurs chapitres en parallèle en **isolant la charge réseau de la charge
locale**, au moyen de sémaphores `asyncio` distincts. Les étapes du pipeline étant
synchrones, elles tournent dans des threads (`asyncio.to_thread`) et `asyncio.gather`
orchestre l'ensemble.

| Sémaphore | Défaut | Ce qu'il borne |
|---|---|---|
| `max_scrape_workers` | = `max_chapters` | téléchargements + découpes simultanés |
| `max_chapters` | 5 | chapitres **analysés** simultanément par Gemini |
| `max_gemini_rpm` | 10 | requêtes Gemini par minute, toutes clés et chapitres confondus |
| `max_tts_workers` | **2** | synthèses Kokoro simultanées (CPU) |
| `max_render_workers` | 1 | montages + rendus ffmpeg simultanés |

Le scraping a **son propre sémaphore** pour prendre de l'avance pendant que le quota
Gemini s'écoule.

⚠ `max_tts_workers = 2` est un **optimum mesuré**, pas une estimation. Sur 16 cœurs
(torch utilise déjà 8 threads par flux) : 1 worker = 3,2× temps réel, **2 workers = 4,7×
(optimum)**, 4 workers = 4,4× — la sur-souscription des cœurs fait régresser. Au-delà de
2, c'est du gâchis. Cette valeur avait d'abord été fixée à 4 sur la base d'une mesure
erronée (somme de temps par tâche, eux-mêmes gonflés par la contention) ; seule une
mesure en temps d'horloge a donné le bon chiffre. **Toute remesure doit se faire en temps
d'horloge.**

**Reprise.** `batch_status.json` (à la racine par défaut, `--status-file` pour en changer)
enregistre l'état de chaque URL. Une exécution suivante **saute les chapitres déjà
`done`** et reprend les autres ; les étapes déjà calculées sur disque sont réutilisées.
Sauvegarde atomique par `os.replace`. `--redo` et `--force` passent outre le saut des
chapitres `done` (`src/modules/batch_processor.py:342`).

Schéma du fichier de suivi :

```json
{
  "created": "…", "updated": "…",
  "chapters": {
    "<url>": {
      "status": "pending|processing|done|failed",
      "attempts": 1, "stage": "done",
      "out_dir": "…", "started": "…", "finished": "…", "error": null,
      "episode_no": 1, "title": "…",
      "n_panels": 118, "model": "gemini-3.5-flash", "n_scenes": 17,
      "voice_s": 454.2, "duration_s": 453.6, "total_duration_s": 454.2,
      "preview": "…", "capcut": "…",
      "timings": { "scrape+slice": 113.4, "analyze": …, "tts": …, "timeline": …, "capcut": …, "preview": … }
    }
  },
  "gemini": { "model": "…", "models": [...], "active_key": "cle #1 (...oywg)",
              "keys": [...], "calls": 2, "rotations": 0, "cascades": 0,
              "rpm_wait_s": 0.0, "latency_s": {...}, "calls_by_model": {...} },
  "run": { "finished": "…", "elapsed_s": 348.4, "chapters": 2, "done": 2,
           "failed": 0, "skipped": 0, "timings": {...} }
}
```

API : `BatchOptions`, `Stages` (étapes injectables pour les tests), `BatchStatus`,
`BatchReport`, `read_url_list`, `resolve_chapter_urls`, `run_batch`, `timing_summary`,
`status_timings`, `format_status`, `format_report`.

### 6.10 Module 7 — Miniatures YouTube (`src/modules/thumbnail/`)

Trois étapes séparées, chacune utilisable seule, orchestrées par `ThumbnailStage`.

**Étape 1 — `analyzer.py`.** Travaille sur le **script déjà écrit**, pas sur les images :
aucune case n'est renvoyée à Gemini, donc l'analyse ne coûte qu'un appel de texte. Le
prompt demande une scène au **contraste ou à l'ironie visuelle** forte, une accroche de 1
à 3 mots, et le côté de la flèche. `build_prompt` n'envoie que le **début et la fin** du
script : le contraste exploitable s'y trouve presque toujours. `normalize_hook` met en
majuscules et borne à 3 mots / 18 caractères ; `normalize_brief` **replace la flèche à
l'opposé du sujet** si le modèle l'a posée du même côté (elle le recouvrirait).

**Étape 2 — `image_backends.py`.** Protocole `ImageBackend` avec trois implémentations :
`GeminiImageBackend` (défaut, cascade `gemini-3-pro-image` → `gemini-3.1-flash-image` →
`gemini-2.5-flash-image` → `gemini-3.1-flash-lite-image`, clés et rotation déjà en place),
`StabilityBackend` et `LocalWebuiBackend`. Choix par `THUMBNAIL_IMAGE_BACKEND`. Le style
`high quality anime, manhwa, ultra-detailed, dramatic lighting` est imposé, et un prompt
négatif exclut texte, filigranes, logos et membres difformes — la miniature reçoit son
texte au compositing, jamais au rendu.

**Étape 3 — `compositor.py`.** Pillow seul, aucun réseau. `build_thumbnail(image_path,
text, arrow_pos)` : recadrage en 1280x720, saturation `ImageEnhance.Color(...).enhance(1.15)`,
flèche jaune redimensionnée et **retournée** quand elle est à droite (elle vise toujours le
centre), puis accroche en police d'affichage (Bangers si fournie, sinon Impact), remplissage
`#FFFF00` ou blanc, contour noir de 8 px via `stroke_width`, boîte inclinée de −5 à −10
degrés. L'angle est **déterministe** (dérivé du texte) : deux rendus du même chapitre sont
identiques. La taille de police est trouvée par dichotomie pour remplir la boîte. Si aucun
asset de flèche n'est fourni, une flèche géométrique simple est dessinée.

**Orchestration — `pipeline.py`.** `ThumbnailStage.build()` propage les erreurs (commande
CLI dédiée) ; **`ThumbnailStage.run()` n'échoue jamais** : il absorbe tout, journalise en
avertissement et renvoie `None`. C'est la porte d'entrée du pipeline vidéo — une miniature
ratée ne doit pas empêcher la vidéo d'exister. `build(base_image=...)` saute l'étape 2 et
ne consomme donc aucun quota d'image.

### 6.11 Utilitaires

- **`src/utils/http.py`** — `build_session`, `ensure_mandatory_headers` (injecte
  `User-Agent` + `Referer`, indispensables), `get_with_retry` (backoff, respect de
  `Retry-After`).
- **`src/utils/image_utils.py`** — `to_numpy_rgb`, `to_pil`, `rgb_to_gray`, `rgb_to_bgr`,
  normalisation `uint8`.
- **`src/utils/audio_assets.py`** — `find_sfx` / `ensure_default_sfx`,
  `find_bgm` / `ensure_default_bgm`, `bgm_for_mood`. Si `config/sfx/` ou `config/bgm/`
  ne contient pas les fichiers attendus, des sons et boucles de **substitution sont
  synthétisés** en NumPy et écrits sur place : le montage est ainsi toujours validable,
  **jamais sans musique**. Formats acceptés : `.wav`, `.mp3`, `.ogg`, `.flac`. Boucles de
  24 s à 24 kHz.
- **`src/utils/report.py`** — `build_html_report` : page de contrôle alignant vignettes
  (data-URI), texte narré et lecteurs audio par scène.
- **`src/utils/config.py`** — `PROJECT_ROOT`, langues par défaut, `load_dotenv`
  (analyseur `.env` minimal), `gemini_api_keys`, `load_gemini_api_key`,
  `gemini_key_hint`.

---

## 7. Règles produit (non négociables)

Ces règles viennent de retours utilisateur explicites et ont coûté des itérations. Les
enfreindre est une régression, pas un choix d'implémentation.

### 7.1 Image

1. **Aucun rognage en largeur, jamais.** Pas d'auto-crop.
2. Une case de plus de 1200 px est **sous-découpée horizontalement** en 2 ou 3 blocs
   conservant **100 % de la largeur d'origine**.
3. **Zoom plafonné à 105 %** de la taille native (`MAX_ZOOM = 1.05`). Aucun
   agrandissement au-delà, pour ne jamais pixelliser.
4. La case est **centrée à sa résolution native** dans le cadre 1920x1080 ; le **fond
   flouté et assombri de 30 %** comble les bandes 16:9 à gauche et à droite.
5. Pas de défilement vertical (`scroll_vertical` n'est plus produit).

### 7.2 Son

6. **La musique est toujours présente**, à **−22 dB**, bouclée si la vidéo est plus
   longue qu'elle, avec fondu enchaîné à chaque changement d'ambiance. Piste A2.
7. Voix off par défaut `am_puck`.
8. Silences resserrés : 0,2 s entre phrases, 0,18 s en fin de scène.

### 7.3 Sous-titres

9. Blocs de **2 à 4 mots** maximum. Décision confirmée le 2026-09-12 : le karaoké
   authentique (surlignage progressif) aurait exigé des blocs de 10 à 15 mots et a été
   écarté pour cette raison. Avec des blocs courts, une animation d'entrée rapide **est**
   déjà l'effet « pop-up mot à mot ».
10. Police **grasse** (Montserrat / Rubik), **blanc**, **contour noir de 2 px**, centrés
    en bas.

### 7.4 Dynamisme

11. Transitions **uniquement aux changements de scène** `action` ou `tension`, jamais à
    l'intérieur d'une scène, jamais au-delà de 0,40 s ni de 15 % de la plus courte des
    deux cases.
12. Le « zéro rognage » est assoupli **pendant une transition seulement** : le mouvement
    propre de l'effet est autorisé ; hors transition le zoom reste plafonné à 105 % et le
    découpage des cases ne rogne jamais en largeur.
13. Effets superposés sur les scènes `action` / `epic` / `mystery` : **effets intégrés
    CapCut** par défaut. `pycapcut` n'expose aucun mode de fusion, donc un asset
    superposé doit porter un **vrai canal alpha** — un MP4 sur fond noir s'afficherait en
    rectangle noir.

### 7.4 Écriture

11. Aucune formule visuelle, aucune intro générique, exactement un appel à l'abonnement
    dans un paragraphe du milieu (voir §6.3).
12. Ordre de lecture strictement croissant d'un paragraphe au suivant.

---

## 8. CLI — référence

Point d'entrée : `python -m src.main <commande>`. Typer, `add_completion=False`.

### 8.1 `run` — un chapitre de bout en bout

```powershell
python -m src.main run "<url viewer>" [options]
```

| Option | Défaut | Rôle |
|---|---|---|
| `--out`, `-o` | `output/<serie>_ep<N>` | dossier de sortie |
| `--language` | `en` | langue de narration et de voix |
| `--voice` | `am_puck` | voix Kokoro |
| `--speed` | 1.0 | vitesse de lecture |
| `--sentence-gap` | 0.2 | silence entre deux phrases |
| `--padding` | 0.18 | silence en fin de scène |
| `--model` | cascade | modèle Gemini préféré |
| `--thinking-budget` | — | budget de réflexion |
| `--batch-size` | 12 | images par lot (mode deux étapes) |
| `--gemini-batch-delay` | 2.5 | pause forcée entre deux envois |
| `--keyframe-workers` | 4 | appels « cases clés » en parallèle |
| `--multi-call` | faux | repasse en mode deux étapes (~18 appels) |
| `--dynamics` | `punchy` | `none`, `subtle` ou `punchy` (voir §6.5) |
| `--transition-compensation` | `none` | `none` ou `shift` (à vérifier dans CapCut) |
| `--thumbnail` | faux | générer aussi la miniature (1 appel texte + 1 image) |
| `--thumbnail-backend` | `gemini` | `gemini`, `stability` ou `local` |
| `--preview-seconds` | 60 | durée de l'aperçu (0 = complet) |
| `--no-preview`, `--no-capcut` | — | sauter ces étapes |
| `--capcut-dir` | auto-détecté | dossier projets CapCut |
| `--name` | `<Série> - <Épisode>` | nom du projet CapCut |
| `--bgm` / `--bgm-dir` / `--no-bgm` | `config/bgm` | musique |
| `--sfx-dir` / `--no-sfx` | `config/sfx` | bruitages |
| `--cta` / `--no-cta` | selon la langue | appel à l'abonnement |
| `--fps` | 60 | images par seconde |
| `--max-gemini-rpm` | 10 | requêtes par minute |
| `--force` | faux | recalculer toutes les étapes |
| `--verbose`, `-v` | faux | logs DEBUG |

### 8.2 `batch` — une série en parallèle

```powershell
python -m src.main batch "<url serie ou episode>" --start-chapter 1 --end-chapter 20
python -m src.main batch --url-list chapitres.txt --max-chapters 5
```

Reprend toutes les options de `run`, plus : `--url-list`, `--start-chapter`,
`--end-chapter`, `--max-chapters` (5), `--max-scrape-workers`, `--max-gemini-rpm` (10),
`--max-tts-workers` (2), `--max-render-workers` (1), `--status-file`, `--out-root`,
`--no-retry-failed`, `--dry-run`, `--redo {analyze|tts|montage}`, `--force`.

### 8.3 `merge` — compilation en un seul projet

```powershell
python -m src.main merge --pattern "output/<serie>_ep*" --name "<Série> - Ep. 1-10"
```

`--pattern` (glob depuis la racine) ou des dossiers en arguments ; `--out`, `--name`,
`--gap` (0.6 s entre chapitres), `--preview-seconds` (**−1 = aucun aperçu**, le défaut),
`--no-capcut`, `--capcut-dir`. Les dossiers sont triés par `episode_no` lu dans
`chapter.json` — **pas** alphabétiquement, sinon l'épisode 10 passerait avant le 2.
Aucun média n'est copié : la timeline fusionnée pointe en chemins absolus.

### 8.4 `voices` — comparatif des voix

```powershell
python -m src.main voices [--voices af_heart,am_puck] [--text "…"] [--workers 2]
```

Écrit les extraits et `voices.html` dans `output/voices`.

### 8.5 `stats` — temps mesurés d'un lot

```powershell
python -m src.main stats [--status-file batch_xxx.json]
```

Relit les durées enregistrées : par étape, par chapitre, par seconde de vidéo.

### 8.6 `thumbnail` — miniature d'un chapitre déjà analysé

```powershell
python -m src.main thumbnail output/<chapitre>
python -m src.main thumbnail output/<chapitre> --image base.png --text "WAKE UP" --arrow left
```

`--backend`, `--text` et `--arrow` (surchargent le modèle), `--image` (réutilise une
illustration, **aucun quota d'image**), `--white`, `--arrow-asset`, `--font`, `--out`.
Avec `--image` **et** `--text`, aucune requête n'est faite : compositing seul.

### 8.7 `preview` / `capcut` — refaire une sortie seule

```powershell
python -m src.main preview output/<chapitre> --seconds 60
python -m src.main capcut  output/<chapitre> --name "…"
```

Repartent de `timeline.json` : ni Gemini, ni Kokoro, donc instantanés et gratuits.

---

## 9. Performances mesurées

Mesuré le 2026-09-12 sur *The Unparalleled Hidden Rank Equipment*, épisodes 1 à 10
(1295 cases, 89,5 min de vidéo), mode une requête, une seule clé gratuite,
`--max-chapters 4 --max-gemini-rpm 5 --max-tts-workers 2` :

| Étape | Médiane par chapitre |
|---|---|
| scraping + découpe | 132 s |
| **analyse Gemini** | **64 s** |
| voix Kokoro | 256 s |
| timeline | 0 s |
| brouillon CapCut | 8 s |
| aperçu 60 s | 55 s |
| **par chapitre, en parallèle** | **153 s** |

- **10 chapitres en 25,5 minutes**, 0 échec.
- 502 s de traitement par chapitre pour 525 s de vidéo → **0,96 s de calcul par seconde
  de vidéo produite**.
- **13 appels Gemini pour 10 chapitres** (dont 3 reprises sur 503 « forte demande ») ;
  0 rotation de clé, 0 descente de cascade, 1,5 s d'attente RPM au total.
- Latence par appel : médiane 53,5 s, p95 67 s, max 121 s — élevée, car tout le chapitre
  part en une requête, mais 14 fois moins d'appels.
- **Le goulot d'étranglement est désormais Kokoro, plus Gemini.** C'est l'inverse de
  l'ère « deux étapes », où 63 % du temps était de l'attente Gemini.

Comparaison des deux modes sur un même chapitre (ép. 5 de *The Lazy Lord*, 131 cases) :

| | Deux étapes | Une requête |
|---|---|---|
| Appels Gemini | 20 | 1 à 2 |
| Analyse | 416 s | 114 s |
| Chapitre complet | 586 s | 297 s |
| Jetons | 241 k | 196 k |
| Mots du script | 1266-1466 | 1373 |

Ordre de grandeur d'un chapitre de 176 cases en mode une requête : **196 k jetons,
13,8 Mo** (limites : 1 M de jetons, 20 Mo en ligne).

---

## 10. Quotas Gemini (palier gratuit)

- **~20 à 25 requêtes par jour, par modèle et par clé.** Observé : `gemini-3.5-flash` a
  servi 21 appels avant de rendre un quota épuisé.
- **5 requêtes par minute.** Utiliser `--max-gemini-rpm 5` sur le palier gratuit ; 10
  provoque un backoff 429 permanent.
- Les Flash-Lite ont des quotas journaliers nettement plus larges → fin de cascade.
- `count_tokens` est un **point d'accès séparé** : il ne consomme pas le quota de
  génération.

**Arithmétique de capacité.** À 1-2 appels par chapitre, une clé gratuite traite
**10 à 20 chapitres par jour et par modèle**, et la cascade de 9 modèles multiplie ce
budget. En mode deux étapes (~18 appels par chapitre) c'était **1 chapitre par jour**.
C'est le gain principal du mode une requête.

---

## 11. Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

**209 tests, tous passants, ~16 s.** Aucun test par défaut ne touche le réseau ni ne
charge un modèle : le client Gemini et le pipeline Kokoro sont injectables et remplacés
par des doubles.

- `tests/conftest.py` — fixture *autouse* qui neutralise les `_sleep` (les backoffs ne
  ralentissent pas la suite) et isole `.env`, `GEMINI_API_KEYS` et
  `GEMINI_MODEL_CASCADE` de l'environnement réel.
- `tests/synthetic_strip.py` — générateur de bandes de webtoon synthétiques (gouttières
  contrôlées) pour tester le slicer de façon déterministe.
- `test_*_local.py` (`analyzer`, `scraper`, `slicer`, `tts`) — tests nécessitant le
  réseau ou les vrais modèles, séparés exprès.

Couverture par fichier : `test_analyzer.py` (890 l.), `test_scraper.py` (571),
`test_slicer.py` (595), `test_timeline.py` (269), `test_gemini_manager.py` (256),
`test_batch.py` (257), `test_tts.py` (242), `test_capcut.py` (234), `test_preview.py`
(176), `test_config.py` (108), `test_voice_lab.py` (84), `test_audio_assets.py` (78),
`test_report.py` (67).

---

## 12. Limites connues et points non vérifiés

À traiter en priorité par qui reprend le projet. **Ne pas présenter ces points comme
validés.**

1. **Aucun brouillon CapCut n'a jamais été confirmé comme s'ouvrant dans CapCut.** Les
   projets sont écrits et copiés dans le dossier de CapCut, et la structure est testée,
   mais l'ouverture réelle dans l'éditeur n'a pas été observée. En découlent trois
   inconnues : la disponibilité des polices `Rubik_Bold` / `Montserrat` dans CapCut, le
   rendu des fondus `add_fade`, et l'épaisseur exacte du contour des sous-titres
   (`SUBTITLE_BORDER_WIDTH = 20.0` est calibré par le calcul, pas par l'œil).
2. **`config/bgm/` et `config/sfx/` ne contiennent que des placeholders synthétisés** en
   NumPy. Ils permettent de valider le montage ; ils sont à remplacer par de vraies
   musiques et de vrais bruitages, en conservant les mêmes noms de fichiers
   (`calm`/`tense`/`action`, `swoosh`/`impact`/`roar`).
2 bis. **Résolu le 2026-09-12.** Ni Montserrat, ni Rubik, ni Bangers n'étaient installées :
   l'aperçu retombait silencieusement sur `arialbd.ttf`, donc **les aperçus rendus avant
   cette date sont en Arial Bold**, pas dans la police spécifiée. Les trois `.ttf` sont
   désormais dans `config/fonts/`, que `FONT_CANDIDATES` consulte **avant** les polices
   système. Les aperçus antérieurs doivent être refaits (`--redo montage`, sans quota).
   Côté CapCut le nom de police est résolu par l'éditeur, ce qui reste non vérifié.
2 ter. **Le quota d'images Gemini est épuisé sur ce compte** : la cascade des quatre
   modèles d'images a été parcourue en entier le 2026-09-12 et chacun a répondu « quota
   journalier atteint ». Les étapes 1 (analyse) et 3 (compositing) sont validées
   bout en bout ; l'étape 2 ne l'a **jamais été avec une vraie génération**. Options :
   activer la facturation, ajouter des clés, ou basculer sur `stability` / `local`.
3. **`debug_overlay.png` pèse ~20 Mo par chapitre.** Sur une série de 20 chapitres, c'est
   400 Mo d'artefact de débogage. Aucune option ne permet de le désactiver.
4. Le mode une requête **ne produit pas les beats** dans `scenes.json` (`beats: []`) :
   la traçabilité case → beat → paragraphe n'existe qu'en `--multi-call`.
5. La régénération d'une analyse n'est pas déterministe : relancer `--redo analyze` donne
   un script différent (longueur comprise). Mesuré : 1653 → 1438 mots sur un même
   chapitre.
6. Les chapitres 6 à 10 de *The Lazy Lord Masters the Sword* restent `failed` dans
   `batch_status.json`, leurs cases déjà téléchargées, prêts à reprendre.
7. Le projet **n'est pas sous git**. Aucun historique, aucune possibilité de revenir en
   arrière autrement qu'à la main.

---

## 13. Pour reprendre le développement

### 13.1 Mise en route

```powershell
cd C:\Users\Nouamane\Documents\ManhwaMaker
.\.venv\Scripts\python.exe -m pytest tests -q          # doit afficher 209 passed
.\.venv\Scripts\python.exe -m src.main run --help
```

Vérifier qu'une clé est disponible (`.gemini_key` ou `.env`). Pour tout ce qui touche à
Kokoro, exporter d'abord `$env:HF_HUB_DISABLE_SYMLINKS = "1"`.

### 13.2 Boucle de travail économe en quota

Le quota Gemini est la ressource rare. Pour itérer sur la voix, le montage, les
sous-titres ou le rendu **sans dépenser un seul appel** :

```powershell
python -m src.main batch --url-list liste.txt --redo tts        # voix + montage
python -m src.main preview output/<chapitre> --seconds 60       # rendu seul
python -m src.main capcut  output/<chapitre>                    # brouillon seul
```

Seul `--redo analyze` (et `--force`) consomme du quota.

### 13.3 Pièges à connaître

| Piège | Conséquence | Parade |
|---|---|---|
| `&&` en PowerShell | erreur de parseur | `A; if ($?) { B }` |
| `python -c "..."` avec guillemets | quoting cassé | écrire un script dans le scratchpad |
| Caractère non ASCII dans un `print` | plantage console cp1252 | `_ascii()` |
| Kokoro sans `HF_HUB_DISABLE_SYMLINKS` | `WinError 1314` | poser la variable |
| Projet CapCut ouvert | `WinError 32` à la copie | déjà géré : repli `<nom> (2)` |
| Tri alphabétique des dossiers de chapitres | ép. 10 avant ép. 2 | trier par `episode_no` |
| `Panel.image` dans un `model_dump` | absent par conception | relire le PNG |
| Mesurer le parallélisme en sommant les temps par tâche | conclusion fausse | mesurer en **temps d'horloge** |
| `add_transition` après `add_segment` | brouillon **sans transition**, sans erreur | attacher avant l'insertion |
| Deux brouillons CapCut construits en parallèle | `ParseError` / « sans piste vidéo » | déjà géré : `_DRAFT_LOCK` |
| Arithmétique NumPy `float32` pleine image par frame | rendu ×6 plus lent | `cv2.add` / `addWeighted` en `uint8` |
| Mesurer la saturation sur du bruit encodé en JPEG | effet masqué par la chrominance | image lisse + sortie PNG |

### 13.4 Où intervenir selon l'objectif

| Objectif | Fichiers |
|---|---|
| Qualité d'écriture du script | prompts et garde-fous de `analyzer.py` (§6.3) |
| Choix des cases montées | `normalize_recap`, `_enforce_reading_order` (`analyzer.py`) |
| Rythme, silences, voix | `tts_engine.py` (constantes en tête de fichier) |
| Durées, mouvements, sous-titres, son | `timeline_builder.py` |
| Rendu visuel CapCut | `capcut_builder.py` + `preview_renderer.py` (garder les deux cohérents) |
| Consommation de quota | `gemini_manager.py` (cascade) et mode une requête (`analyzer.py`) |
| Parallélisme, reprise | `batch_processor.py` |
| Découpe des cases | `slicer.py` |
| Transitions, animations de texte, effets superposés | décision dans `timeline_builder.py`, émission dans `capcut_builder.py` **et** `preview_renderer.py` |
| Miniatures | `thumbnail/analyzer.py` (accroche), `image_backends.py` (rendu), `compositor.py` (mise en page) |

⚠ `capcut_builder.py` et `preview_renderer.py` doivent rester **cohérents** : l'aperçu
n'a d'intérêt que s'il reproduit fidèlement ce que fera CapCut. `capcut_builder` importe
d'ailleurs `make_background` et `native_scale` depuis `preview_renderer`, et les deux
partagent les constantes de zoom de `timeline_builder`.

---

## 14. Tableau récapitulatif des constantes de réglage

| Constante | Valeur | Fichier |
|---|---|---|
| `DEFAULT_VARIANCE_THRESHOLD` | 15.0 | `slicer.py` |
| `DEFAULT_MIN_GAP` / `DEFAULT_MARGIN_PADDING` | 20 / 15 px | `slicer.py` |
| `MIN_PANEL_HEIGHT` | 180 px | `slicer.py` |
| `MERGE_SMALL_BELOW` / `MERGE_MAX_GAP` | 250 / 150 px | `slicer.py` |
| `SPLIT_MAX_HEIGHT` / `SPLIT_MAX_PIECES` | 1200 px / 3 | `slicer.py` |
| `GIANT_PANEL_HEIGHT` | 1500 px | `slicer.py` |
| `DEFAULT_MODEL` | `gemini-3.5-flash` | `analyzer.py` |
| `DEFAULT_BATCH_SIZE` / `MAX_BATCH_SIZE` | 12 / 15 | `analyzer.py` |
| `DEFAULT_MAX_IMAGE_WIDTH` | 1024 px | `analyzer.py` |
| `DEFAULT_MAX_SLICE_HEIGHT` / `MAX_SLICES_PER_PANEL` | 2048 px / 4 | `analyzer.py` |
| `MAX_KEY_PANELS_PER_PARAGRAPH` | 4 | `analyzer.py` |
| `MAX_CANDIDATES_PER_PARAGRAPH` | 8 | `analyzer.py` |
| `SCRIPT_WORDS_PER_PANEL` | 11 | `analyzer.py` |
| `SCRIPT_MIN_WORDS` / `SCRIPT_MAX_WORDS` | 250 / 1500 | `analyzer.py` |
| `SCRIPT_MIN_PARAGRAPHS` / `SCRIPT_MAX_PARAGRAPHS` | 4 / 40 | `analyzer.py` |
| `SCRIPT_MIN_LENGTH_RATIO` | 0.65 | `analyzer.py` |
| `MAX_INLINE_PAYLOAD_BYTES` | 16 Mo | `analyzer.py` |
| `BATCH_DELAY_S` | 2.5 s | `analyzer.py` |
| `DEFAULT_KEYFRAME_WORKERS` | 4 | `analyzer.py` |
| `CTA_POSITION` | 0.4 | `analyzer.py` |
| `DEFAULT_TEMPERATURE` / `DEFAULT_TIMEOUT_MS` | 0.4 / 120 000 | `analyzer.py` |
| `BACKOFF_SCHEDULE` | 2, 4, 8, 16 s | `gemini_manager.py` |
| `MAX_ATTEMPTS_PER_KEY` | 4 | `gemini_manager.py` |
| `RATE_LIMIT_COOLDOWN_S` / `MAX_RATE_LIMIT_ROUNDS` | 30 s / 2 | `gemini_manager.py` |
| `DEFAULT_MAX_RPM` | 10 | `gemini_manager.py` |
| `SAMPLE_RATE` | 24 000 Hz | `tts_engine.py` |
| `DEFAULT_PADDING_S` | 0.18 s | `tts_engine.py` |
| `DEFAULT_SENTENCE_GAP_S` | 0.2 s | `tts_engine.py` |
| voix anglaise par défaut | `am_puck` | `tts_engine.py` |
| `MAX_ZOOM` | 1.05 | `timeline_builder.py` |
| `KEN_BURNS_ZOOM` / `PUNCH_IN_ZOOM` / `PUNCH_IN_S` | 0.05 / 0.05 / 0.2 s | `timeline_builder.py` |
| `DEFAULT_MIN_CLIP_S` | 2.5 s | `timeline_builder.py` |
| `DEFAULT_MAX_SUBTITLE_WORDS` | 4 | `timeline_builder.py` |
| `DEFAULT_DYNAMICS` | `punchy` | `timeline_builder.py` |
| `CUE_ANIMATION_RATIO` | 0.35 (borné 0,12–0,40 s) | `timeline_builder.py` |
| `TRANSITION_EMOTIONS` | `action`, `tension` | `timeline_builder.py` |
| `DEFAULT_TRANSITION_S` / `MIN` / `MAX` | 0,30 / 0,20 / 0,40 s | `timeline_builder.py` |
| `TRANSITION_MAX_CLIP_RATIO` | 0.15 | `timeline_builder.py` |
| `MIN_VFX_SCENE_S` / `DEFAULT_VFX_OPACITY` | 2,0 s / 0,85 | `timeline_builder.py` |
| `VFX_STRENGTH` | 0.55 | `preview_renderer.py` |
| `DEFAULT_TRANSITION_COMPENSATION` | `none` | `capcut_builder.py` |
| `THUMBNAIL_WIDTH` x `HEIGHT` | 1280x720 | `thumbnail/compositor.py` |
| `SATURATION_BOOST` | 1.15 (+15 %) | `thumbnail/compositor.py` |
| `TEXT_STROKE_PX` / `TEXT_ANGLE_RANGE` | 8 px / −10 à −5 deg | `thumbnail/compositor.py` |
| `MAX_HOOK_WORDS` / `MAX_HOOK_CHARS` | 3 / 18 | `models/thumbnail.py` |
| `MIN_CUE_DURATION_S` | 0.25 s | `timeline_builder.py` |
| `DEFAULT_BGM_CROSSFADE_S` | 1.5 s | `timeline_builder.py` |
| `DEFAULT_CHAPTER_GAP_S` | 0.6 s | `timeline_builder.py` |
| `DEFAULT_BGM_GAIN_DB` | −22 dB | `models/timeline.py` |
| `DEFAULT_SFX_GAIN_DB` | −12 dB | `timeline_builder.py` |
| `BACKGROUND_DARKEN` | 0.7 (−30 %) | `preview_renderer.py` |
| `SUBTITLE_STROKE_PX` | 2 px | `preview_renderer.py` |
| `SUBTITLE_BORDER_WIDTH` | 20.0 (≈ 2 px) | `capcut_builder.py` |
| `SUBTITLE_FONT_SIZE` / `SUBTITLE_TRANSFORM_Y` | 9.0 / −0.78 | `capcut_builder.py` |
| `max_chapters` / `max_tts_workers` / `max_render_workers` | 5 / **2** / 1 | `batch_processor.py` |
| résolution / fps | 1920x1080 / 60 | `pipeline.py` |
| `DEFAULT_PREVIEW_SECONDS` | 60 s | `pipeline.py` |

---

## 15. Dépendances

```
opencv-python>=4.9,<5      numpy>=1.26,<3       pillow>=10.2
requests>=2.31             beautifulsoup4>=4.12 lxml>=5.0
google-genai>=1.0
kokoro>=0.9.4              soundfile>=0.12      pydub>=0.25
typer>=0.12                pydantic>=2.6
pytest>=8.0
```

Non listés dans `requirements.txt` mais utilisés : **`pycapcut` 0.0.3** (génération du
brouillon) et **`imageio-ffmpeg`** (binaire ffmpeg du rendu d'aperçu). `kokoro` installe
`torch` et `misaki` ; espeak-ng (via `espeakng-loader` sur Windows) sert aux langues non
anglaises et aux mots hors dictionnaire.
