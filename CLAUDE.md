# CLAUDE.md — Auto-Manhwa Recap Generator

## Style de réponse (prioritaire)

- Réponds en français.
- **Uniquement des puces.** Jamais de paragraphes rédigés.
- Une idée par puce. Une phrase courte par puce.
- Mots simples. Pas de jargon technique sans l'expliquer en trois mots.
- Va directement au résultat. Pas de préambule, pas de récapitulatif final.
- 10 puces maximum. Si le sujet en demande plus, c'est qu'il va dans un fichier.
- Ne liste pas les options écartées. Donne la recommandation et sa raison, en une puce.
- Ne raconte pas ce que tu viens de faire : c'est déjà visible. Donne le résultat.
- Erreur : une puce pour l'erreur, une puce pour la cause. Rien d'autre.
- Pas de tableau sauf demande explicite ou vraie comparaison chiffrée.
- Rapports longs (audit, revue) : dans un fichier, pas dans le chat.

## Le projet

CLI Python (Windows, pas de git) : URL webtoons.com ou asurascans.com → brouillon CapCut + aperçu MP4.
Chaîne : scrape+stitch → découpe en cases → 1 appel Claude Opus 5.5 (script + cases-clés, CLI local ; Gemini en repli) → voix Kokoro locale → `timeline.json` → brouillon CapCut + aperçu ffmpeg.

Interpréteur : `.\.venv\Scripts\python.exe` (jamais `python` nu).

```powershell
.\.venv\Scripts\python.exe -m src.main batch "<url serie>" --start-chapter 1 --end-chapter 3 --max-chapters 3 --max-gemini-rpm 5
.\.venv\Scripts\python.exe -m src.main batch "<url serie>" --start-chapter 1 --end-chapter 20 --max-chapters 3 --max-gemini-rpm 5 --no-cta --compile   # une seule video
.\.venv\Scripts\python.exe -m src.main run "<url viewer>"
.\.venv\Scripts\python.exe -m src.main stats        # relit batch_status.json
.\.venv\Scripts\python.exe -m src.studio             # interface web http://127.0.0.1:8777 (lancer, suivre, rendre)
.\.venv\Scripts\python.exe -m pytest -q             # ~30 s (TOONSPLIT_SLOW=1 : +3 tests modeles)
```

| Étape | Fichier |
|---|---|
| scrape / découpe | `src/modules/scraper.py` (aiguille selon le site), `asura.py` (Asura Scans), `slicer.py` (cases de lecture, lues par l'IA) ; `figure_panels.py` (cases personnages `figures/`, seules montées par défaut, `--panels slicer` pour l'ancien mode) |
| upscale personnages | `src/modules/upscaler.py` (Real-ESRGAN anime ONNX → `figures/hd_<L>x<H>/`, seul dossier monté ; `--no-upscale` pour s'en passer) |
| script (Claude) | `src/modules/script_writer.py` + prompt `script_prompt.md` ; banc d'essai `eval/script_bench/run_bench.py` |
| analyse Gemini (repli, `--script-ai gemini`) | `src/modules/analyzer.py`, `src/utils/gemini_manager.py` |
| memoire de serie | `src/modules/series_memory.py` (`characters.json` par chapitre) |
| voix | `src/modules/tts_engine.py` |
| montage | `src/modules/timeline_builder.py` |
| sortie | `src/modules/capcut_builder.py`, `preview_renderer.py` (garder cohérents) |
| essai Kdenlive (hors pipeline) | `src/modules/kdenlive_builder.py` : projet `.kdenlive` + rendu `melt` sans interface, `python -m src.modules.kdenlive_builder output/<chapitre> --fps 30` (voir `RECHERCHE_EDITEURS_VIDEO.md`) ; style dynamique (transition à chaque coupe selon l'émotion, mouvements variés, impacts, étalonnage) dans `kdenlive_style.py` |
| lot | `src/modules/batch_processor.py` |
| interface web (Studio) | `src/studio/` : `server.py` (API FastAPI + flux SSE), `jobs.py` (file d'attente, un processus par traitement), `eta.py` (temps restant), `library.py` (vidéos de `output/`), `static/` (Preact + htm copiés dans `vendor/`, sans compilation) ; progression écrite par `src/utils/progress.py` quand `MM_PROGRESS_FILE` est défini |
| decoupe intelligente (test, hors pipeline) | `src/modules/toonsplit/`, `python -m src.modules.toonsplit` (voir `RAPPORT_TOONSPLIT.md`) |

## Règles non négociables (profil LONG)

- Cases en **résolution native** sur fond flouté. Jamais d'upscale, jamais de recadrage.
  Seule exception (décidée le 25/09) : les cases personnages, agrandies par IA (Real-ESRGAN) à 90 % du cadre, x3 max (plus petites, elles ne sont pas montées).
- **Zoom ≤ 105 %** (`MAX_ZOOM` dans `timeline_builder.py`).
- **Musique toujours présente**, BGM à −22 dB, bouclée.
- Sous-titres 2-4 mots, contour 2 px.
- Narration, voix et liens Webtoons en **anglais** (`webtoons.com/en/`). Voix par défaut `am_fenrir,am_michael` (mélange des deux, choisi le 25/09).

Le profil **SHORT** (9:16) déroge volontairement aux deux premières règles (recadrage par saillance, zoom jusqu'à 3,2×). Ce sont des règles LONG, pas des règles globales.

## Script par Claude (défaut depuis le 25/09)

- Claude Opus 5.5 via le CLI local (`claude -p`, abonnement, sans clé API). Choisi au banc d'essai : Opus 8,5/10, Sonnet 6,5 (répliques dans la mauvaise bouche), Gemini 5, Haiku 2,5.
- Opus 5.5 mesuré en production (Bad Born Blood ch. 1, 182 cases) : 81 s et 1,04 $ au tarif API, décompté des limites de l'abonnement (~42 $ pour 40 chapitres). Chapitre complet de bout en bout : ~7,5 min.
- Effort fixé à `medium` (`script_writer.DEFAULT_EFFORT`, mesuré le 25/09 sur 2 chapitres, `output/script_bench_audit/`) : même coût et durée que le défaut du CLI ; `low` nettement moins bon (5/10, script deux fois plus court) ; `high` jusqu'à 2× plus lent, +13 % de coût, sans gain constant. Comparer : `run_bench.py <chapitre> --models opus@low,opus@medium --judge`.
- CLI ≥ 2.1.282 obligatoire (le 2.1.270 refuse `claude-opus-5-5`). Mise à jour : `npm install -g @anthropic-ai/claude-code@latest`.
- Claude en échec (limite d'usage, CLI absent) → le chapitre passe à l'analyse Gemini, ancien prompt. Un quota Gemini épuisé n'arrête plus un lot Claude.
- L'appel à l'abonnement (« subscribe ») est toujours inséré vers 40 % du script (`--no-cta` pour l'enlever) : il brise le 4e mur que le prompt protège, décision en attente.

## Contraintes Gemini (free tier, 1 clé dans `.gemini_key`)

- Toujours `--max-gemini-rpm 5`. 10 provoque un backoff 429 permanent.
- Requête unique par défaut (~2 appels/chapitre) **mais** bascule silencieuse en mode deux étapes (~40 appels) si le payload dépasse 16 Mo. Les chapitres denses coûtent 20× plus cher. `max_image_width` (1024) n'est pas exposé en CLI.
- Cascade : `3.5-flash`, `2.5-flash`, `3.5-flash-lite`, `3.1-flash-lite` marchent. `3.7-flash` et `3.8-flash` renvoient des 503 en boucle sans jamais aboutir.
- Ne jamais afficher une clé.

## Pièges

- Vidéo de plusieurs chapitres = `batch --compile` (ou `merge`) : compilation `output/<serie>_compilation_ch1-20/` autonome (médias reliés par liens physiques dans `media/`, fiches de série dans `series_memory/`), projet CapCut, extrait MP4 de 2 min, puis **suppression des dossiers de chapitres** (`--keep-chapters` pour les garder). La vidéo complète se rend dans CapCut (ffmpeg : 57 min et 1,8 Go pour 20 chapitres). Les chapitres suivants (21-40) relisent la mémoire archivée. Relancer une plage déjà compilée : `--force` (dossiers supprimés, statut « done »).
- Asura Scans (ajouté le 25/09) : URL `https://asurascans.com/comics/<serie>-<id>/chapter/<n>`, page de série = toute la série en `batch`. La bannière de crédits (1re page, paysage 1200x800) est écartée au téléchargement. Chapitres bonus décimaux (`74.5`) : `episode_no` peut être décimal, dossier `<serie>_ep74.5`, pris dans une plage `--start-chapter 74 --end-chapter 75`. Le suffixe `-<id>` du slug n'entre ni dans la clé de série ni dans le nom de dossier (Asura le faisait tourner sur son ancien domaine, non vérifié sur le nouveau). Chapitres ~30 % plus longs que Webtoons : tient en une requête Gemini au palier JPEG 75 (13,6 Mo mesurés sur Bad Born Blood ch. 1).
- `--preview-seconds 0` = rendu **complet**, pas « pas d'aperçu » (c'est `--no-preview`).
- `--redo` n'existe que sur `batch`, pas sur `run`.
- `batch --thumbnail` est accepté et ne fait rien.
- `stage_montage` n'a aucun cache : timeline + CapCut + ffmpeg refaits à chaque passage.
- Mode `figures` (défaut depuis le 24/09) : `scenes.json` reste en numéros de cases de lecture, la traduction vers `figures/` se fait au montage (`remap_analysis`). Extraction ~1,5 min/chapitre en CPU, en cache dans `figures/figures_params.json`. Les personnages d'une même case (même bande horizontale) sont regroupés en une image (`--figures-separate` pour les séparer). Ils sont ensuite agrandis par IA (`upscaler.py`) : ~4 s/chapitre sur GPU, ~85 s en CPU, en cache dans `hd_<L>x<H>/upscale_params.json`. Le zoom ≤ 105 % s'applique à l'image agrandie ; le rythme du montage reste calé sur la taille d'origine (`native_height`). Tri au montage (`select_figures`, décidé le 25/09) : personnage à plus de x3 pour remplir le cadre jamais monté ; personnage hors cases clés du script monté seulement si confiance du détecteur ≥ 0,4 (le détecteur note mal les gros plans, pas plus haut).
- GPU : le venv utilise `onnxruntime-directml` (Radeon RX 7800 XT) à la place d'`onnxruntime` (même module, jamais les deux). Détection des personnages vérifiée identique (69/69 boîtes). waifu2x donne des images fausses sous DirectML.
- Essai Kdenlive (26/09, Kdenlive 26.08 installé par winget dans `%LOCALAPPDATA%\Programs\kdenlive`) : chapitre de 5 min 51 rendu par `melt` en 111 s en 30 i/s, 226 s en 60 i/s (aperçu Python : 173 s en 60 i/s). La carte AMD n'encode que la vidéo (`h264_amf`), tous les effets sont calculés par le processeur, et plusieurs `melt` en parallèle ne vont pas plus vite. Les sous-titres ne sont que dans le rendu `melt` : Kdenlive plante sur un filtre de sous-titres écrit à la main (le `.ass` s'importe dans Kdenlive). Les modes de composition des pistes sont perdus à l'ouverture dans Kdenlive : effets superposés en calques transparents. Style dynamique : +50 % de temps de rendu (168 s en 30 i/s pour le même chapitre). Au-delà de 12 min, rendu **par parties** (`render_in_parts`, 26/09) : sur la compilation de 2 h 20, melt chargeait le projet en 73 s puis mettait 30 ms par image (15 sur un chapitre), et l'encodeur AMD a manqué de mémoire à 48 % (`AllocSurface() failed with error 6`) → MP4 sans index, illisible. Chaque partie (un chapitre de compilation, sinon ~6 min de scènes) est un petit projet vidéo seul ; le son de toute la vidéo est mixé en Python (`mix_audio`) ; ffmpeg assemble sans réencoder. Parties gardées dans `kdenlive/render_parts/` jusqu'à l'assemblage : relancer reprend à la première partie manquante. Tout MP4 s'écrit en `.part.mp4` puis est renommé ; la bibliothèque du Studio écarte un MP4 sans « moov ».
- Studio (26/09) : port 8777 (le 8765 est pris par un autre `server.py` de l'utilisateur). Chaque traitement lance la vraie commande (`batch` avec `--max-chapters 3 --max-gemini-rpm 5`, ou le rendu Kdenlive) dans son propre processus, sortie dans un journal : fermer le navigateur ou arrêter le serveur ne l'arrête pas, et au redémarrage le serveur le retrouve par son pid. Un lot à la fois, un rendu à la fois. État dans `studio_data/` (ignoré par git). Temps restant : simulation de l'ordonnancement du lot (3 préparations, scripts dans l'ordre de la série, 2 voix, 1 montage) avec les médianes de `batch_status.json`, calée sur le dernier lot (41 min prévues pour 20 chapitres, 39 mesurées), puis corrigée par les chapitres déjà finis du lot en cours. `?static` dans l'URL coupe le flux en direct (captures d'écran sans tête). « Refaire à partir de » refuse un dossier au nom personnalisé (`_prod`) : le pipeline écrirait ailleurs.
- `build_capcut_draft` ne reçoit aucun profil → le format SHORT n'est pas exportable vers CapCut.
- `batch_status.json` est un journal global unique, réécrit sans verrou.
- PowerShell : pas de `&&`, pas de ternaire. Caractère non-ASCII dans un `print` Python → crash console cp1252.
- Mesurer le parallélisme en **temps d'horloge**, jamais en somme de temps par tâche.

## État à vérifier

Aucun brouillon CapCut n'a jamais été ouvert dans l'éditeur : signe des axes de keyframe, polices, fondus et dérive des transitions restent non vérifiés.
`RESUME.md` et `PRD.md` (12/09) sont périmés : ils ignorent le format SHORT et sous-estiment `src/` de 67 %. Vérifier dans le code avant de citer la doc.
