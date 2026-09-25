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

CLI Python (Windows, pas de git) : URL webtoons.com → brouillon CapCut + aperçu MP4.
Chaîne : scrape+stitch → découpe en cases → 1 requête Gemini (script + cases-clés) → voix Kokoro locale → `timeline.json` → brouillon CapCut + aperçu ffmpeg.

Interpréteur : `.\.venv\Scripts\python.exe` (jamais `python` nu).

```powershell
.\.venv\Scripts\python.exe -m src.main batch "<url serie>" --start-chapter 1 --end-chapter 3 --max-chapters 3 --max-gemini-rpm 5
.\.venv\Scripts\python.exe -m src.main run "<url viewer>"
.\.venv\Scripts\python.exe -m src.main stats        # relit batch_status.json
.\.venv\Scripts\python.exe -m pytest -q             # 457 tests, ~30 s (TOONSPLIT_SLOW=1 : +3 tests modeles)
```

| Étape | Fichier |
|---|---|
| scrape / découpe | `src/modules/scraper.py`, `slicer.py` (cases de lecture, lues par l'IA) ; `figure_panels.py` (cases personnages `figures/`, seules montées par défaut, `--panels slicer` pour l'ancien mode) |
| upscale personnages | `src/modules/upscaler.py` (Real-ESRGAN anime ONNX → `figures/hd_<L>x<H>/`, seul dossier monté ; `--no-upscale` pour s'en passer) |
| analyse Gemini | `src/modules/analyzer.py`, `src/utils/gemini_manager.py` |
| memoire de serie | `src/modules/series_memory.py` (`characters.json` par chapitre) |
| voix | `src/modules/tts_engine.py` |
| montage | `src/modules/timeline_builder.py` |
| sortie | `src/modules/capcut_builder.py`, `preview_renderer.py` (garder cohérents) |
| lot | `src/modules/batch_processor.py` |
| decoupe intelligente (test, hors pipeline) | `src/modules/toonsplit/`, `python -m src.modules.toonsplit` (voir `RAPPORT_TOONSPLIT.md`) |

## Règles non négociables (profil LONG)

- Cases en **résolution native** sur fond flouté. Jamais d'upscale, jamais de recadrage.
  Seule exception (décidée le 25/09) : les cases personnages, agrandies par IA (Real-ESRGAN) à 90 % du cadre, x4 max.
- **Zoom ≤ 105 %** (`MAX_ZOOM` dans `timeline_builder.py`).
- **Musique toujours présente**, BGM à −22 dB, bouclée.
- Sous-titres 2-4 mots, contour 2 px.
- Narration, voix et liens Webtoons en **anglais** (`webtoons.com/en/`). Voix par défaut `am_puck`.

Le profil **SHORT** (9:16) déroge volontairement aux deux premières règles (recadrage par saillance, zoom jusqu'à 3,2×). Ce sont des règles LONG, pas des règles globales.

## Contraintes Gemini (free tier, 1 clé dans `.gemini_key`)

- Toujours `--max-gemini-rpm 5`. 10 provoque un backoff 429 permanent.
- Requête unique par défaut (~2 appels/chapitre) **mais** bascule silencieuse en mode deux étapes (~40 appels) si le payload dépasse 16 Mo. Les chapitres denses coûtent 20× plus cher. `max_image_width` (1024) n'est pas exposé en CLI.
- Cascade : `3.5-flash`, `2.5-flash`, `3.5-flash-lite`, `3.1-flash-lite` marchent. `3.7-flash` et `3.8-flash` renvoient des 503 en boucle sans jamais aboutir.
- Ne jamais afficher une clé.

## Pièges

- `--preview-seconds 0` = rendu **complet**, pas « pas d'aperçu » (c'est `--no-preview`).
- `--redo` n'existe que sur `batch`, pas sur `run`.
- `batch --thumbnail` est accepté et ne fait rien.
- `stage_montage` n'a aucun cache : timeline + CapCut + ffmpeg refaits à chaque passage.
- Mode `figures` (défaut depuis le 24/09) : `scenes.json` reste en numéros de cases de lecture, la traduction vers `figures/` se fait au montage (`remap_analysis`). Extraction ~1,5 min/chapitre en CPU, en cache dans `figures/figures_params.json`. Les personnages d'une même case (même bande horizontale) sont regroupés en une image (`--figures-separate` pour les séparer). Ils sont ensuite agrandis par IA (`upscaler.py`) : ~4 s/chapitre sur GPU, ~85 s en CPU, en cache dans `hd_<L>x<H>/upscale_params.json`. Le zoom ≤ 105 % s'applique à l'image agrandie ; le rythme du montage reste calé sur la taille d'origine (`native_height`).
- GPU : le venv utilise `onnxruntime-directml` (Radeon RX 7800 XT) à la place d'`onnxruntime` (même module, jamais les deux). Détection des personnages vérifiée identique (69/69 boîtes). waifu2x donne des images fausses sous DirectML.
- `build_capcut_draft` ne reçoit aucun profil → le format SHORT n'est pas exportable vers CapCut.
- `batch_status.json` est un journal global unique, réécrit sans verrou.
- PowerShell : pas de `&&`, pas de ternaire. Caractère non-ASCII dans un `print` Python → crash console cp1252.
- Mesurer le parallélisme en **temps d'horloge**, jamais en somme de temps par tâche.

## État à vérifier

Aucun brouillon CapCut n'a jamais été ouvert dans l'éditeur : signe des axes de keyframe, polices, fondus et dérive des transitions restent non vérifiés.
`RESUME.md` et `PRD.md` (12/09) sont périmés : ils ignorent le format SHORT et sous-estiment `src/` de 67 %. Vérifier dans le code avant de citer la doc.
