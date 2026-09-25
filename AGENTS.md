# AGENTS.md — Auto-Manhwa Recap Generator

## Style de réponse (prioritaire)

- **Réponds en français, court.** Vise 5-15 lignes. Une réponse longue doit être justifiée par une demande explicite.
- **Pas de préambule, pas de récapitulatif final.** Attaque directement par le résultat ou la réponse.
- **Pas de tableau, pas de titres en gras à chaque paragraphe** sauf si je le demande ou si je compare vraiment plusieurs choses.
- **Ne liste pas les options que tu n'as pas retenues.** Donne ta recommandation et sa raison en une ligne.
- **Ne répète pas ce que tu viens de faire** si c'est visible dans les outils. Dis le résultat, pas le chemin parcouru.
- Si un run échoue, donne l'erreur et la cause probable en 2 lignes, pas l'historique complet.
- Les rapports longs (audit, redécouverte) vont dans un fichier, pas dans le chat.

## Le projet

CLI Python (Windows, pas de git) : URL webtoons.com → brouillon CapCut + aperçu MP4.
Chaîne : scrape+stitch → découpe en cases → 1 requête Gemini (script + cases-clés) → voix Kokoro locale → `timeline.json` → brouillon CapCut + aperçu ffmpeg.

Interpréteur : `.\.venv\Scripts\python.exe` (jamais `python` nu).

```powershell
.\.venv\Scripts\python.exe -m src.main batch "<url serie>" --start-chapter 1 --end-chapter 3 --max-chapters 3 --max-gemini-rpm 5
.\.venv\Scripts\python.exe -m src.main run "<url viewer>"
.\.venv\Scripts\python.exe -m src.main stats        # relit batch_status.json
.\.venv\Scripts\python.exe -m pytest -q             # 249 tests, ~19 s
```

| Étape | Fichier |
|---|---|
| scrape / découpe | `src/modules/scraper.py`, `slicer.py` |
| analyse Gemini | `src/modules/analyzer.py`, `src/utils/gemini_manager.py` |
| voix | `src/modules/tts_engine.py` |
| montage | `src/modules/timeline_builder.py` |
| sortie | `src/modules/capcut_builder.py`, `preview_renderer.py` (garder cohérents) |
| lot | `src/modules/batch_processor.py` |

## Règles non négociables (profil LONG)

- Cases en **résolution native** sur fond flouté. Jamais d'upscale, jamais de recadrage.
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
- `build_capcut_draft` ne reçoit aucun profil → le format SHORT n'est pas exportable vers CapCut.
- `batch_status.json` est un journal global unique, réécrit sans verrou.
- PowerShell : pas de `&&`, pas de ternaire. Caractère non-ASCII dans un `print` Python → crash console cp1252.
- Mesurer le parallélisme en **temps d'horloge**, jamais en somme de temps par tâche.

## État à vérifier

Aucun brouillon CapCut n'a jamais été ouvert dans l'éditeur : signe des axes de keyframe, polices, fondus et dérive des transitions restent non vérifiés.
`RESUME.md` et `PRD.md` (12/09) sont périmés : ils ignorent le format SHORT et sous-estiment `src/` de 67 %. Vérifier dans le code avant de citer la doc.
