# Musiques de fond (BGM) par ambiance

Les émotions des scènes sont ramenées à trois ambiances ; à chaque changement d'ambiance la
musique change avec un fondu enchaîné de 1,5 s (pistes **A2 bgm / A2b bgm** du brouillon
CapCut, −22 dB sous la voix). Une musique plus courte que la vidéo est **bouclée**.

| Ambiance | Émotions Gemini | Fichier attendu |
|---|---|---|
| `calm` | neutral, calm, happy, romance, humor | `calm.*` |
| `tense` | tension, mystery, fear, sad | `tense.*` |
| `action` | action, epic | `action.*` |
| repli | ambiance sans fichier | `default.*` |

Formats : WAV, MP3, OGG ou FLAC. Si ce dossier ne contient **aucune** musique, le pipeline
synthétise trois boucles de substitution (`calm.wav`, `tense.wav`, `action.wav`, ~24 s, ignorées
par git) pour qu'il y ait toujours de la musique : remplacez-les par vos propres titres (droits)
sous les mêmes noms ; dès qu'un fichier est présent, rien n'est synthétisé et les ambiances
manquantes se replient sur `default.*` ou sur la première musique disponible.
Autre dossier : `--bgm-dir` ; musique unique : `--bgm fichier` ; aucune musique : `--no-bgm`.
