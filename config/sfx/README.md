# Bruitages (SFX)

Un bruitage court est inséré au début de chaque case des scènes d'émotion `action`, en
suivant le cycle impact → swoosh → roar → swoosh, et un `impact` au début de chaque case
`action_heavy` (punch-in) des autres scènes (piste **A3 sfx** du brouillon CapCut, mixé à
−12 dB dans l'aperçu).

Fichiers attendus dans ce dossier (WAV, MP3, OGG ou FLAC) :

- `impact.*`
- `swoosh.*`
- `roar.*`

S'ils manquent, le pipeline synthétise des **bruitages de substitution** (`<kind>.wav`, ignorés
par git) pour valider le montage : remplacez-les par de vrais effets sous les mêmes noms.
Autre dossier : `--sfx-dir`, désactivation : `--no-sfx`.
