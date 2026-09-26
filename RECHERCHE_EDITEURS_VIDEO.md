# Remplacer CapCut : quel éditeur vidéo automatisable ?

Recherche du 25/09/2026.

**Besoin :**
- construire le montage par programme ;
- avoir assez d'effets et de transitions ;
- lancer le rendu sans intervention, idéalement sur la carte AMD RX 7800 XT ;
- gratuit de préférence, et bien documenté.

## Comparaison

| Éditeur | Prix | Automatisation | Effets / transitions | Rendu sur la carte AMD | Documentation |
|---|---|---|---|---|---|
| **Kdenlive** (moteur MLT) | Gratuit, open source | Projet en XML ; rendu sans interface (`melt`) | Des centaines d'effets, fondus, glissés, dizaines de volets (+ KDE Store) | Oui : profil AMF (Windows) pour l'encodage ; les effets restent sur le processeur | Bon manuel (version 26.08), format XML moyennement décrit |
| Shotcut (moteur MLT) | Gratuit | Même format XML ; rendu `melt` | Moins d'effets que Kdenlive | AMF : `hevc_amf` fonctionne, bug `h264_amf` signalé en juin 2026 | Moyenne (surtout le forum) |
| DaVinci Resolve gratuit | Gratuit | **Python retiré le 08/09/2026 (v21.1)** ; scripts Lua internes seulement, sans accès aux fichiers | Le plus riche (Fusion) | **Non** : encodage GPU H.264/H.265 réservé à NVIDIA dans la version gratuite | Bonne |
| DaVinci Resolve Studio | 299 $ (licence à vie) | API Python complète, mais transitions et effets non plaçables clip par clip (passer par un import OTIO / FCPXML) | Le plus riche | Oui (cartes AMD RDNA 3, AV1 compris) | Bonne, surtout grâce à la communauté |
| Blender (VSE) | Gratuit | **API Python la mieux documentée**, rendu sans interface | Fondus, volets, tout paramètre animable ; effets via le compositeur | Non : pas d'encodage GPU (toujours au programme 2026) | Excellente |
| Remotion (code React) | Gratuit jusqu'à 3 personnes | Tout en code, pas de montage manuel possible | Paquet de transitions et effets CSS / WebGL illimités | Non : encodage GPU NVIDIA seulement, rendu par navigateur sur le processeur | Excellente |
| Editly (Node + ffmpeg) | Gratuit | Montage décrit en JSON | ~70 transitions (gl-transitions), peu d'effets | Via ffmpeg | Faible, un seul mainteneur |
| CapCut (actuel) | Gratuit | **Aucune API officielle** ; format de brouillon rétro-conçu (pyCapCut…) | Très riche | Oui, mais rendu à la main dans l'application | Aucune officielle |

## Recommandation : Kdenlive

- **Gratuit et durable** : logiciel open source maintenu par KDE, documentation à jour (26.08).
- **Tout automatique** : on génère un fichier projet et `melt` rend la vidéo sans ouvrir l'application. On ne passe plus par un rendu manuel dans CapCut.
- **Retouches toujours possibles** : le même fichier s'ouvre dans Kdenlive, comme le brouillon CapCut aujourd'hui.
- **Format public et stable** : le XML de MLT n'est pas rétro-conçu, il ne cassera pas à la prochaine mise à jour, contrairement au JSON de CapCut.
- **Tout notre montage s'y exprime** :
  - images sur fond flouté ;
  - zoom par images clés ;
  - sous-titres avec contour ;
  - musique à −22 dB ;
  - bruitages ;
  - fondus et volets entre scènes.
- **Carte graphique AMD** : l'encodage passe par AMF (profil Windows officiel, `h264_amf`).

**Limites à mesurer :**
- Les effets (flou, zoom) sont calculés sur le processeur. Il faut donc chronométrer un chapitre avant de décider.
- La version Windows est moins éprouvée que la version Linux.

**Écartés :**
- **DaVinci Resolve gratuit** : il perd son API Python et n'encode pas sur carte AMD.
  - Resolve Studio (299 $) est la seule option plus puissante.
  - Même payant, il oblige à passer par des imports de fichiers intermédiaires pour les transitions.
- **Blender** : l'API est idéale, mais le rendu est sans encodage sur la carte graphique, donc lent pour une compilation d'une heure.
- **Remotion** : excellente documentation, mais pas de montage manuel possible et un rendu lent sans carte NVIDIA.

## Résultats de l'essai (26/09)

Chapitre témoin : Bad Born Blood ch. 1 (5 min 51, 64 plans, 439 sous-titres, 8 effets superposés).

| Rendu | Durée du rendu | Vitesse |
|---|---|---|
| Aperçu Python actuel (60 i/s, processeur) | 173 s | 2,0× le temps réel |
| `melt` 60 i/s, encodage par la carte AMD | 226 s | 1,6× |
| `melt` 30 i/s, encodage par la carte AMD | **111 s** | 3,2× |
| Kdenlive lui-même (30 i/s, réglage par défaut, sans sous-titres) | 199 s | 1,8× |
| CapCut | non mesuré (export manuel) | ? |

**Ce qui marche :**
- Le projet est généré à partir du `timeline.json` et rendu sans interface, en une commande.
- Kdenlive ouvre le projet et le rend lui-même.
- L'image obtenue est quasi identique à l'aperçu : fond flouté, zoom, fondus, volets, flash, glitch, lueur, pluie, lignes de vitesse, sous-titres.

**Ce qui ne marche pas comme prévu :**
- **Vitesse** : pas de gain par rapport à notre rendu actuel.
  - La carte AMD n'encode que la vidéo : effets, zoom et superpositions restent calculés par le processeur.
  - `melt` n'utilise qu'une partie des 16 cœurs, et 8 rendus en parallèle ne vont pas plus vite.
  - Seul le passage en 30 i/s divise le temps par deux.
- **Sous-titres** : ils ne sont que dans le rendu automatique. Kdenlive plante sur un filtre de sous-titres écrit à la main ; le fichier `.ass` s'importe dans Kdenlive (Projet > Sous-titres > Importer).
- **Retouches** : le fond flouté fait partie de l'image de chaque plan. On ne peut pas le modifier séparément dans Kdenlive.
- **Fidélité** : les transitions et animations de texte de CapCut sont approchées, pas reproduites à l'identique.

## Prochaine étape proposée

- Écrire un essai `kdenlive_builder.py` à côté de `capcut_builder.py`, à partir du même `timeline.json`.
- Rendre un chapitre avec `melt` et l'encodage AMF.
- Comparer le temps de rendu et le résultat avec CapCut.
- Si c'est concluant, garder les deux sorties le temps de la transition.

## Sources

- [Resolve 21.1 retire Python de la version gratuite (xere.my)](https://xere.my/journal/davinci-resolve-21-1-free-python-scripting-removed-lua-benchmark/)
- [Resolve gratuit : encodage matériel limité à NVIDIA (forum Blackmagic)](https://forum.blackmagicdesign.com/viewtopic.php?f=21&t=149274)
- [Resolve Studio : AV1 sur cartes AMD (VideoCardz)](https://videocardz.com/newz/davinci-resolve-studio-beta-gets-av1-encoding-support-for-amd-gpus)
- [Resolve : import OTIO et transitions (VioletFlare)](https://violetflare.ai/blog/davinci-resolve-otio-import/)
- [Kdenlive : rendu et ligne de commande (manuel 26.08)](https://docs.kdenlive.org/en/exporting/render.html)
- [Kdenlive : profil AMF pour Windows (commit KDE)](https://invent.kde.org/multimedia/kdenlive/-/commit/3d68ada7cd57e9c2cdefa93ebbf401cd82848a3c)
- [Kdenlive : transitions (manuel 26.08)](https://docs.kdenlive.org/en/compositing/transitions.html)
- [Shotcut : bug h264_amf (forum)](https://forum.shotcut.org/t/error-exporting-with-h264-amf-codec/51576)
- [Shotcut : export en ligne de commande (forum)](https://forum.shotcut.org/t/export-at-the-command-line/44067)
- [Blender : encodage matériel encore en projet (GSoC 2025)](https://devtalk.blender.org/t/gsoc-2025-proposal-hardware-accelerated-video-encoding-decoding-for-blenders-vse-using-ffmpeg/39768)
- [Blender : performances du VSE en 2026 (Aras)](https://aras-p.info/blog/2026/08/18/More-Blender-VSE-tidbits/)
- [Remotion : licence gratuite jusqu'à 3 personnes](https://www.remotion.dev/docs/license/faq)
- [Remotion : encodage matériel NVIDIA seulement](https://www.remotion.dev/docs/hardware-acceleration)
- [Editly (GitHub)](https://github.com/mifi/editly)
- [CapCut : aucune API officielle (samautomation)](https://samautomation.work/capcut-api/)
- [pyCapCut (GitHub)](https://github.com/GuanYixuan/pyCapCut)
