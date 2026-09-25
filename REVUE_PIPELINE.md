# Revue du pipeline — ce que chaque étape fait vraiment, et quoi corriger

Base factuelle : le run 3 chapitres de *Mirror World God-Tier Revenge* (ep1/ep2/ep3), ses artefacts sur disque (`chapter.json`, `panels.json`, `scenes.json`, `voiceover.json`, `timeline.json`, `draft_content.json`, `preview_60s.mp4`) et `batch_status.json`. Tous les chiffres ci-dessous ont été mesurés sur ces fichiers, pas estimés. Les propositions ont ensuite subi une passe de vérification adverse ; là où elle contredit la revue initiale, c'est la vérification qui tranche et je le signale.

---

## 1. Étape par étape

### 1.1 Scraping et assemblage (`src/modules/scraper.py`, `src/utils/http.py`)

**Ce qu'elle fait.** Une session `requests` unique force `/en/`, récupère la page viewer avec le couple `User-Agent` + `Referer` anti-403, extrait les URLs de chunks depuis `div#_imageList img._images` (`data-url` d'abord), télécharge chaque chunk **séquentiellement en mémoire**, puis empile tout en un strip RGB unique via `np.vstack` + `Image.fromarray`.

**Ce que le run révèle.** Le scraping est **correct à 100 %** : 172/135/113 chunks, indices CDN de 0001 à N sans aucun trou, `len(image_urls) == len(chunk_sizes)` partout. Tous les chunks font 900 px de large, donc la branche de rééchantillonnage LANCZOS n'a **jamais** été exécutée — vérifié sur les 29 `chapter.json` de `output/` : une seule largeur distincte par série (720/800/900). L'assemblage est bit-exact. Le choix `?type=q90` est le bon : PSNR 37–58 dB contre l'original, et les originaux ne sont pas en cache edge (20 chunks = 32,9 s contre 0,5 s).

**Ce qui cloche.**
- Un `time.sleep(0.15)` inconditionnel entre chunks : **25,7 s sur ep1, soit 36,2 % des 70,9 s de `scrape+slice`**, contre ~4,1 s de transfert réel et ~1,2 s de décodage. 62,5 s sur les trois chapitres. Inatteignable depuis `run`/`batch` (le `--delay` n'existe que sur le CLI autonome).
- `Image.fromarray` en fin d'assemblage : **+1383 MB transitoires et 0,30 s**, pour rien — `slice_panels` et `render_debug_overlay` acceptent déjà un `np.ndarray`. Pic mesuré 2230 MB par chapitre (strip réel : 592 MB ; chunks résidents : 575 MB).
- `discover_episodes` s'arrête à `max_pages=60` alors que les pages font 10 épisodes. Sur Tower of God : **541 épisodes sur 653, il manque 2 à 112 et 221**, sans le moindre avertissement. La branche « série entière » de `resolve_chapter_urls` renvoie la liste trouée telle quelle.

### 1.2 Découpe en cases (`src/modules/slicer.py`)

**Ce qu'elle fait.** Variance par ligne → lignes < 15 = gouttière → segments de contenu → fusion des petits segments → padding 15 px → drop sous 180 px → découpe mécanique des segments > 1200 px en 3 morceaux maximum, chaque coupe « calée » sur la ligne la plus calme à ±15 %. Appelée sans un seul argument : `panels = slice_panels(strip)` (pipeline.py:256).

**Ce que le run révèle.** Le rappel des gouttières est excellent : 90/101, 57/73 et 59/63 des zones inter-cases sont parfaitement noires. Mais **77/80/88 % des cases sont des morceaux mécaniques** : 183/136/127 cases issues de seulement 102/72/64 segments. La qualité des coupes est mauvaise et mesurable : sur les 208 coutures, la ligne coupée a une variance médiane de 1964/2518/2312 — **100 %/98 %/100 % des coupes tombent au-dessus du seuil de gouttière du module lui-même**. 26/38/43 % des coutures tranchent une bulle de dialogue en deux (confirmé visuellement : ep1 #113→#114 coupe une bulle en plein texte, #14→#15 coupe une bouche).

La recherche « ±15 % de ligne calme » est un no-op : 0/60, 1/45, 0/49 segments longs contiennent une gouttière au seuil 15. Il n'y a rien de calme à trouver. En revanche **40/60, 32/45 et 46/49 contiennent une discontinuité ligne-à-ligne franche** (z > 25) — les bordures de cases collées bord à bord, omniprésentes dans ce style — et seules 2/5/0 % des coupes actuelles tombent dessus.

Géométrie : **31/32/28 % des cases dépassent 1080 px** et sont donc affichées sous leur taille native (échelle plancher 0,39/0,36/0,37 ; 42/39/40 % du temps d'écran). 16/14/9 segments touchent le plafond de 3 morceaux et gardent des pièces de 2745/3034/2935 px.

Pertes silencieuses : vérification faite en reconstruisant le strip depuis `debug_overlay.png`, **8 encarts de narration perdus** (1 sur ep1, 7 sur ep2, **0 sur ep3** — la revue initiale annonçait 19, c'est 2,4× trop). Hauteurs paddées 86–179 px, tous sous le seuil de 180 px. Dont « No wonder Mirror World is an SSS-Rank hidden skill. » — absent de `ep2/scenes.json` : Gemini ne l'a jamais vu.

**Ce qui cloche.** Les constantes 1200/1500/180 n'ont aucune relation avec le cadre de sortie, et le profil de format n'atteint pas l'étape. La coupe est une décision de montage prise avant l'analyse, et le montage n'utilise ensuite que 55/65/47 % de ce qui a été coupé.

### 1.3 Analyse Gemini (`src/modules/analyzer.py`, `src/utils/gemini_manager.py`)

**Ce qu'elle fait.** Mode deux étapes (déclenché car le payload inline dépasse 16 MiB) : beats par lots de 12 cases avec fenêtre glissante, puis script texte seul (cible `min(1500, panels × 11)` mots), puis sélection des cases-clés sur vignettes 640 px, 4 threads. Tout passe par un `GeminiManager` partagé : limiteur global 5 RPM, cascade monotone de 9 modèles, checkpoint de reprise.

**Ce que le run révèle.** 117 appels, 76 réussis, **41 échoués (35 %)**, dont 27 sur gemini-3.7/3.8-flash avec 0 succès — à eux seuls ils ajoutent 324 s au plancher RPM. Répartition : beats 51 %, script 4 %, **keyframes 45 %**. `rpm_wait_s` 5342,7 s contre 1418,3 s de latence réelle : **le run est à ~73 % contraint par le RPM**, chaque appel supprimé vaut ~12 s de wall clock.

La cascade a dérivé : `gemini-3.5-flash-lite` est abandonné à ok=14 / failed=0 et n'est pas dans `exhausted_models`. Aucun chemin de code ne peut avancer le curseur pour un modèle sans échec — c'est un thread périmé qui a bougé le curseur partagé. Conséquence : **34 des 76 appels réussis (44,7 %) servis par des flash-lite**, et les trois chapitres finissent sur `gemini-3.1-flash-lite`, 7ᵉ des 9.

Couverture : 100/183, 88/136, 60/127 cases retenues. Les cases écartées ne sont pas des transitions — hauteur médiane 887/912/907 px contre 962/940/984 px pour les retenues. ep3 a un trou de **14 cases consécutives** (9,4 % du chapitre) et un autre de 10. ep1 et ep3 laissent tomber les dernières cases du chapitre : le cliffhanger n'apparaît jamais.

Cadence : **7,14/7,72/7,78 s par case-clé**, contre les « ~4,5 s » que le code documente lui-même. Pires cas : 19,5 s sur une seule case (ep1 para24, alors que le beat en offrait une deuxième), 20,8 s (ep2 para3).

Contrôle de longueur : inexistant en mode deux étapes. `SCRIPT_MIN_LENGTH_RATIO` n'est lu que dans `write_recap` (mode un appel, qui n'a pas tourné). Résultat : ep2 → 58 beats → 38 paragraphes → 679 s (5,00 s/case source) ; ep3 → 38 beats → 22 paragraphes → 467 s (3,68 s/case). **36 % d'écart** produit par la granularité d'un seul appel beats.

**Le pire défaut est l'identité des personnages.** Le héros : ep1 « the warrior » ×9 / « the protagonist » ×7 / « Aiden » ×2 ; ep2 « Aiden » ×48 ; ep3 « the protagonist » ×10, « Aiden » ×0, « Eden » ×2. Le garçon à lunettes : quatre appellations différentes en ep1 (dont « The unnamed character with glasses »), « Joey » ×15 en ep2, « Benny » ×11 en ep3 — où il est aussi appelé « Benny's great-grandson ». Et ep2 para37 fait de Benny Cooper l'arrière-grand-père d'Aiden, ep3 para0 celui du garçon. Trois épisodes, trois identités, une contradiction frontale. Le seul contexte inter-chapitre transmis à un prompt est la ligne « Series: … - Episode: … ».

Les garde-fous existants tiennent (0 formule visuelle interdite, 0 intro générique, CTA à 40/39/36 %), mais laissent passer « the scene shifts dramatically », « A flashback unfolds », « System messages display », et les onomatopées lues à voix haute (« a 'WHOOSH' sound », « a sickening 'SPLURT' »). ep2 est largement un dump de stats : 45 % des paragraphes contiennent des chiffres de jeu.

### 1.4 Voix Kokoro (`src/modules/tts_engine.py`)

**Ce qu'elle fait.** Une passe `KPipeline` **par phrase**, jointes par 0,2 s de silence numérique, 0,18 s de padding final, WAV 24 kHz mono, sans aucun étalonnage de niveau. `duration_s`/`speech_s` sont des comptes d'échantillons exacts, et pilotent toute la timeline en aval.

**Ce que le run révèle.** Le débit est bon et stable : médiane 149,0 mots/min, écart-type 11,8 ; les outliers suivent le contenu (les scènes avec chiffres tombent à 134,6). Le niveau est remarquablement constant : RMS voisée sur 0,89 dB d'amplitude sur 95 scènes — une normalisation par scène serait nuisible.

Mais **22,3–22,9 % de l'audio est du silence structurel** (163,4/151,6/104,6 s ; la revue initiale annonçait 17–18 %, c'était une sous-estimation) : 0,23 s d'amorce sur chaque scène, **1,02 s de queue sur chaque scène contre 0,18 s configurés**, et 175 pauses inter-phrases de **1,22 s médiane contre 0,2 s configurés**. Le réglage `sentence_gap_s` et le commentaire du module qui explique que 0,4 s « donnait une narration traînante » mesurent donc quelque chose qui n'atteint jamais l'auditeur. Débit délivré : 146,2 mots/min contre 188,8 sur l'audio voisé seul.

Les sous-titres héritent du problème : `build_subtitle_cues` répartit au prorata des caractères sur `speech_s`, aveugle aux pauses. Erreur médiane 0,24 s, p90 0,64 s, max 1,58 s, et **100 % des scènes démarrent leur premier bloc ~0,23 s avant la voix**.

Dictionnaire de prononciation : les 13 clés de `config/pronunciations.json` apparaissent **zéro fois** dans les 27 795 caractères de ce run — ce sont des restes d'une série précédente. Couverture 0/95 alors qu'« Aiden » est prononcé 29 fois. Pire, trois entrées dégradent activement : « Destia » seul donne `dˈɛstiə`, le dictionnaire impose `dˈɛstˈiˌɑ`. Le format `/IPA/` que le code supporte est exact et utilisé par 0 des 13 entrées.

Niveau : **−22,0 LUFS** sur les trois `voiceover_full.wav` et sur les trois `preview_60s.mp4`, soit ~8 LU sous la référence YouTube qui n'atténue que et ne remonte jamais.

### 1.5 Timeline et rythme (`timeline_builder.py`, `pacing.py`, `framing.py`)

**Ce qu'elle fait.** La piste image est esclave de la voix : chaque scène occupe exactement sa `duration_s` TTS. `LongPacing` donne 2,5 s de base à chaque case-clé puis répartit le reste au prorata de `max(hauteur, 400)`. Une case-clé = un plan. Décoration ensuite : transitions, VFX, SFX, BGM par fusion des humeurs consécutives.

**Ce que le run révèle.** **Le montage est structurellement trop lent.** 248 plans pour 1859,9 s = une coupe toutes les 7,14/7,72/7,87 s, soit 7,7–8,4 coupes/min. p90 à 10,3/11,4/10,7 s, max 19,45/20,78/13,89 s. **Les plans de plus de 8 s occupent 34,8/56,8/59,7 % de la durée.** ep1 et ep2 contiennent chacun trois scènes rendues comme une seule image fixe tenue 16–21 s.

Le plancher de 2,5 s et toute la machinerie « trop de cases pour le temps disponible » sont **inertes** : aucun des 248 plans n'approche 2,5 s (minimum global 3,70 s) et `limit_panels_for_duration` n'a supprimé **0 case sur 95 scènes**. L'étape a un plancher et aucun plafond : `max_clip_s` vaut `None` pour LONG.

Et `expand_panel_ids` — la fonction qui récupère toutes les cases contiguës — existe déjà mais est verrouillée derrière `if profile.pacing.max_clip_s is not None`, donc SHORT uniquement. Simulée sur les vraies scènes elle récupère 182/183, 136/136, 127/127 cases et donnerait une médiane de 3,57/4,90/3,22 s, soit **15,3/12,0/16,3 coupes/min, à coût Gemini nul**.

Le sous-titre monte 5× plus vite que l'image : un bloc toutes les 1,45–1,59 s contre une coupe toutes les 7,1–7,8 s. 80 % des blocs reçoivent l'animation d'entrée maximale de 0,40 s → 25–27 % de la durée avec du texte en mouvement ; 43–60 % portent en plus une boucle permanente de tremblement. `SubtitleRules.min_words=2` n'est lu par personne : blocs de 3 ou 4 mots exclusivement (moyenne 3,65).

Le Ken Burns est imperceptible sur les longues tenues : amplitude fixe de 5 % étalée sur la durée du plan → 0,78 %/s en médiane, 0,24 %/s sur la tenue de 20,8 s. Mesuré en croissance de largeur affichée : médiane 6,3 px/s, et 22 % des plans ken_burns sous 4 px/s.

Les VFX couvrent **37,8/38,2/40,3 % de la vidéo** avec des overlays dont la durée est celle de la scène entière (médiane 19,8/16,0/20,7 s, max 28,5 s). Et contrairement à ce que supposait la revue, **c'est visible dans le MP4 livré** : ep2 et ep3 ouvrent sur 24,7 s de pluie ; à t=10 s toute l'image, case et fond flou compris, est striée de blanc.

La BGM redémarre à l'échantillon 0 à chaque segment (pas de point d'entrée dans le modèle) sur des assets de 24 s : **36/36/25 redémarrages audibles par chapitre**, un toutes les ~20 s, et l'humeur bascule toutes les 42/34/33 s sans hystérésis.

Dérive des transitions : 6,90/7,20/2,40 s sur 23/24/8 transitions marquées `is_overlap`.

Cadrage : les cases font 900 px de large, donc l'image occupe 34–40 % du cadre 1920×1080 et 39–42 % du temps montre une case **sous** sa résolution native. Pire, la durée est pondérée par la hauteur **brute**, donc les cases les moins lisibles reçoivent le plus de temps : une case 900×2745 affichée à 0,39× tient l'écran 15,7 s.

### 1.6 Rendu preview et brouillon CapCut (`capcut_builder.py`, `preview_renderer.py`)

**Ce qu'elle fait.** Deux émetteurs indépendants sur la même `Timeline`. CapCut : 8 pistes, fond pré-rendu en JPEG par case, case brute à l'échelle native, keyframes de zoom, sous-titres en `TextSegment`. Preview : réimplémentation en OpenCV/NumPy, encodée en x264 crf 20, **60 premières secondes seulement**.

**Ce que le run révèle.** La géométrie des cases est **vérifiée exacte**. Contrairement à la mémoire projet, **CapCut a bien ouvert et réenregistré des brouillons générés** : trois brouillons portent `app_version 9.4.0` / `new_version 185.0.0` et un `draft_cover.jpg`. CapCut a tout préservé : durée à ±0,015 s près (quantification 60 fps), zéro trou introduit, keyframes et `transform_y` identiques au bit, tous les chemins médias conservés. En comparant la couverture CapCut à l'instant re-rendu par le preview : la case occupe les colonnes 561–1357 côté CapCut, 560–1357 côté preview — **797 contre 798 px, un pixel d'écart**.

Les sous-titres, non. Même image, même bloc : CapCut le dessine 1289 px de large, encre sur 61 px, contour 4 px ; le preview 1007 px, 44 px, contour 2 px. Soit **84,5 px d'em côté CapCut contre 63,5 px côté preview (+33 %)**, dans une police différente. La calibration de l'unité `size` de CapCut, établie sur deux brouillons d'orientations opposées : `px = size × largeur_canvas / 204`, à 0,4 % près. Le commentaire « 20 ≈ 2 px en 1080p » est faux d'un facteur 2 exact.

Le preview regarde le mauvais 8 %. Il couvre 8,4/8,8/12,9 % du chapitre. Pour ep1 : 8 plans sur 100, 3 scènes sur 35, **0 SFX sur 22, 0 VFX sur 13**, 1 transition sur 23. ep2 : 0/28 SFX. ep3 : 0/12 SFX. La totalité du sound design du pipeline est invisible dans l'artefact que l'utilisateur juge.

Coût : 32,55 s pour re-rendre les 60 s d'ep1 (17,84 s de composition + 14,71 s d'encodage). À 30 fps : 14,18 s pour le même résultat visuel. `draw_subtitle` coûte 4,41 ms/frame, **deux fois le coût de la case**, parce que le RGBA du sous-titre est ré-échelonné à chaque frame.

Durabilité : le brouillon référence 241/220/148 fichiers par chemin absolu dans `output/<projet>/`, et `copy_draft` ne copie que les deux JSON. Nettoyer `output/` casse silencieusement tous les projets CapCut pendant que les MP4 continuent de jouer.

### 1.7 Orchestration batch et reprise (`batch_processor.py`, `main.py`)

**Ce qu'elle fait.** Une coroutine par chapitre, quatre sémaphores (scrape / analyze 5 / tts 2 / render 1) et un limiteur RPM global partagé. Le ledger `batch_status.json` est réécrit atomiquement à chaque changement de champ. Reprise pilotée par le statut au niveau batch, et par les artefacts sur disque au niveau étape.

**Ce que le run révèle.** La phase Gemini est **ordonnancée quasi parfaitement** : fenêtre de 1402,4 s contre un plancher théorique de 1344 s pour 117 appels à 5 RPM, soit ~96 % du plafond. Le problème n'est pas l'ordonnancement, c'est le nombre d'appels.

Le vrai défaut est la queue. Le run finit à +1916 s, soit **470 s (24,5 %) après le dernier appel Gemini**, limiteur totalement inactif, pendant que 959,4 s de CPU local s'écoulent à travers `sem_tts=2` et `sem_render=1`. Les trois chapitres ayant été admis en analyse simultanément, ils ont fini à 74 s d'intervalle et tout le CPU s'est empilé à la fin. Cette attente est **non instrumentée** : `_timed` démarre le chronomètre à l'intérieur du thread, après l'acquisition du sémaphore. Récupérée par soustraction : 7,0/37,0/**261,3 s** — 13,6 % du wall clock d'ep3 invisible.

Le résumé affiche **−1824 s de calcul local**, reproduit exactement : `sum(timings)` 5208,2 s moins `sum(gemini_s)` 7032,6 s. `gemini_seconds` est `analyzer.api_seconds`, accumulé à travers les 4 threads keyframes **et incluant le sommeil du limiteur**. ep1 déclare 2430 s de Gemini dans une étape d'analyse de 1302 s. Le même mensonge apparaît dans le log par chapitre : « 1302s dont 2430s chez Gemini ».

`stats` affiche « 279s par chapitre, 0,50s de calcul par seconde de vidéo » alors que le bloc `run` du même fichier dit 1763,1 s et 2,64 — **erreur d'un facteur 5**, parce que `timing_summary` filtre les étapes sous 0,05 s et que les chapitres réutilisés tirent les médianes vers le bas. `share_pct` est écrit à 0,0 cinq fois sans jamais être calculé.

`stats` annonce « 6 échecs » là où **zéro chapitre est réellement cassé** : cinq sont des court-circuits de quota d'un autre chapitre (ils ont même leurs cases déjà scrapées sur disque, 429 MB) et le sixième est bloqué sur un checkpoint de 4 lots de beats complétés depuis le 10/09, que rien ne signale.

Ce qui marche très bien : la reprise au niveau artefact. Les cinq chapitres quota-bloqués ont gardé leurs cases, donc une reprise coûte 0 appel de scraping.

---

## 2. Plan de travail classé

Ordre = rapport valeur/effort, ce qui débloque le plus en tête. Effort = celui retenu après vérification, pas celui annoncé.

### Rang 1 — Débloquer la mesure et le budget

**1. Corriger `gemini_seconds` et exposer le nombre d'appels par chapitre**
*Problème :* le rapport affiche −1824 s de calcul local parce qu'on soustrait une somme multi-thread incluant le sommeil du limiteur, et rien dans le ledger ne dit quel chapitre a coûté combien des 117 appels.
*Correctif :* supprimer la soustraction (`batch_processor.py:516-518`), relabeliser la valeur survivante « cumulé multi-thread », corriger le même log trompeur à `pipeline.py:320-324`, ajouter `gemini_calls` (depuis `analyzer.n_calls`, déjà compté) à `PipelineResult` et aux deux `status.update`, et chronométrer les quatre acquisitions de sémaphore séparément (`queue_s`). **Ne pas** attribuer le sommeil du limiteur par chapitre : avec un limiteur unique partagé, l'attribution est arbitraire, et `acquire()` retourne déjà les secondes attendues si on en veut le total. **Ne pas** ajouter les offsets Gantt : rien ne les consomme.
*Pourquoi en tête :* `gemini_calls` par chapitre est la seule métrique qui permet de dimensionner `--batch-size` contre la taille du chapitre, c'est-à-dire le levier principal sur un budget de 20–25 appels/jour. Et `queue_s` est la seule façon de valider ou d'infirmer la proposition d'admission étalée (rang 4).
*Effort :* **S** (version réduite). *Fichiers :* `src/pipeline.py`, `src/modules/batch_processor.py`, `tests/test_batch.py`.

**2. `--max-gemini-rpm` par défaut à 5**
*Problème :* le défaut est 10 à trois endroits (`batch_processor.py:74`, `main.py:163`, `gemini_manager.py:76`) alors que `CLAUDE.md:48` dit « toujours 5, 10 provoque un backoff 429 permanent ». Chaque run lancé sans le flag pousse le limiteur dans des 429 à 30 s de refroidissement, puis casse la cascade après 2 tours.
*Correctif :* défaut à 5 partout, ou mieux une variable d'environnement à côté de la clé, et corriger l'exemple `PRD.md:403`. Logger le RPM effectif au démarrage.
*Effort :* **S**. *Fichiers :* les trois ci-dessus, `PRD.md`.

### Rang 2 — Défauts visibles à l'écran, correctifs de quelques lignes

**3. Plafonner la durée des VFX**
*Problème :* `duration = end - start` donne des overlays de 20 s couvrant 38–40 % de la vidéo, et c'est observable dans le `preview_60s.mp4` livré (ep2/ep3 ouvrent sur 24,7 s de pluie plein cadre).
*Correctif :* `duration = min(end - start, MAX_VFX_S)` avec une constante à côté de `MIN_VFX_SCENE_S`. Couverture 38–40 % → 3,6–4,4 %. **Ne pas** ancrer sur le plan `punch_in` (disponible dans seulement 17 scènes VFX sur 37, et ça change la signature) ; **ne pas** ajouter de plafond par minute (densité déjà à 1,1–1,3/min, et après le clamp il n'y a plus rien à plafonner). Si on dépense plus de deux lignes, faire le plafond **par type** : `rain` et `speed_lines` sont des overlays plein cadre agressifs, `glow` est une vignette respirante à 0,35 de force qui se tient très bien sur la durée.
*Effort :* **S**. *Fichiers :* `src/modules/timeline_builder.py` (constante ~:147, expression :443), `tests/test_timeline.py:377`.

**4. Supprimer le `sleep(0.15)` du téléchargement**
*Problème :* 25,7 s de sommeil pur sur ep1, 36,2 % de `scrape+slice`, injoignable depuis `run`/`batch`.
*Correctif :* `delay: float = 0.0` (ou 0.02) aux deux endroits, plus un `scrape_delay_s` dans `PipelineOptions` passé à `pipeline.py:255` pour pouvoir remonter la politesse si le CDN se plaint. **Ne pas** ajouter de `ThreadPoolExecutor` : mesuré proprement, le parallélisme n'apporte que ~1,4 s des ~27 s disponibles (le « 8,4× » comparait séquentiel-avec-sleep à parallèle-sans-sleep), `max_scrape_workers` existe déjà au niveau chapitre et valait 3, et 3 chapitres × 4 threads = 12 connexions sur un seul hôte. **Ne pas** ajouter 403 aux statuts réessayables : c'est le garde-fou UA/Referer, un 403 est une erreur de configuration, pas un transitoire.
*Effort :* **XS**. *Fichiers :* `src/modules/scraper.py` (2 lignes), `src/pipeline.py`.
*Portée honnête :* ~25 s/chapitre, invisible en batch (le chemin critique est le limiteur RPM saturé pendant 1916 s), visible sur un `run` mono-chapitre.

**5. Activer le rognage des silences sur le profil LONG**
*Problème :* `strip_silence` est écrit et documenté mais `max_internal_silence_s=None` pour LONG, donc il n'a jamais tourné ; 22,3–22,9 % de chaque chapitre est du silence structurel, avec des queues de 1,02 s là où 0,18 s sont configurés.
*Correctif :* `max_internal_silence_s=0.45` dans `_long_profile()` + un flag CLI sur les **deux** commandes. Gains re-mesurés sur les 95 WAV : **−10,2/−9,4/−10,4 %** de durée pour ~0,15 s de CPU par chapitre. Trois corrections à l'énoncé initial : la fonction n'est **pas** testée (aucun test n'existe, il faut en écrire un) ; l'amorce n'est **pas** rognée à 0,45 s (elle mesure 0,225–0,230 s) ; et `padding_s` ne devient **pas** l'écart réel entre scènes, car le rognage précède l'ajout du padding — la queue devient 0,45+0,18=0,63 s. Si l'objectif est vraiment « padding_s = l'écart », il faut rogner tête/queue à ~0 et ne plafonner que les runs internes, deux lignes de plus. Et `--max-silence 0` est un piège (ça donne 1 échantillon, agressivité maximale) : prévoir `--no-strip-silence`.
*Bénéfice secondaire non anticipé :* `build_subtitle_cues` répartit au prorata linéaire sur `speech_s`, qui contient aujourd'hui 0,84 s de silence de queue et des pauses de 1,2 s. Retirer le silence rend ce modèle matériellement plus juste — la synchro des sous-titres s'améliore gratuitement.
*Risque vérifié comme nul :* `limit_panels_for_duration` ne supprime aucune case aujourd'hui, et ne le fait toujours pas après rognage (100/88/60 cases conservées dans les deux cas).
*Effort :* **S**, mais 5 fichiers. *Fichiers :* `format_factory.py:92`, `pipeline.py` (champ + override :353), `main.py` (deux commandes), `tests/test_format.py`, nouveau test de `strip_silence`.
*Avant de committer :* auditionner ep3 (22 scènes, resynthèse la moins chère) à 0,45 et 0,6, avec `--force` ou `--redo tts` car le cache manifeste bloque sinon.

### Rang 3 — Servir la règle de résolution native

**6. Caler la hauteur de découpe sur le cadre de sortie**
*Problème :* `SPLIT_MAX_HEIGHT=1200` n'a aucune relation avec le cadre, donc 31/32/28 % des cases sont affichées sous leur taille native, jusqu'à 0,36×.
*Correctif :* une ligne — `slice_panels(strip, split_max_height=options.profile().framing.height, split_max_pieces=8)` à `pipeline.py:256` (`options` est déjà dans la portée). **La formule proposée était fausse** : `native_scale()` plafonne à 1.0 et ne sur-échantillonne jamais en LONG, `max_upscale=1.05` n'est consommé que par le fenêtrage SHORT. Le point d'équilibre pour une colonne de 900 px est donc exactement **1080**, pas 1134 (à 1134 px on rend déjà à 0,952×). Plafond de morceaux à 8 plutôt que supprimé : 8×1080 = 8640 px couvre le plus gros segment observé (8152 px). **Ne pas** dériver la valeur SHORT de `max_upscale` (ça donnerait 6144 px, soit découpe désactivée — changement majeur non validé) : laisser SHORT à 1200 ou le fixer explicitement après un rendu de contrôle. **Ne pas** ajouter 6 flags CLI sur une commande qui en a déjà 109.
*Coût quota et sa mitigation :* +3/+3/+2 lots de beats par chapitre sur 32/23/21 appels. Mais passer `batch_size` de 12 à 15 (le plafond `MAX_BATCH_SIZE` existe) donne `ceil(224/15) = 15` lots, soit **moins** d'appels qu'aujourd'hui (16). Faire les deux ensemble.
*Effort :* **S** (version réduite ; la version complète avec le profil threadé et 6 options est M). *Fichiers :* `slicer.py:73,75`, `pipeline.py:256`, `tests/test_slicer.py:282`.
*Piège d'ordonnancement :* `--force` re-découpe alors que `stage_analyze` réutilise `scenes.json` — les indices de cases ne voudraient plus dire la même chose. Un changement de hauteur de découpe doit forcer un redo de l'analyse, ou au minimum avertir.

**7. Récupérer les encarts de narration perdus**
*Problème :* 8 encarts de narration (1 sur ep1, 7 sur ep2, 0 sur ep3) sont supprimés sous le seuil de 180 px, dont « No wonder Mirror World is an SSS-Rank hidden skill. » qui n'apparaît nulle part dans `ep2/scenes.json`.
*Correctif :* `MIN_PANEL_HEIGHT` de 180 à **80** — les 8 encarts mesurent 86–179 px paddés, donc la constante seule les récupère tous les 8. **Ne pas** écrire le détecteur de structure de texte proposé : il n'apporte rien sur ce corpus et ajoute un réglage à régler. Corriger au passage le garde `block_is_small` de `merge_small_segments`, qui empêche un encart isolé de s'attacher à un voisin normal même à 112 px de distance. Gérer le risque montage là où il est : c'est `expand_panel_ids` en SHORT qui ferait entrer une bande de texte de 100 px dans le montage ; LONG n'affiche que les cases choisies par Gemini, et le prompt lui dit déjà d'ignorer les cases purement textuelles.
*Garde-fou obligatoire :* mesurer `payload_bytes` avant/après. ep3 est à 16,81 MiB contre une limite de 16,0 MiB — il est déjà en mode deux étapes, donc indifférent ; mais un chapitre qui tient aujourd'hui **sous** la limite en un seul appel peut basculer en mode ~40 appels, ce qui est une régression massive sur le budget free tier. Juger sur les octets, pas sur le nombre de cases.
*Effort :* **M** (les tests `tests/test_slicer.py:367-374` et `:419-421` figent le comportement actuel). *Fichiers :* `slicer.py:63,288,386-392`, `pipeline.py`, `main.py`, `tests/test_slicer.py`.

**8. Ramper le Ken Burns dans le temps, pas dans l'amplitude**
*Problème :* amplitude fixe de 5 % étalée sur la durée du plan → 0,24 %/s sur une tenue de 20 s.
*Correctif :* **la formule proposée est un no-op** — `min(0.012 × durée, 0.05)` sature dès 4,17 s, et seuls 5 plans sur 248 sont plus courts (elle *baisserait* leur zoom). Ce qu'il faut changer est la cartographie temporelle, pas l'amplitude : `factor = 1.0 + zoom * ease_in_out(min(1.0, elapsed_s / KEN_BURNS_RAMP_S))` avec `RAMP_S = 4.0`. `elapsed_s` est déjà un paramètre de `zoom_factor` et déjà passé. Côté CapCut, trois keyframes au lieu de deux, sur le modèle exact du bloc `punch_in` situé huit lignes plus haut. Supprimer `motion_for` (mort, et l'appeler depuis `pacing.py` créerait un import circulaire).
*Risque nouveau, à regarder avant de committer :* le grossissement moyen sur un plan passe de 2,50 % à ~3,4 %, et un plan de 19 s resterait à **105 % pendant ~15 s** au lieu de l'effleurer en fin de plan. La règle des 105 % a déjà été revue une fois après avoir vu de la pixellisation ; le levier de correction est une rampe plus longue, pas un plafond plus haut.
*À faire après le rang 4 sur le rythme*, et calibrer `RAMP_S` sur la nouvelle distribution de durées.
*Effort :* **S**. *Fichiers :* `timeline_builder.py`, `preview_renderer.py:139`, `capcut_builder.py:381-382`, `tests/test_preview.py:93`.

### Rang 4 — Correctness et honnêteté du ledger

**9. Empêcher `--redo` de relancer une analyse Gemini complète**
*Problème :* `stage_analyze` retombe sur l'analyseur complet quand `scenes.json` est absent, alors que l'aide CLI promet « aucun quota Gemini pour `--redo tts` ».
*Correctif :* quatre lignes dans `stage_analyze` plutôt que dans `process_batch` — lever une `FileNotFoundError` explicite quand `options.redo` cible une étape au-dessus d'`analyze` et que `scenes.json` manque. Le `except` par chapitre du batch l'attrape déjà, l'écrit dans le ledger et continue. Ça couvre aussi les appels programmatiques à `run_pipeline`. **Ne pas** exiger `audio/voiceover.json` pour `--redo montage` : le TTS ne coûte pas de quota, un chapitre avec `scenes.json` et sans voix doit être **complété**, pas sauté. Corriger aussi l'aide `main.py:198`.
*Recadrage nécessaire :* le mécanisme décrit était faux et l'ampleur surestimée d'un facteur ~40. Les cinq chapitres cités sont `failed`, et `retry_failed=True` les admet **avant** que `force` soit consulté — la même commande sans `--redo` dépense exactement le même quota. Et leurs payloads mesurent 10,3–13,8 MiB, donc chemin un-seul-appel : **~5 appels au total, pas ~195**. C'est un nettoyage de correctness, pas un sauvetage de quota. Ne planifiez pas sur le chiffre de 195.
*Effort :* **S**. *Fichiers :* `src/pipeline.py:287`, `src/main.py:198`, `tests/test_batch.py`.

**10. Distinguer les chapitres bloqués par le quota des vrais échecs**
*Problème :* `stats` annonce 6 échecs là où **aucun chapitre n'est cassé**, et `--no-retry-failed` — que quelqu'un passerait raisonnablement pour éviter de re-brûler du quota — abandonnerait définitivement les cinq chapitres bloqués et celui qui dort sur un checkpoint de 4 lots.
*Correctif :* déclencher sur `isinstance(exc, QuotaExhaustedError)` **aux deux sites**, pas seulement sur le court-circuit — le « vrai » échec est le plus reprenable des six. Option la moins invasive : ne pas ajouter de valeur à l'enum persisté, écrire `STATUS_PENDING` sans incrémenter `attempts` plus un champ `quota_blocked: true` ; `should_process`, `counts()` et `format_status` fonctionnent déjà, et une ancienne version relit le fichier correctement. Une ligne dans la table de marques donne le libellé « QUOTA ». **Ne pas** annuler les tâches en attente : le gain mesuré est de quelques secondes (ce sont des coroutines parquées sur un sémaphore, sans CPU ni réseau), c'est l'édition la plus risquée du lot, et elle **détruirait de la valeur** — un chapitre encore en queue sur `sem_scrape` scrape aujourd'hui quand même et pré-cache ses cases, ce qui a économisé 32–46 s × 5 dans le run observé.
*À ne pas oublier :* `report.errors` pilote le code de sortie (`main.py:250-251`) et `status.data["run"]["failed"]` — un run entièrement quota-bloqué sort aujourd'hui en 1.
*Détail séparé et indépendant :* afficher la présence d'`analysis_checkpoint.json` et son nombre de lots dans `format_status` (~6 lignes). À livrer seul.
*Effort :* **M**. *Fichiers :* `batch_processor.py` (:60, :136, :147-162, :349, :386-396, :483, :501/545), `tests/test_batch.py:84,217,236`, `CLAUDE.md:60`.

**11. Combler les trous de la découverte d'épisodes**
*Problème :* `max_pages=60` × 10 épisodes/page renvoie 541 épisodes sur 653 sans avertissement, et la branche « série entière » livre la liste trouée telle quelle.
*Correctif :* **les deux tiers de la proposition initiale sont faux.** (a) `div.paginate` / « Next Page » ne veut pas dire « page suivante » mais « groupe de dix pages suivant » : la règle s'arrête page 71 et laisse encore 12 épisodes manquants. Et elle est redondante — le `if not new: break` existant termine correctement et sans dépendre du markup (mesuré : il se déclenche page 74 sur la plus longue série du site). Il suffit de passer `max_pages` à 500, en vrai garde-fou. (b) `raise ScraperError` est **nuisible** : `resolve_chapter_urls` enveloppe la découverte dans un `except Exception` et retombe sur une branche où une URL de liste donne `lo=hi=1`, donc le batch devient **un seul épisode**. Logger, ne pas lever. (c) La seule vraie correction : dans la branche non bornée, calculer les manquants et les dériver avec `episode_url` — c'est littéralement la même expression que la branche bornée utilise déjà deux lignes plus bas. Vérifié : l'URL dérivée redirige correctement et `slug_from_url` donne le même `out_dir` que l'URL canonique, donc pas de fork de dossier ni de quota gaspillé.
*Cadrage de l'impact :* personne en free tier ne rendra 541 chapitres, donc « 112 manquants » n'est pas le dommage. Le dommage est **l'ordre** : avec `max_chapters=5`, le premier — et peut-être seul — jour d'un run « série entière » produit les épisodes 1, 113, 114, 115, 116. Une journée entière de quota sur les mauvais chapitres.
*Effort :* **S**. *Fichiers :* `scraper.py:203,216-223`, `batch_processor.py:219-220`, `tests/`.

### Rang 5 — Fidélité CapCut, à faire ensemble

**12. Dériver le style de sous-titre CapCut du profil**
*Problème :* CapCut dessine les sous-titres à 84,5 px d'em et 4 px de contour là où le preview dessine 63,5 px et 2 px, dans une police différente, avec trois constantes qui ne dérivent de rien.
*Correctif :* pas de nouveau paramètre — `Timeline` porte déjà `format` (« LONG » dans les trois `timeline.json`), il suffit de refaire dans `build_capcut_draft` ce que `PreviewRenderer` fait déjà en une ligne. Ça évite de toucher les 3 sites d'appel, dont `main.py:371` qui n'a aucun `PipelineOptions`, et les 6 appels dans les tests. Calibration : `size = font_size_px × 204 / largeur` (6,8 en 1920). **La formule de contour proposée est fausse** — mesuré sur deux brouillons, `stroke_px = (border/100) × 0,2367 × em_px`, donc 2 px à 64 px de fonte donne **~13**, pas les « ~7,3 » annoncés (et l'expression écrite dans la proposition donne en fait 73,4, incohérente avec sa propre réponse).
*Arbitrage :* **baisser CapCut, ne pas monter le preview.** La contrainte dure dit contour 2 px ; c'est CapCut qui est à 4 px, donc c'est CapCut qui viole la règle aujourd'hui. Et monter le preview invaliderait tous les `preview_*.mp4` déjà jugés. La liste de polices ne peut pas être littéralement partagée (le catalogue CapCut n'a pas de Montserrat Bold) : mettre Rubik-Bold en tête de `LONG_FONTS` pour que les deux côtés atterrissent sur la même fonte.
*Le vrai bug livré est SHORT, pas LONG.* En LONG rien ne casse : sur les 1234 blocs des trois chapitres, le plus large fait 1861 px sur 1920, zéro débordement. En SHORT, `size 9.0` sur un canvas de 1080 donne un em de 47,6 px là où le profil demande 92 px — **un facteur 2** — plus un `transform_y` de −0,78 au lieu de l'ancrage 0,74. C'est ça qui justifie le travail.
*Effort :* **S** en code, **M** avec la vérification (il faut rouvrir CapCut et remesurer une nouvelle couverture). *Fichiers :* `capcut_builder.py:52-58,270,450-459`, `tests/test_capcut.py`.

**13. Étaler le preview de 60 s sur tout le chapitre**
*Problème :* le preview couvre 8,4/8,8/12,9 % du chapitre et contient **0 SFX sur 22/28/12** et 0 ou 1 VFX — l'artefact traité comme vérité terrain ne contient rien du sound design ni de la dynamique.
*Correctif :* remplacer `max_duration_s` par une liste de fenêtres `(start, durée)`, choisies par couverture d'événements (ouverture, ~4 s autour de chaque transition, chaque début de VFX, chaque SFX, chaque changement d'humeur BGM, chaque couture de boucle), séparées par 0,5 s de noir timecodé. Budget disponible : à 30 fps le même temps mur achète ~120 s de matière (14,18 s pour 60 s au lieu de 32,55 s), et 30 fps + crf 26 coûte 14,4 s pour 4,8 MB. Exposer `--preview-fps` et `--preview-crf`, aujourd'hui en dur, et garder `--preview-mode head|sampled|full`.
*Effort :* **M**. *Fichiers :* `preview_renderer.py`, `pipeline.py:428-436`, `main.py`.

**14. Clé de cache des fonds incluant la taille**
*Problème :* `bg_{index:03d}.jpg` n'encode pas `width×height`. Un `--format SHORT --redo montage` sur un projet LONG existant réutilise silencieusement des fonds 1920×1080 dans un brouillon 1080×1920, et le preview ne le montre pas (il recalcule le fond en mémoire à la bonne taille).
*Correctif :* `bg_{index:03d}_{w}x{h}.jpg`, et honorer `framing.background_blur` que SHORT met à `False` alors que la piste V1 est toujours écrite.
*Effort :* **S**. *Fichiers :* `capcut_builder.py:158-181,352`.

### Rang 6 — Coûteux mais structurant

**15. Cache de synthèse par scène**
*Problème :* changer une ligne de narration ou une entrée de prononciation coûte 325/303/173 s de resynthèse complète.
*Correctif :* **tel que spécifié, le cache ne peut jamais se déclencher** — sans `--redo tts`, `stage_tts` court-circuite sur l'existence de `voiceover.json` et `synthesize_analysis` n'est jamais appelé ; avec, la proposition dit qu'on ignore le cache. Il faut inverser le garde `pipeline.py:337` : `--redo tts` appelle la synthèse **avec** cache actif, et `--force` (ou un `--resynth-all` explicite) devient l'override dur. Et il faut ajouter `--redo` à la commande `run`, qui ne l'a pas — aujourd'hui, itérer sur l'audio d'un seul chapitre n'est possible qu'avec `--force`, **qui relance l'analyse et dépense ~40 appels Gemini**. C'est le vrai argument pour cette tâche, et la proposition initiale ne le fait pas.
Abandonner le sha1 : `voiceover.json` stocke déjà le texte prononcé post-substitution et tous les paramètres. Matcher sur le texte, pas sur l'index (renumérotation gratuite, même propriété que le digest), et les trois manifestes déjà sur disque en bénéficient dès le prochain run — un nouveau champ garantit au contraire un miss total la première fois. Ajouter `max_silence_s` et `silence_threshold_db` au manifeste, qui ne les enregistre pas alors qu'ils pilotent `strip_silence`. Prendre `speech_s` depuis l'ancienne entrée, pas le recalculer : il pilote le découpage des blocs de sous-titres.
*Portée honnête :* seuls les changements de texte et de prononciation en profitent (mesuré : ajouter « Wright » touche 5 scènes sur 35 → ~45 s au lieu de 325 s, 7×). Changer voix/vitesse/padding/silence invalide tout.
*Bonus :* ferme au passage une collision réelle — `out_dir` ne porte pas le format, donc un run SHORT après un LONG réutilise silencieusement l'audio LONG.
*Effort :* **M**. *Fichiers :* `tts_engine.py`, `models/audio.py`, `pipeline.py:337,366`, `main.py`, `tests/test_tts.py`.

### Non vérifié adversarialement, mais probablement au-dessus de tout ce qui précède

Ces quatre propositions viennent des revues d'étape et n'ont **pas** subi la passe de vérification. Elles sont listées parce que les laisser de côté donnerait une fausse image des priorités — mais elles doivent passer le même contrôle avant d'être planifiées.

- **Bible de série inter-chapitres** (`analyzer.py`, M). Le héros a trois identités en trois épisodes, le comparse quatre appellations dans un seul, et deux épisodes consécutifs se contredisent sur qui est l'arrière-petit-fils de Benny Cooper. Écrire un `series_state.json` (roster canonique, noms propres, deux derniers paragraphes) et l'injecter dans les prompts beats et script. **Zéro appel Gemini supplémentaire** — ce sont des ajouts texte à des prompts existants. Pour une chaîne de recap regardée dans l'ordre, c'est le défaut le plus coûteux du pipeline.
- **Plafonner `max_clip_s` en LONG et activer `expand_panel_ids`** (`format_factory.py:83`, `timeline_builder.py:667`, M). Une coupe toutes les 7,5 s, 35–60 % de la durée sur des plans de plus de 8 s, et 44 % des cases découpées jamais montrées. La fonction de récupération existe déjà et est verrouillée sur SHORT. Simulé : médiane 3,57/4,90/3,22 s, 15,3/12,0/16,3 coupes/min, coût Gemini nul. Changer aussi la pondération `max(hauteur, 400)` en `max(min(hauteur, frame_height), 400)` : pondérer par la hauteur brute donne le plus de temps aux cases les moins lisibles. **C'est le complément indispensable du rang 6** : découper à 1080 px ne met pas plus de contenu à l'écran si le montage n'en montre que la moitié.
- **Corriger la dérive du curseur de cascade** (`gemini_manager.py:377-385`, M). `_advance_model` incrémente sans vérifier que le modèle courant est bien celui qui a échoué, depuis jusqu'à 7 threads. Résultat prouvé par le ledger : 44,7 % des appels réussis servis par des flash-lite, un modèle abandonné à 14 succès / 0 échec, et un curseur monotone qui ne remonte jamais. Passer le modèle défaillant en argument, remplacer le curseur unique par un état par modèle (cooldown + exhausted).
- **Réparer les redémarrages de musique** (`models/timeline.py:187`, M). `BgmClip` n'a pas de point d'entrée source, donc chaque segment rejoue le fichier de 24 s depuis l'échantillon 0 : 36/36/25 redémarrages audibles par chapitre, un toutes les ~20 s, sur une piste présente 100 % du temps. Ajouter `source_offset_s`, une tête de lecture par humeur, une hystérésis d'ouverture de segment (~45–60 s, ce qui ferait passer ep1 de 18 à ~8 segments) et un fondu de 0,5 s au point de bouclage. Le gain `-22 dB` et la règle « musique toujours présente » sont inchangés.

---

## 3. Écarté après vérification

- **Valider le brouillon CapCut contre la timeline à la fin de l'étape.** Les 8 contrôles proposés passent **tous, sur les trois chapitres, avec zéro violation** — et ils ne peuvent pas échouer : l'absence de trou est arithmétique (`contiguous_ranges` cale chaque fin sur le début suivant), le plafond 1,05 est une constante de module, et l'existence des médias est déjà garantie par 5 `raise CapCutError` situés **avant** l'ajout du segment. Pire, le validateur produirait un faux positif sur `--transition-compensation shift` (dérive volontaire de 300 ms contre une tolérance d'une frame à 16,7 ms) et donnerait un feu vert au moment exact où le brouillon et le preview divergent réellement. À la place : **un test d'intégration** sur une timeline réaliste (round-trip d'un `timeline.json` sauvegardé), dans les deux modes de compensation.
- **`ThreadPoolExecutor` pour le téléchargement des chunks.** Mesuré proprement : ~5 % du gain, le sommeil en représente ~95 %. Le « 8,4× » comparait deux configurations différant sur deux variables. 8 workers était déjà plus lent que 4.
- **`div.paginate` comme condition d'arrêt de la découverte.** « Next Page » = groupe de dix pages suivant : la règle s'arrête page 71 et laisse 12 épisodes manquants. Le `if not new: break` existant est déjà correct et indépendant du markup.
- **`raise ScraperError` sur découverte incomplète.** L'exception est avalée par le `except Exception` de `resolve_chapter_urls`, qui retombe sur `lo=hi=1` : le batch passe de « 541 sur 653 » à « 1 sur 653 ».
- **Détecteur de structure de texte pour les encarts.** Sur ce corpus, baisser la constante à 80 px récupère 8/8. Le détecteur ajoute un réglage à régler pour zéro gain mesurable.
- **Ken Burns exprimé en amplitude/seconde.** `min(0.012 × durée, 0.05)` sature dès 4,17 s : 243 plans sur 248 sont inchangés, et les 5 restants voient leur zoom *baisser*. C'est la cartographie temporelle qu'il faut changer, pas l'amplitude.
- **Le champ sha1 dans `SceneAudio`.** Le manifeste contient déjà toutes les entrées ; un digest nouveau garantit un cache miss total au premier run sur les trois manifestes existants.
- **Attribution du sommeil du limiteur par chapitre, et offsets Gantt.** Avec un limiteur unique partagé, l'attribution est arbitraire — le sommeil d'un chapitre est causé par ses voisins. Le total global existe déjà, et `acquire()` retourne déjà les secondes attendues. Rien ne consomme un Gantt.
- **Commande `capcut calibrate`.** Le brouillon d'ep3 existe déjà dans le dossier CapCut, CapCut écrit lui-même un `.bak` à la sauvegarde, et sa quantification frame est de ~16 ms contre une transition de 300 ms : la mesure est sans ambiguïté sans une ligne de code produit.
- **Annuler les tâches en attente sur `quota_hit`.** Gain de quelques secondes (coroutines parquées, sans CPU), édition la plus risquée du lot, et elle détruirait le pré-cache des cases qui a économisé 32–46 s × 5.
- **Ancrage VFX sur le plan `punch_in` et plafond par minute.** Disponible dans 17 scènes VFX sur 37 seulement, et la densité est déjà à 1,1–1,3/min — après le clamp il n'y a plus rien à plafonner.
- **Ajouter 403 aux statuts réessayables.** C'est le garde-fou UA/Referer : réessayer masque le seul message d'erreur qui dit ce qui ne va pas.
- **Six nouveaux flags CLI sur le slicer.** Aucun besoin de réglage mesuré sur ce corpus, sur une commande qui compte déjà 109 options.
- **Monter le preview à 85 px / contour 4 px.** Violerait la règle des 2 px de contour et invaliderait tous les previews déjà jugés. C'est CapCut qu'il faut baisser.
- **« Faire remonter `transition_drift_s` dans le résumé ».** Déjà loggé en INFO par chapitre à `timeline_builder.py:748-753`.
- **Le chiffre de « ~195 appels Gemini » sur `--redo montage`.** Les cinq chapitres concernés mesurent 10,3–13,8 MiB, donc chemin un-seul-appel : ~5 appels. Surestimation d'un facteur ~40.
- **Flag « garder la case pour l'analyse seulement ».** Nouveau champ `Panel` + round-trip `panels.json` + filtrage dans pacing : M/L pour un problème qui se règle en excluant les cases courtes de l'élargissement SHORT.
- **Formule de contour CapCut `stroke_px/(0.0426 × font_px) × 100`.** Incohérente avec sa propre réponse annoncée d'un facteur 10, et fausse contre mesure d'un facteur ~5.

**Deux corrections factuelles à la mémoire projet, au passage.** (1) « CapCut installé, brouillons copiés mais jamais confirmés ouverts » est **faux** : trois brouillons portent `app_version 9.4.0` et un `draft_cover.jpg`, CapCut les a ouverts et réenregistrés en préservant tout. (2) Le pic mémoire d'un assemblage est de 2230 MB, pas ~1 GB, et trois tournent en parallèle par défaut (~6,7 GB atteignables).

---

## 4. Ce qui reste incertain

**Tranchable en une minute d'attention humaine :**

- **La sémantique de `is_overlap` dans CapCut.** Tous les brouillons que CapCut a réenregistrés contenaient 0 transition — et leurs originaux aussi, donc CapCut n'a rien retiré, ils précèdent simplement la fonctionnalité. Personne ne sait si CapCut rapproche les plans. Enjeu : 6,90/7,20/2,40 s de dérive image/voix. La prudence dit que la confiance de 0,75 annoncée sur la *direction* du bug n'est pas méritée — le prior honnête est 50/50. Protocole : fermer CapCut complètement (sinon copie « (2) »), ouvrir le brouillon **ep3** (le plus petit, 60 plans, 8 transitions, dérive attendue 2,40 s), Ctrl+S, quitter. Trois lectures suffisent : la `duration` de tête (466,81 s → « none » correct ; 464,41 s → « shift » nécessaire), l'écart entre le dernier segment V2 et le dernier segment A1, et l'œil de l'utilisateur pendant que le projet est ouvert. Si CapCut ne rapproche rien, la récompense est négative-mais-utile : supprimer `"shift"`, `compensated_ranges()` et les deux options CLI, soit ~40 lignes de surface non testée en moins.

- **Les 22 % de silence sont-ils du gras ou de la respiration ?** C'est une question d'oreille, pas de code. 175 pauses de 1,2 s peuvent être un temps dramatique délibéré. Auditionner ep3 à 0,45 et 0,6 avant de figer le défaut de profil.

**Tranchable en regardant un rendu :**

- **Pixellisation à 105 % soutenu.** Sous la rampe Ken Burns, un plan de 19 s resterait à grossissement maximal pendant ~15 s au lieu de l'effleurer. La règle des 105 % a déjà été revue une fois après avoir vu de la pixellisation. Rendre un chapitre et regarder.
- **Les cases supplémentaires (découpe à 1080, seuil 80 px) se regardent-elles bien ?** Un encart de narration de 100 px dans un cadre 1920×1080 en `contain`, à résolution native, est très fin. Vérifier sur le preview avant de généraliser.
- **Les cases récupérées par `expand_panel_ids` sont-elles narrativement utilisables en LONG ?** `limit_panels_for_duration` sélectionne les survivants par hauteur de pixel, pas par position narrative : une scène à 22 cases élargies peut perdre un bloc contigu en son milieu. Un contrôle visuel sur `preview_60s.mp4` avant d'en faire un défaut.
- **Raccourcir un `glow` gaspille-t-il quelque chose ?** Côté preview c'est une vignette respirante à 0,35 de force, qui se lit comme un éclairage et non comme un effet. Côté CapCut, l'intensité du built-in reste inférée de son nom de catalogue. C'est l'argument pour un plafond par type plutôt qu'uniforme.

**Non expliqué :**

- **Une case affichée en miroir.** Dans `ep2/preview_60s.mp4` à t=40 s, le texte anglais incrusté dans le dessin (« Skills », « Basic Heal », « Entangling Vines ») est inversé. `grep -rn "flip" src/modules/*.py` ne renvoie rien — l'origine est donc le slicer ou la source, pas un retournement délibéré. Une case retournée horizontalement avec du texte lisible dedans est un défaut bien plus visible que la durée des VFX. À investiguer séparément.
- **La constante 204 à travers les versions de CapCut.** Calibrée sur deux brouillons en orientations opposées (accord à 0,4 %), sur une seule installation 9.4.0. Un modèle proportionnel à la hauteur est réfuté, donc la forme est bonne ; la valeur peut bouger à une mise à jour. La re-dériver depuis un `draft_cover.jpg` chaque fois qu'il y en a un.
- **La parité verticale exacte des sous-titres.** CapCut centre la boîte d'**encre**, pas une boîte de ligne (vérifié au pixel sur deux brouillons), et les deux échantillons disponibles ont des hampes hautes sans jambages. Une constante `transform_y` unique dérivera de quelques pixels selon les descendantes. Accepter l'écart, ou mesurer un bloc avec jambage avant de figer la formule.