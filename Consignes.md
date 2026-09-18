# Product Requirement Document (PRD) — Auto-Manhwa Recap Generator

## 1. Description du Projet
CLI Python automatisant la création de vidéos récapitulatives de Manhwas (Webtoons) pour YouTube. 
Input : URL d'un chapitre Webtoons.com.
Output : Un dossier projet CapCut (.draft) prêt à être ouvert et exporté en 16:9 60 FPS.

## 2. Stack Technique
- Langage : Python 3.11+
- Scraper : requests / BeautifulSoup (avec headers HTTP Referer Webtoons)
- Traitement Image : OpenCV, Pillow, NumPy
- VLM / LLM : google-genai (Gemini 2.0 Flash / 1.5 Pro)
- TTS : KokoroTTS (KPipeline local)
- Pipeline CapCut : pyCapCut / Générateur JSON draft_content

## 3. Architecture des Modules

### Module 1 : Scraper & Stitcher (`scraper.py`)
- Télécharger les images du chapitre Webtoons.com avec User-Agent + Referer.
- Recomposer la bande continue en fusionnant les morceaux de 1280px via PIL (np.vstack).

### Module 2 : Smart Slicer (`slicer.py`)
- Découper la bande continue en cases individuelles via la variance des lignes de pixels (gouttières blanches/noires).
- Paramètres : variance < 15.0, min_gap = 20px, margin_padding = 15px.
- Fallback : Si un panel mesure > 1500px, marquer le tag "type": "scroll_vertical".

### Module 3 : Gemini VLM Analyzer (`analyzer.py`)
- Envoyer les cases découpées à l'API Gemini par lots de 10 à 15 images max.
- Générer un JSON structuré (Pydantic) contenant :
  - panel_ids : liste des cases associées
  - narration : texte de synthèse au présent, 3ème personne
  - emotion : ton de la scène

### Module 4 : Kokoro TTS Engine (`tts_engine.py`)
- Traiter chaque texte de narration via KokoroTTS (KPipeline).
- Appliquer un dictionnaire de remplacement phonétique pour les noms propres.
- Exporter chaque segment en .wav + ajouter +0.3s de silence padding.
- Retourner la durée exacte de chaque segment pour la timeline.

### Module 5 : CapCut Draft Builder (`capcut_builder.py`)
- Assembler la séquence 1920x1080 60 FPS.
- Background : Image de la case étirée + Flou Gaussien + filtre sombre -30%.
- Foreground : Case originale centrée + Zoom Ken Burns doux (100% -> 105%).
- Alignement strict de la durée d'affichage de l'image sur la durée du clip KokoroTTS associé.
- Pistes : V1 (Blur BG), V2 (Main Panel), A1 (Voiceover Kokoro), A2 (BGM -22dB), T1 (Subtitles).

## 4. Garde-fous et Sécurités
1. Headers HTTP obligatoires pour éviter l'erreur 403 sur Webtoons.
2. Traitement par lot (batching) sur Gemini API pour éviter les timeouts.
3. Remplacement phonétique systématique avant synthèse vocale.
4. Auto-scroll vertical pour les cases géantes d'action.