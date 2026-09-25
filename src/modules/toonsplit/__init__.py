"""toonsplit : découpage intelligent d'un strip webtoon en plans vidéo pleine largeur.

L'IA décide **quoi** garder (rôle du bloc, zones à garder / exclure, texte pour la voix
off) ; le code décide **où** couper (détecteurs au pixel près puis recherche sous
contraintes). Étapes : :mod:`.blocks` (gouttières), :mod:`.detectors` (têtes, personnes,
bulles, onomatopées), :mod:`.ai` (spec IA et juge, Claude CLI ou Gemini), :mod:`.search` (fenêtres),
:mod:`.pipeline` (:func:`split_strip`), :mod:`.figures` (personnages seuls, recadrage 2D), :mod:`.evaluate` (comparaison à des crops de
référence). Ligne de commande : ``python -m src.modules.toonsplit --help``.
"""

from src.modules.toonsplit.pipeline import Shot, SplitResult, analyze_strip, split_strip

__all__ = ["Shot", "SplitResult", "analyze_strip", "split_strip"]
