"""ManhwaMaker Studio : interface web locale pour lancer, suivre et rendre les vidéos.

``python -m src.studio`` démarre le serveur (:mod:`src.studio.server`) sur
http://127.0.0.1:8777. Les traitements tournent dans des processus séparés
(:mod:`src.studio.jobs`) dont la progression est lue en direct (:mod:`src.utils.progress`)
et convertie en temps restant (:mod:`src.studio.eta`).
"""
