"""Schémas Pydantic décrivant un chapitre scrapé (Webtoons ou Asura Scans).

Ce module ne contient que des structures de données : aucune requête réseau,
aucun traitement d'image. Il est partagé par le scraper (qui le remplit) et
par les modules aval (slicer, analyzer) qui ont besoin des métadonnées.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChapterMeta(BaseModel):
    """Métadonnées d'un chapitre Webtoons.

    Attributes:
        url: URL demandée par l'utilisateur (telle que fournie).
        final_url: URL effective après redirections HTTP (``response.url``).
        series_title: Titre de la série (ex. ``"Tower of God"``).
        episode_title: Titre de l'épisode (ex. ``"[Season 1] Ep. 0"``).
        title_no: Identifiant numérique de la série (paramètre ``title_no``
            de l'URL), ``None`` s'il est absent.
        episode_no: Numéro de l'épisode (paramètre ``episode_no`` de l'URL
            Webtoons, sinon libellé ``span.tx`` de la page ; numéro de chapitre
            Asura, décimal pour les chapitres bonus comme ``74.5``), ``None``
            s'il est absent.
        image_urls: URLs des morceaux d'image, dans l'ordre de lecture.
        chunk_sizes: Dimensions ``(largeur, hauteur)`` de chaque morceau
            téléchargé, dans le même ordre que ``image_urls``. Vide tant que
            les images n'ont pas été téléchargées.

    Conformément au contrat partagé, seuls ``chunk_sizes`` a une valeur par
    défaut : ``title_no`` / ``episode_no`` doivent être fournis explicitement
    (éventuellement ``None``) et ``image_urls`` est obligatoire.
    """

    url: str
    final_url: str
    series_title: str
    episode_title: str
    title_no: int | None
    episode_no: int | float | None
    image_urls: list[str]
    chunk_sizes: list[tuple[int, int]] = Field(default_factory=list)

    @property
    def n_chunks(self) -> int:
        """Nombre de morceaux d'image composant le chapitre."""
        return len(self.image_urls)

    def summary(self) -> str:
        """Résumé mono-ligne **ASCII pur** utilisable dans les logs et la console.

        Les titres Webtoons contiennent souvent des guillemets typographiques,
        tirets cadratins ou alphabets non latins ; ils sont translittérés en
        ``?`` pour que la chaîne soit imprimable sur une console cp1252.
        """
        text = (
            f"{self.series_title} | {self.episode_title} "
            f"(title_no={self.title_no}, episode_no={self.episode_no}, "
            f"chunks={self.n_chunks})"
        )
        return text.encode("ascii", "replace").decode("ascii")
