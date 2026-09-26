"""Schémas Pydantic de la timeline vidéo (Module 5).

La timeline est le **plan de montage** commun au brouillon CapCut et au rendu de
prévisualisation ffmpeg : pour chaque scène narrée, les cases clés sont affichées à
la suite pendant exactement la durée du segment audio de la scène (garde-fou : la
durée d'affichage est calée sur la voix off).

Chaque case est affichée **à sa résolution native**, centrée dans le cadre 16:9 (le
fond flouté et assombri comble les côtés) ; une case plus grande que le cadre est
réduite pour y tenir, jamais agrandie. Mouvements (``Motion``) :

- ``ken_burns`` : zoom doux 100 % → 105 % sur la durée du clip ;
- ``punch_in`` : zoom d'impact rapide, 105 % atteint en 0,2 s puis maintenu
  (cases ``action_heavy``) ;
- ``scroll_vertical`` : ancien défilement, conservé pour relire de vieilles
  timelines ; rendu comme ``ken_burns``.

Design sonore : ``sfx`` (bruitages courts) et ``bgm`` (musique de fond par
ambiance, bouclée si elle est plus courte que la vidéo, fondus enchaînés aux
changements d'ambiance, −22 dB).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from src.models.format_profile import CropWindow, VideoFormat

#: Mouvement appliqué à une case au premier plan.
#:
#: ``fast_pan`` est le balayage rapide du format court. Il parcourt la marge disponible
#: autour de la fenêtre de cadrage : sur des cases plus larges que le 9:16, cette marge
#: est **horizontale**, la fenêtre occupant déjà toute la hauteur.
Motion = Literal["ken_burns", "punch_in", "fast_pan", "scroll_vertical"]
#: Gain par défaut de la musique de fond (dB) : sous la voix Kokoro sans la couvrir.
DEFAULT_BGM_GAIN_DB: float = -22.0
#: Intensité du dynamisme du montage (animations de sous-titres, transitions, VFX) :
#: ``none`` = montage sobre d'origine, ``subtle`` = animations de texte seules,
#: ``punchy`` = animations + transitions + effets superposés.
DynamicsLevel = Literal["none", "subtle", "punchy"]


class CueAnimation(BaseModel):
    """Animation d'un bloc de sous-titre (piste T1).

    Les noms sont ceux du catalogue CapCut (``TextIntro`` / ``TextLoopAnim`` de
    ``pycapcut``), volontairement stockés en texte : la timeline reste lisible et
    sérialisable sans dépendre de ``pycapcut``.

    Attributes:
        intro: animation d'entrée (vide = aucune).
        loop: animation en boucle, jouée sur le reste du bloc (vide = aucune).
        duration_s: durée de l'animation d'entrée ; elle doit rester une fraction
            du bloc, sans quoi un bloc de 0,4 s serait entièrement consommé par
            son animation.
    """

    intro: str = ""
    loop: str = ""
    duration_s: float = Field(default=0.0, ge=0)


class ClipTransition(BaseModel):
    """Transition entre deux cases, **portée par la case qui précède**.

    C'est la convention de ``pycapcut`` (``VideoSegment.add_transition`` s'applique
    au segment précédent), reprise telle quelle pour éviter toute ambiguïté.

    Attributes:
        kind: nom de la transition dans le catalogue CapCut.
        duration_s: durée de la transition.
        overlap: ``True`` si la transition empiète sur les deux cases voisines
            (``is_overlap`` du catalogue CapCut : vrai pour 1131 des 1137
            transitions). C'est ce drapeau qui justifie de garder des durées
            courtes, une transition qui empiète consommant du temps de montage.
    """

    kind: str
    duration_s: float = Field(gt=0)
    overlap: bool = True


class VfxClip(BaseModel):
    """Effet visuel superposé sur une plage de temps (scène « spectaculaire »).

    Deux sources possibles, dans cet ordre de préférence :

    - ``effect`` : effet **intégré** au catalogue CapCut (aucun fichier requis,
      composition correcte garantie) ;
    - ``file`` : vidéo locale à **canal alpha** posée sur une piste vidéo
      superposée. ``pycapcut`` n'expose aucun mode de fusion : une vidéo sur fond
      noir s'afficherait en rectangle noir, d'où l'exigence d'un vrai alpha.

    Attributes:
        scene_index: scène concernée.
        kind: identifiant logique (``speed_lines``, ``sparks``, ``rain``, ``glow``).
        effect: nom de l'effet CapCut intégré (vide si ``file`` est utilisé).
        file: chemin absolu d'un asset à canal alpha (vide si ``effect`` est utilisé).
        start_s: début sur la timeline.
        duration_s: durée.
        opacity: opacité de l'asset superposé (ignorée pour un effet intégré).
    """

    scene_index: int = Field(ge=0)
    kind: str
    effect: str = ""
    file: str = ""
    start_s: float = Field(ge=0)
    duration_s: float = Field(gt=0)
    opacity: float = Field(default=1.0, ge=0.0, le=1.0)

    @property
    def end_s(self) -> float:
        return self.start_s + self.duration_s


class PanelClip(BaseModel):
    """Affichage d'une case sur la piste V2 (et son fond flouté sur V1).

    Attributes:
        scene_index: scène à laquelle la case appartient.
        panel_index: ``Panel.index`` de la case.
        file: fichier PNG de la case, relatif à ``Timeline.panels_dir``.
        width: largeur de la case (px).
        height: hauteur de la case (px).
        start_s: début sur la timeline (secondes).
        duration_s: durée d'affichage (secondes).
        motion: mouvement (voir :data:`Motion`).
        transition: transition vers la case **suivante** (``None`` = coupe franche).
        crop: fenêtre affichée dans la case (``None`` = la case entière). Elle porte son
            propre mode de pose, ``cover`` ou ``contain``.
        subshot: rang du plan parmi ceux tirés de la même case (0 = premier).
        emotion: émotion de la scène (pilote le style du rendu Kdenlive ; vide sur les
            timelines écrites avant son ajout).
    """

    scene_index: int = Field(ge=0)
    panel_index: int = Field(ge=0)
    file: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    start_s: float = Field(ge=0)
    duration_s: float = Field(gt=0)
    motion: Motion
    transition: ClipTransition | None = None
    crop: CropWindow | None = None
    subshot: int = Field(ge=0, default=0)
    emotion: str = ""

    @property
    def end_s(self) -> float:
        return self.start_s + self.duration_s


class AudioClip(BaseModel):
    """Segment de voix off sur la piste A1."""

    scene_index: int = Field(ge=0)
    file: str
    start_s: float = Field(ge=0)
    duration_s: float = Field(gt=0)

    @property
    def end_s(self) -> float:
        return self.start_s + self.duration_s


class SfxClip(BaseModel):
    """Bruitage court (piste A3) déclenché au début d'une case.

    Attributes:
        scene_index: scène concernée.
        panel_index: case dont la transition est soulignée.
        start_s: instant de déclenchement.
        kind: type de bruitage (``swoosh``, ``impact``, ``roar``).
        file: fichier audio (chemin absolu).
        gain_db: gain appliqué (dB).
    """

    scene_index: int = Field(ge=0)
    panel_index: int = Field(ge=0)
    start_s: float = Field(ge=0)
    kind: str
    file: str
    gain_db: float = -12.0


class BgmClip(BaseModel):
    """Segment de musique de fond (piste A2) pour une ambiance donnée.

    La musique est **bouclée** si le segment dure plus longtemps qu'elle. Les
    segments consécutifs se chevauchent de la durée du fondu enchaîné : le
    précédent s'éteint (``fade_out_s``) pendant que le suivant monte (``fade_in_s``).
    """

    mood: str
    file: str
    start_s: float = Field(ge=0)
    duration_s: float = Field(gt=0)
    gain_db: float = DEFAULT_BGM_GAIN_DB
    fade_in_s: float = Field(default=0.0, ge=0)
    fade_out_s: float = Field(default=0.0, ge=0)

    @property
    def end_s(self) -> float:
        return self.start_s + self.duration_s


class WordTiming(BaseModel):
    """Minutage d'un mot à l'intérieur d'un bloc de sous-titre.

    Sert au surlignage progressif du format court. Les bornes sont **absolues** sur la
    timeline, comme celles du bloc qui les contient.
    """

    text: str
    start_s: float = Field(ge=0)
    end_s: float = Field(gt=0)


class SubtitleCue(BaseModel):
    """Sous-titre sur la piste T1 (bloc de 1 à 4 mots selon le format)."""

    scene_index: int = Field(ge=0)
    start_s: float = Field(ge=0)
    end_s: float = Field(gt=0)
    text: str
    #: Animation du bloc (``None`` = texte fixe, comportement d'origine).
    animation: CueAnimation | None = None
    #: Minutage mot à mot, pour le surlignage (vide = pas de surlignage).
    words: list[WordTiming] = Field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def word_at(self, t: float) -> int:
        """Rang du mot prononcé à l'instant ``t``, ou ``-1`` hors minutage."""
        for index, word in enumerate(self.words):
            if word.start_s <= t < word.end_s:
                return index
        return len(self.words) - 1 if self.words and t >= self.words[-1].end_s else -1


class Timeline(BaseModel):
    """Plan de montage complet d'un chapitre.

    Attributes:
        width: largeur de la séquence (px).
        height: hauteur de la séquence (px).
        fps: images par seconde.
        series_title: titre de la série.
        episode_title: titre de l'épisode.
        panels_dir: dossier des cases PNG (chemin absolu).
        audio_dir: dossier des WAV (chemin absolu).
        clips: cases affichées, ordre chronologique, contiguës.
        audio: segments de voix off, ordre chronologique, contigus.
        subtitles: sous-titres, ordre chronologique.
        sfx: bruitages.
        bgm: segments de musique de fond par ambiance.
        vfx: effets visuels superposés (scènes ``action`` / ``epic`` / ``mystery``).
        total_duration_s: durée totale de la séquence.
        bgm_file: musique de fond unique (compatibilité ; convertie en ``bgm``).
        bgm_gain_db: gain de la musique de fond (dB, −22 par défaut).
    """

    width: int = Field(default=1920, gt=0)
    height: int = Field(default=1080, gt=0)
    fps: int = Field(default=60, gt=0)
    series_title: str = ""
    episode_title: str = ""
    panels_dir: str
    audio_dir: str
    clips: list[PanelClip]
    audio: list[AudioClip]
    subtitles: list[SubtitleCue]
    sfx: list[SfxClip] = Field(default_factory=list)
    bgm: list[BgmClip] = Field(default_factory=list)
    vfx: list[VfxClip] = Field(default_factory=list)
    total_duration_s: float = Field(ge=0)
    bgm_file: str | None = None
    bgm_gain_db: float = DEFAULT_BGM_GAIN_DB
    #: Format ayant produit ce montage ; ``LONG`` pour toute timeline écrite avant
    #: l'introduction du bi-format, ce qui préserve la relecture des anciens fichiers.
    format: VideoFormat = "LONG"

    @property
    def n_scenes(self) -> int:
        return len({clip.scene_index for clip in self.clips})

    @property
    def n_transitions(self) -> int:
        """Nombre de transitions posées entre deux cases."""
        return sum(1 for clip in self.clips if clip.transition is not None)

    @property
    def transition_drift_s(self) -> float:
        """Dérive maximale (secondes) que les transitions pourraient induire dans CapCut.

        Les transitions marquées ``overlap`` consomment du temps sur les cases
        voisines : si CapCut applique ce recouvrement, l'image prend cette avance
        sur la voix en fin de chapitre. Sert d'avertissement chiffré au montage.
        """
        return sum(c.transition.duration_s for c in self.clips if c.transition and c.transition.overlap)


__all__ = [
    "Motion",
    "DynamicsLevel",
    "DEFAULT_BGM_GAIN_DB",
    "CueAnimation",
    "ClipTransition",
    "WordTiming",
    "PanelClip",
    "AudioClip",
    "SfxClip",
    "BgmClip",
    "VfxClip",
    "SubtitleCue",
    "Timeline",
]
