"""Describes parameters relevant for analysis. The parameters are specified
for the output, and what's required to produce that output is calculated from
that.
"""
import typing

class AnalysisSettings:
    """Describes the analysis settings. Note that, where relevant and
    equivalent, parallel computations are described as if they were
    performed sequentially.

    :ivar bool hparam_selection_specifics: True if the output should include
        a text explanation about how specifically we came about selecting the
        learning rate and batch size, by going through the numbers, False
        otherwise.

    :ivar bool hparam_selection_specific_imgs: True if the output should
        include figures with captions for the hyper paramater selection process
        as it occurred for this model specifically, False otherwise.

    :ivar bool training_metric_imgs: True if the output should include figures
        with captions for the performance of the model through training, i.e.,
        accuracy vs epoch and loss vs epoch figures.

    :ivar bool suppress_extra_images: Suppresses generating images which are
        not displayed in the html report.

    :ivar bool typical_run_pca3dvis: True if the output should include a video
        from pca3dvis for the points as they moved through the network, False
        otherwise.

    :ivar bool typical_run_pca3dvis_draft: If `typical_run_pca3dvis` is True,
        then the value of `typical_run_pca3dvis_draft` corresponds to if the
        video should have draft settings. If `typical_run_pca3dvis` is False,
        this has no effect.

    """
    def __init__(self,
                 hparam_selection_specifics: bool,
                 hparam_selection_specific_imgs: bool,
                 training_metric_imgs: bool,
                 suppress_extra_images: bool,
                 typical_run_pca3dvis: bool,
                 typical_run_pca3dvis_draft: bool):
        self.hparam_selection_specifics = hparam_selection_specifics
        self.hparam_selection_specific_imgs = hparam_selection_specific_imgs
        self.training_metric_imgs = training_metric_imgs
        self.suppress_extra_images = suppress_extra_images
        self.typical_run_pca3dvis = typical_run_pca3dvis
        self.typical_run_pca3dvis_draft = typical_run_pca3dvis_draft

    def __repr__(self):
        return f'AnalysisSettings(**{self.__dict__})'

def none() -> AnalysisSettings:
    """Analysis settings that produce absolutely nothing

    :returns: the none preset for analysis settings, which produces no output
    :rtype: AnalysisSettings
    """
    return AnalysisSettings(
        hparam_selection_specifics=False,
        hparam_selection_specific_imgs=False,
        training_metric_imgs=False,
        suppress_extra_images=True,
        typical_run_pca3dvis=False,
        typical_run_pca3dvis_draft=False
    )

def text() -> AnalysisSettings:
    """Analysis output that uses text only

    :returns: the text preset for analysis settings, which produces text
        output.
    :rtype: AnalysisSettings
    """
    return AnalysisSettings(
        hparam_selection_specifics=True,
        hparam_selection_specific_imgs=False,
        training_metric_imgs=False,
        suppress_extra_images=True,
        typical_run_pca3dvis=False,
        typical_run_pca3dvis_draft=False
    )

def images() -> AnalysisSettings:
    """Analysis output that uses text and images only

    :returns: the images preset for analysis settings, which produces text
        and image output.
    :rtype: AnalysisSettings
    """
    return AnalysisSettings(
        hparam_selection_specifics=True,
        hparam_selection_specific_imgs=True,
        training_metric_imgs=True,
        suppress_extra_images=False,
        typical_run_pca3dvis=False,
        typical_run_pca3dvis_draft=False
    )

def images_minimum() -> AnalysisSettings:
    """Analysis output that uses text and images only, and does not produce
    images which do not make it into the text report.

    :returns: the images_minimum preset for analysis settings, which produces
        text and enough images for a pretty text output.
    :rtype: AnalysisSettings
    """
    return AnalysisSettings(
        hparam_selection_specifics=True,
        hparam_selection_specific_imgs=True,
        training_metric_imgs=True,
        suppress_extra_images=True,
        typical_run_pca3dvis=False,
        typical_run_pca3dvis_draft=False
    )


def animations_draft() -> AnalysisSettings:
    """Analysis output that uses text, images, and animations but the
    animations are given draft settings (i.e. low-fps and low resolution)
    to speed up output. For the purposes of this module, animations are
    simply videos under 15 seconds.

    :returns: the animations-draft preset for analysis settings, which produces
        text, image, and draft-quality animations in the output.
    :rtype: AnalysisSettings
    """
    return images()

def animations() -> AnalysisSettings:
    """Analysis output that uses text, images, and animations. Animations are
    videos which are under 15 seconds for the purpose of this module.

    :returns: the animations preset for analysis settings, which produces text,
        images, and animations in the output.
    :rtype: AnalysisSettings
    """
    return images()

def video_draft() -> AnalysisSettings:
    """Analysis output that uses text, images, animations, and video but the
    animations and video(s) are given draft settings (i.e. low-fps and
    low-resolution) to speed up output

    :returns: the video-draft preset for analysis settings, which produces
        text, images, draft-quality animations, and draft-quality videos in the
        output.
    :rtype: AnalysisSettings
    """
    res = animations_draft()
    res.typical_run_pca3dvis = True
    res.typical_run_pca3dvis_draft = True
    return res

def video() -> AnalysisSettings:
    """Analysis output that uses text, images, animations, and video

    :returns: the video preset for analysis settings, which produces text,
        images, animations, and videos in the output.
    :rtype: AnalysisSettings
    """
    res = animations()
    res.typical_run_pca3dvis = True
    res.typical_run_pca3dvis_draft = False
    return res

def images_min_plus_pca3dvis() -> AnalysisSettings:
    """Images min preset plus the pca3dvis video for one trial

    :returns: the images-min+video preset for analysis settings, which is
        the images-min preset with the pca3dvis video
    :rtype: AnalysisSettings
    """
    res = images_min()
    res.typical_run_pca3dvis = True
    res.typical_run_pca3dvis_draft = False
    return res

NAME_TO_PRESET = {
    'none': none,
    'text': text,
    'images': images,
    'image': images,
    'images-min': images_minimum,
    'images_min': images_minimum,
    'images-min+pca3dvis': images_min_plus_pca3dvis,
    'images_min+pca3dvis': images_min_plus_pca3dvis,
    'images-minimum': images_minimum,
    'images_minimum': images_minimum,
    'animation-draft': animations_draft,
    'animation_draft': animations_draft,
    'animations-draft': animations_draft,
    'animations_draft': animations_draft,
    'animation': animations,
    'animations': animations,
    'video-draft': video_draft,
    'video_draft': video_draft,
    'videos-draft': video_draft,
    'videos_draft': video_draft,
    'video': video,
    'videos': video
}

def get_settings(preset: typing.Union[
        str, AnalysisSettings]) -> AnalysisSettings:
    """Gets the analysis settings from the given preset name or analysis
    settings.

    :param preset: either a str name of a preset or the settings to return

    :returns: the corresponding preset or just the argument if its already
        AnalysisSettings
    """
    if isinstance(preset, AnalysisSettings):
        return preset
    return NAME_TO_PRESET[preset]()
