"""Describes parameters relevant for analysis. The parameters are specified
for the output, and what's required to produce that output is calculated from
that.
"""

class AnalysisSettings:
    """Describes the analysis settings. Note that, where relevant and
    equivalent, parallel computations are described as if they were
    performed sequentially.

    :ivar bool lr_selection_explanation: True if the output should include a
    broad text explanation of how the learning rate was selected, False
    otherwise.

    :ivar bool lr_selection_results: True if the output should include a text
    explanation of the selected learning rate, False otherwise.

    :ivar bool batch_selection_explanation: True if the output should include
    a broad text explanation of how the minibatch size was selected, False
    otherwise.

    :ivar bool batch_selection_results: True if the output should include a
    text explanation of the selected minibatch size, False otherwise.

    :ivar bool hparam_selection_specifics: True if the output should include
    a text explanation about how specifically we came about selecting the
    learning rate and batch size, by going through the numbers, False
    otherwise.

    :ivar bool hparam_selection_specific_imgs: True if the output should
    include figures with captions for the hyper paramater selection process as
    it occurred for this model specifically, False otherwise.

    :ivar bool training_explanation: True if the output should include a broad
    text explanation of the training procedure, False otherwise.

    :ivar bool training_metrics: True if the output should include text metrics
    for the models performance through training, False otherwise.

    :ivar bool training_metric_imgs: True if the output should include figures
    with captions for the performance of the model through training, i.e.,
    accuracy vs epoch and loss vs epoch figures.

    :ivar bool final_metrics: True if the output should include in text the
    final accuracy and loss, False otherwise.

    :ivar bool typical_run_pca3dvis: True if the output should include a video
    from pca3dvis for the points as they moved through the network, False
    otherwise.

    :ivar bool typical_run_pca3dvis_draft: If `typical_run_pca3dvis` is True,
    then the value of `typical_run_pca3dvis_draft` corresponds to if the video
    should have draft settings. If `typical_run_pca3dvis` is False, this has
    no effect.
    """
    def __init__(self,
                 lr_selection_explanation: bool,
                 lr_selection_results: bool,
                 batch_selection_explanation: bool,
                 batch_selection_results: bool  ,
                 hparam_selection_specifics: bool,
                 hparam_selection_specific_imgs: bool,
                 training_explanation: bool,
                 training_metrics: bool,
                 training_metric_imgs: bool,
                 final_metrics: bool,
                 typical_run_pca3dvis: bool,
                 typical_run_pca3dvis_draft: bool):
        self.lr_selection_explanation = lr_selection_explanation
        self.lr_selection_results = lr_selection_results
        self.batch_selection_explanation = batch_selection_explanation
        self.batch_selection_results = batch_selection_results
        self.hparam_selection_specifics = hparam_selection_specifics
        self.hparam_selection_specific_imgs = hparam_selection_specific_imgs
        self.training_explanation = training_explanation
        self.training_metrics = training_metrics
        self.training_metric_imgs = training_metric_imgs
        self.final_metrics = final_metrics
        self.typical_run_pca3dvis = typical_run_pca3dvis
        self.typical_run_pca3dvis_draft = typical_run_pca3dvis_draft

def none() -> AnalysisSettings:
    """Analysis settings that produce absolutely nothing

    :returns: the none preset for analysis settings, which produces no output
    :rtype: AnalysisSettings
    """
    return AnalysisSettings(
        lr_selection_explanation=False,
        lr_selection_results=False,
        batch_selection_explanation=False,
        batch_selection_results=False,
        hparam_selection_specifics=False,
        hparam_selection_specific_imgs=False,
        training_explanation=False,
        training_metrics=False,
        training_metric_imgs=False,
        final_metrics=False,
        typical_run_pca3dvis=False,
        typical_run_pca3dvis_draft=False
    )

def text() -> AnalysisSettings:
    """Analysis output that uses text only

    :returns: the text preset for analysis settings, which produces text
    output.
    :rtype: AnalysisSettings
    """
    res = none()
    res.lr_selection_explanation = True
    res.lr_selection_results = True
    res.batch_selection_explanation = True
    res.batch_selection_results = True
    res.hparam_selection_specifics = True
    res.training_explanation = True
    res.training_metrics = True
    res.final_metrics = True
    return res

def images() -> AnalysisSettings:
    """Analysis output that uses text and images only

    :returns: the images preset for analysis settings, which produces text
    and image output.
    :rtype: AnalysisSettings
    """
    res = text()
    res.hparam_selection_specific_imgs = True
    res.training_metric_imgs = True
    return res

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
