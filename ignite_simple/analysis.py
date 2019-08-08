"""This module is tasked with generating analysis text, images, animations,
and videos using only the output from the model manager, the dataset, the
loss metric, and the accuracy style.
"""
import typing
import logging
import ignite_simple  # pylint: disable=unused-import
import ignite_simple.utils as utils
import ignite_simple.analarams as aparams

def analyze(dataset_loader: typing.Tuple[str, str, tuple, dict],
            loss_loader: typing.Tuple[str, str, tuple, dict],
            folder: str,
            settings: typing.Union[str, aparams.AnalysisSettings],
            accuracy_style: str, cores: int):
    """Performs the requested analysis of the given folder, which is assumed to
    have been generated from model_manager.train. The output folder structure
    is as follows:

    .. code:: none

        folder/
            analysis/
                TODO

    :param dataset_loader: the module and corresponding attribute that gives a
        training and validation dataset when invoked with the specified
        arguments and keyword arguments.

    :param loss_loader: the module and corresponding attribute that gives a
        loss nn.Module when invoked with the specified arguments and keyword
        arguments

    :param folder: where the model manager saved the trials to analyze

    :param settings: the settings for analysis, or the name of the preset to
        use. Common preset names are `none`, `text`, `images`, `animations`,
        and `videos`. For the full list, see the ignite_simple.analarams
        module.

    :param accuracy_style: how performance was calculated. one of
        'classification', 'multiclass', and 'inv-loss'. See train for
        details.

    :param cores: the number of physical cores this can assume are available
        to speed up analysis.
    """
    dataset_loader = utils.fix_imports(dataset_loader)
    loss_loader = utils.fix_imports(loss_loader)

    settings = aparams.get_settings(settings)

    logger = logging.getLogger(__name__)

    logger.info('Analyzing %s...', folder)
