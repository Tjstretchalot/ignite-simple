"""Produces the html folder for analysis. This is essentially just copying
the html/ folder and preprocessing certain ids.
"""

from ignite_simple.analarams import AnalysisSettings
import json
from bs4 import BeautifulSoup
import os
import shutil
import math
import numpy as np

def _set_all(soup, iden, val):
    for ele in soup.find_all(id=iden):
        ele.string = str(val)

def find_referenced_images(html):
    """Searches the given html file for images which are referenced in the standard way
    (src attribute on img tag) and returns them as they are written."""
    with open(html, 'r') as infile:
        soup = BeautifulSoup(infile.read(), 'html.parser')

    return set(img['src'] for img in soup.find_all('img'))


HTML_REFD_IMAGES = find_referenced_images(os.path.join(
    os.path.dirname(__file__), '../html/index.html'))

# we strip the ../ that the images are referenced with
HTML_REFD_IMAGES = set(i[3:] for i in HTML_REFD_IMAGES)

def text_analyze(settings: AnalysisSettings, folder: str):
    """Analyzes the given folder produced by the model manager, storing the
    results in folder/analysis/html

    Args:
        settings (AnalysisSettings): the analysis settings.
        folder (str): the folder that was passed to the model manager.
    """

    with open(os.path.join(folder, 'hparams', 'misc.json')) as infile:
        misc = json.load(infile)

    with np.load(os.path.join(folder, 'results.npz')) as infile:
        results = dict(infile.items())

    in_html_folder = os.path.join(os.path.dirname(__file__), '../html')
    in_html_folder = os.path.abspath(in_html_folder)

    out_html_folder = os.path.abspath(os.path.join(folder, 'analysis', 'html'))

    cwd = os.getcwd()
    if os.path.exists(out_html_folder):
        shutil.rmtree(out_html_folder)

    shutil.copytree(in_html_folder, out_html_folder)
    os.chdir(cwd)

    with open(os.path.join(in_html_folder, 'index.html'), 'r') as infile:
        soup = BeautifulSoup(infile.read(), 'html.parser')

    for key, val in misc.items():
        _set_all(soup, key.replace('_', '-'), val)

    _set_all(soup, 'trials-count', results['final_perf_train'].shape[0])
    _set_all(soup, 'trials-train-best-performance', results['final_perf_train'].max())
    _set_all(soup, 'trials-train-best-loss', results['final_loss_train'].min())
    _set_all(soup, 'trials-train-mean-performance', results['final_perf_train'].mean())
    _set_all(soup, 'trials-train-std-performance', results['final_perf_train'].std())
    _set_all(soup, 'trials-train-mean-loss', results['final_loss_train'].mean())
    _set_all(soup, 'trials-train-std-loss', results['final_loss_train'].std())
    _set_all(soup, 'trials-val-best-performance', results['final_perf_val'].max())
    _set_all(soup, 'trials-val-best-loss', results['final_loss_val'].min())
    _set_all(soup, 'trials-val-mean-performance', results['final_perf_val'].mean())
    _set_all(soup, 'trials-val-std-performance', results['final_perf_val'].std())
    _set_all(soup, 'trials-val-mean-loss', results['final_loss_val'].mean())
    _set_all(soup, 'trials-val-std-loss', results['final_loss_val'].std())

    _set_all(soup, 'initial-half-cycle', misc['initial_cycle_time'] // 2)

    if misc['batch_sweep_trials_each'] <= 0:
        soup.find(id='batch-size-sweep-other').string = ''
    else:
        soup.find(id='batch-size-sweep-fastest').string = ''

    if math.isnan(misc['second_lr_num_trials']):
        soup.find(id='second-lr-sweep').string = ''
    else:
        soup.find(id='no-second-lr-sweep').string = ''

    if not settings.hparam_selection_specifics:
        soup.find(id='initial-lr-sweep').string = ''
        soup.find(id='batch-size-sweep').string = ''
        soup.find(id='no-second-lr-sweep').string = ''
        soup.find(id='second-lr-sweep').string = ''

    if not settings.hparam_selection_specific_imgs:
        for div in soup.select('.hparam-figures'):
            div.string = ''

    if not settings.training_metric_imgs:
        for div in soup.select('.trial-figures'):
            div.string = ''

    with open(os.path.join(out_html_folder, 'index.html'), 'w') as outfile:
        outfile.write(str(soup))
