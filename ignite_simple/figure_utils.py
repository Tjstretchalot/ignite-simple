"""Some utility functions related to creating good looking figures."""
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import os
import typing

def set_title(fig: Figure, ax: Axes, title: str, digital: bool):
    """Sets the title for the given axes to the given value. This is more
    particular than the default matplotlib Axes.set_title.

    :param Figure fig: the figure the axes is in
    :param Axes ax: the axes which you want to have a title
    :param str title: the desired title text
    :param bool digital: if True, a large font size is selected. Otherwise,
        a smaller font size is selected

    :returns: TextCollection that was added
    """

    figw = fig.get_figwidth()
    figh = fig.get_figheight()
    figw_px = figw * fig.get_dpi()

    pad = max(int(0.3125 * figh), 1)

    font_size = int((8 / 1.92) * figw) if digital else int((4 / 1.92) * figw)
    font_size = max(5, font_size)

    axtitle = ax.set_title(title, pad=pad)
    axtitle.set_fontsize(font_size)
    renderer = fig.canvas.get_renderer()
    bb = axtitle.get_window_extent(renderer=renderer)
    while bb.width >= (figw_px - 26) * 0.9 and font_size > 9:
        font_size = max(5, font_size - 5)
        axtitle.set_fontsize(font_size)
        bb = axtitle.get_window_extent(renderer=renderer)
    return axtitle

def set_ticklabel_sizes(fig: Figure, ax: Axes, digital: bool):
    """Updates the sizes of the tick labels for the given figure based on its
    canvas size and canvas dpi.

    :param Figure fig: The figure to update
    :param Axes ax: The specific axes within the figure to update
    :param bool digital: True if this is for digital display,
        False for physical display
    """
    figw = fig.get_figwidth()

    font_size = int((30 / 19.2) * figw) if digital else int((20 / 19.2) * figw)
    font_size = max(font_size, 5)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(font_size)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(font_size)

def _filtered_savefig(filter_, fig, path, *args, **kwargs):
    if filter_(path):
        fig.savefig(path, *args, **kwargs)

def save_fig(fig: Figure, ax: Axes, title: str, outfile_wo_ext: str,
             filter_: typing.Optional[typing.Callable] = None):
    r"""Saves the given figure with many different commonly used figure sizes,
    resizing labels and titles as appropriate. This technique works only for
    a single axis plot, since for other styles different font sizes would be
    appropriate.

    :param Figure fig: The figure to save
    :param Axes ax: The axis on the figure
    :param str title: The title of the plot
    :param str outfile_wo_ext: where to save the figure, omitting the file
        extension (as many files may be saved)
    :param optional[callable] filter_: if not None, a callable object which
        accepts a string which is a (possibly relative) path to a file which
        will only be saved if the result of filter\_ is True.
    """
    filter_ = filter_ if filter_ is not None else lambda x: True

    set_title(fig, ax, title, True)
    ax.xaxis.label.set_size(48)
    ax.yaxis.label.set_size(48)
    set_ticklabel_sizes(fig, ax, True)
    _filtered_savefig(filter_, fig, outfile_wo_ext + '_1920x1080.png', dpi=100)
    set_title(fig, ax, title, False)
    ax.xaxis.label.set_size(24)
    ax.yaxis.label.set_size(24)
    set_ticklabel_sizes(fig, ax, False)
    _filtered_savefig(filter_, fig, outfile_wo_ext + '_19.2x10.8.pdf', dpi=300, transparent=True)

    fig.set_figwidth(7.25)  # paper width
    fig.set_figheight(4.08)  # 56.25% width
    fig.subplots_adjust(left=0.15, right=0.925)

    set_title(fig, ax, title, True)
    ax.xaxis.label.set_size(24)
    ax.yaxis.label.set_size(24)
    set_ticklabel_sizes(fig, ax, True)
    _filtered_savefig(filter_, fig, outfile_wo_ext + '_725x408.png', dpi=100)
    set_title(fig, ax, title, False)
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)
    set_ticklabel_sizes(fig, ax, False)
    _filtered_savefig(filter_, fig, outfile_wo_ext + '_7.25x4.08.pdf', dpi=300, transparent=True)

    fig.set_figwidth(3.54)  # column width
    fig.set_figheight(1.99)  # 56.25% width
    fig.subplots_adjust(left=0.2, right=0.975)

    set_title(fig, ax, title, True)
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    set_ticklabel_sizes(fig, ax, True)
    _filtered_savefig(filter_, fig, outfile_wo_ext + '_354x199.png', dpi=100)
    set_title(fig, ax, title, False)
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)
    set_ticklabel_sizes(fig, ax, False)
    _filtered_savefig(filter_, fig, outfile_wo_ext + '_3.54x1.99.pdf', dpi=300, transparent=True)

    fig.set_figwidth(1.73)  # half column width
    fig.set_figheight(1.73)  # square
    fig.subplots_adjust(left=0.35, right=0.925, bottom=0.25, top=0.825)

    set_title(fig, ax, title, True)
    set_ticklabel_sizes(fig, ax, True)
    _filtered_savefig(filter_, fig, outfile_wo_ext + '_173x173.png', dpi=100)
    set_title(fig, ax, title, False)
    set_ticklabel_sizes(fig, ax, False)
    _filtered_savefig(filter_, fig, outfile_wo_ext + '_1.73x1.73.pdf', dpi=300, transparent=True)

    fig.set_figwidth(1.73)  # half column width
    fig.set_figheight(0.972)  # 56.25% width

    fig.subplots_adjust(left=0.35, right=0.925, bottom=0.25, top=0.825)

    ax.xaxis.label.set_size(5)
    ax.xaxis.labelpad = -5
    ax.yaxis.label.set_size(5)

    set_title(fig, ax, title, True)
    set_ticklabel_sizes(fig, ax, True)
    _filtered_savefig(filter_, fig, outfile_wo_ext + '_173x97.png', dpi=100)
    set_title(fig, ax, title, False)
    set_ticklabel_sizes(fig, ax, False)
    _filtered_savefig(filter_, fig, outfile_wo_ext + '_1.73x0.97.pdf', dpi=300, transparent=True)

def fig_exists(outfile_wo_ext: str, filter_: typing.Optional[typing.Callable] = None) -> bool:
    """Returns true if the files from save_fig with the given outfile already
    exist.

    :param outfile_wo_ext: where we expect the file to have been saved
    :type outfile_wo_ext: str
    :param filter_: an optional callable which accepts images and returns True if they
        are relevant and False if they can be ignored
    :type filter_: Optional[callable]
    :return: True if the file(s) exists, False otherwise
    :rtype: bool
    """
    filter_ = filter_ if filter_ is not None else lambda x: True

    for size in ((19.2, 10.8), (7.25, 4.08), (3.54, 1.99), (1.73, 0.97), (1.73, 1.73)):
        img = outfile_wo_ext + f'_{size[0]}x{size[1]}.pdf'
        if filter_(img) and not os.path.exists(img):
            return False
        img = outfile_wo_ext + f'_{int(size[0]*100)}x{int(size[1]*100)}.png'
        if filter_(img) and not os.path.exists(img):
            return False
    return True

def make_vs_title(x: str, y: str):
    """Creates the axes title for a plot with the given x-axis variable and
    y-axis variable

    :param x: the name for the x-variable
    :type x: str
    :param y: the name for the y-variable
    :type y: str
    :return: The correct axes title
    :rtype: str
    """
    return f'{y} vs {x}'
