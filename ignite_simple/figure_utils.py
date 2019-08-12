"""Some utility functions related to creating good looking figures."""
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

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

    axtitle = ax.set_title(title, pad=pad)
    axtitle.set_fontsize(font_size)
    renderer = fig.canvas.get_renderer()
    bb = axtitle.get_window_extent(renderer=renderer)
    while bb.width >= (figw_px - 26) * 0.9 and font_size > 9:
        font_size = max(9, font_size - 5)
        axtitle.set_fontsize(font_size)
        bb = axtitle.get_window_extent(renderer=renderer)
    return axtitle

def make_vs_title(x: str, y: str):
    return f'{y} vs {x}'
