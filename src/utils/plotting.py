import matplotlib
from matplotlib import cm, gridspec
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import numpy as np

# BASE_DIR = Path("/src/")
# STYLE = BASE_DIR / "src" / "utils" / "dissemination.mplstyle"
# matplotlib.style.use(STYLE)
from pathlib import Path

BASE_DIR = Path(__file__).parent
STYLE = BASE_DIR / "dissemination.mplstyle"

# Constants
TEXTWIDTH = 9.0  # inches
FONTSIZE = 9.0
CMAP = cm.plasma
CMAP_R = cm.plasma_r
COLORS = [CMAP(i / 4.0) for i in range(5)]
COLORS_NEUTRAL = ["0.0", "0.4", "0.7", "1.0"]
STYLE_SETTINGS = ["dark_background", "presentation"]

# Apply default rcParams for consistent font sizes
def setup():
    """Apply general Matplotlib style and settings."""
    matplotlib.rcParams.update({
        "font.size": FONTSIZE,
        "axes.titlesize": FONTSIZE,
        "axes.labelsize": FONTSIZE,
        "xtick.labelsize": FONTSIZE,
        "ytick.labelsize": FONTSIZE,
        "legend.fontsize": FONTSIZE,
        "figure.titlesize": FONTSIZE,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })

# Helper function for margin calculation
def _calculate_margins(large_margin, small_margin, left_margin_large, right_margin_large, top_margin_large, bottom_margin_large):
    """Calculate margins based on input flags."""
    margins = {
        "left": large_margin if left_margin_large else small_margin,
        "right": large_margin if right_margin_large else small_margin,
        "top": large_margin if top_margin_large else small_margin,
        "bottom": large_margin if bottom_margin_large else small_margin
    }
    return margins

# Helper function to calculate figure size based on margins
def _calculate_figure_size(width, height, margins, aspect_ratio=1.0):
    """Calculate figure size with margins applied."""
    left, right, top, bottom = margins.values()

    if width is not None:
        height = width / aspect_ratio
    elif height is not None:
        width = height * aspect_ratio

    return width, height

# Create figure by specifying height
def figure_by_height(
    height=TEXTWIDTH * 0.5,
    large_margin=0.14,
    small_margin=0.03,
    make3d=False,
    left_margin_large=True,
    right_margin_large=False,
    bottom_margin_large=True,
    top_margin_large=False,
):
    """Create a figure, size specified by height."""
    margins = _calculate_margins(large_margin, small_margin, left_margin_large, right_margin_large, top_margin_large, bottom_margin_large)
    width, height = _calculate_figure_size(None, height, margins)

    fig = plt.figure(figsize=(width, height))
    ax = Axes3D(fig) if make3d else plt.gca()
    
    plt.subplots_adjust(
        left=margins['left'],
        right=1.0 - margins['right'],
        bottom=margins['bottom'],
        top=1.0 - margins['top'],
        wspace=0.0,
        hspace=0.0,
    )

    return fig, ax

# Create figure by specifying width
def figure_by_width(
    width=TEXTWIDTH * 0.5,
    large_margin=0.14,
    small_margin=0.03,
    make3d=False,
    left_margin_large=True,
    right_margin_large=False,
    bottom_margin_large=True,
    top_margin_large=False,
):
    """Create a figure, size specified by width."""
    margins = _calculate_margins(large_margin, small_margin, left_margin_large, right_margin_large, top_margin_large, bottom_margin_large)
    width, height = _calculate_figure_size(width, None, margins)

    fig = plt.figure(figsize=(width, height))
    ax = Axes3D(fig) if make3d else plt.gca()

    plt.subplots_adjust(
        left=margins['left'],
        right=1.0 - margins['right'],
        bottom=margins['bottom'],
        top=1.0 - margins['top'],
        wspace=0.0,
        hspace=0.0,
    )

    return fig, ax

# Create figure with colorbar by specifying height
def figure_with_cbar_by_height(
    height=TEXTWIDTH * 0.5,
    large_margin=0.14,
    small_margin=0.03,
    cbar_sep=0.02,
    cbar_width=0.04,
    make3d=False,
    left_margin_large=True,
    right_margin_large=False,
    bottom_margin_large=True,
    top_margin_large=False,
):
    """Create a figure with colorbar, size specified by height."""
    margins = _calculate_margins(large_margin, small_margin, left_margin_large, right_margin_large, top_margin_large, bottom_margin_large)
    width, height = _calculate_figure_size(None, height, margins)

    right = margins["right"] + cbar_width + cbar_sep
    cleft = 1.0 - (large_margin + cbar_width) * height / width
    cbottom = margins["bottom"]
    cwidth = cbar_width * height / width
    cheight = 1.0 - margins["top"] - margins["bottom"]

    fig = plt.figure(figsize=(width, height))
    ax = Axes3D(fig) if make3d else plt.gca()
    
    plt.subplots_adjust(
        left=margins['left'] * height / width,
        right=1.0 - right * height / width,
        bottom=margins['bottom'],
        top=1.0 - margins['top'],
        wspace=0.0,
        hspace=0.0,
    )
    cax = fig.add_axes([cleft, cbottom, cwidth, cheight])

    plt.sca(ax)

    return fig, (ax, cax)

# Grid of panels, no colorbars, size specified by height
def grid_by_height(
    nx=4,
    ny=2,
    height=0.5 * TEXTWIDTH,
    aspect_ratio=1.0,
    large_margin=0.14,
    small_margin=0.03,
    sep=0.02,
    left_margin_large=True,
    right_margin_large=False,
    bottom_margin_large=True,
    top_margin_large=False,
):
    """Create a grid of panels with consistent margins."""
    margins = _calculate_margins(large_margin, small_margin, left_margin_large, right_margin_large, top_margin_large, bottom_margin_large)
    
    panel_size = (1.0 - margins['top'] - margins['bottom'] - (ny - 1) * sep) / ny
    width = height * aspect_ratio * (margins['left'] + nx * panel_size + (nx - 1) * sep + margins['right'])
    avg_width_abs = (height * panel_size * nx * ny) / (nx * ny + ny)
    avg_height_abs = height * panel_size
    wspace = sep * height / avg_width_abs
    hspace = sep * height / avg_height_abs

    fig = plt.figure(figsize=(width, height))
    gs = gridspec.GridSpec(ny, nx, width_ratios=[1.0] * nx, height_ratios=[1.0] * ny)
    plt.subplots_adjust(
        left=margins['left'] * height / width,
        right=1.0 - margins['right'] * height / width,
        bottom=margins['bottom'],
        top=1.0 - margins['top'],
        wspace=wspace,
        hspace=hspace,
    )
    return fig, gs

def grid_by_width(
    nx=4,
    ny=2,
    width=TEXTWIDTH,
    aspect_ratio=1.0,
    large_margin=0.14,
    small_margin=0.03,
    sep=0.02,
    left_margin_large=True,
    right_margin_large=False,
    bottom_margin_large=True,
    top_margin_large=False,
):
    """Grid of panels, no colorbars, size specified by width."""

    left = large_margin if left_margin_large else small_margin
    right = large_margin if right_margin_large else small_margin
    top = large_margin if top_margin_large else small_margin
    bottom = large_margin if bottom_margin_large else small_margin

    # Panel size calculation
    panel_size = (1.0 - top - bottom - (ny - 1) * sep) / ny
    height = width / (left + nx * panel_size + (nx - 1) * sep + right) / aspect_ratio

    # wspace and hspace calculation for optimal spacing
    avg_width_abs = (height * panel_size * nx * ny) / (nx * ny + ny)
    avg_height_abs = height * panel_size
    wspace = sep * height / avg_width_abs
    hspace = sep * height / avg_height_abs

    # Set up figure and adjust layout
    fig = plt.figure(figsize=(width, height))
    gs = gridspec.GridSpec(ny, nx, width_ratios=[1.0] * nx, height_ratios=[1.0] * ny)
    plt.subplots_adjust(
        left=left * height / width,
        right=1.0 - right * height / width,
        bottom=bottom,
        top=1.0 - top,
        wspace=wspace,
        hspace=hspace,
    )
    return fig, gs

# Add scatter plot with consistent styling
def plot_scatter(xs, ys, **scatter_kwargs):
    """Plot a scatter plot with consistent styling."""
    defaults = {"alpha": 0.6, "lw": 3, "s": 80, "color": "C0", "facecolors": "none", "marker": "."}
    scatter_kwargs = {**defaults, **scatter_kwargs}
    plt.scatter(xs, ys, **scatter_kwargs)

# Plot a line with consistent styling
def plot_line(xs, ys, **plot_kwargs):
    """Plot a line with consistent styling."""
    plot_kwargs["linewidth"] = plot_kwargs.get("linewidth", 3)
    background_plot_kwargs = {**plot_kwargs, "linewidth": plot_kwargs["linewidth"] + 2, "color": "white"}
    del background_plot_kwargs["label"]

    plt.plot(xs, ys, **background_plot_kwargs, zorder=30)
    plt.plot(xs, ys, **plot_kwargs, zorder=31)

# Plot error bars (vertical)
def plot_errorbar(xs, ys, error_lower, error_upper, colors="C0", error_width=12, alpha=0.3):
    """Plot vertical error bars with consistent styling."""
    colors = [colors] * len(xs) if isinstance(colors, str) else colors
    for ii, (x, y, err_l, err_u) in enumerate(zip(xs, ys, error_lower, error_upper)):
        marker, _, bar = plt.errorbar(
            x=x, y=y, yerr=np.array((err_l, err_u))[:, None], ls="none", color=colors[ii], zorder=1
        )
        plt.setp(bar[0], capstyle="round")
        marker.set_fillstyle("none")
        bar[0].set_alpha(alpha)
        bar[0].set_linewidth(error_width)

# Plot error bars (horizontal)
def plot_x_errorbar(xs, ys, error_lower, error_upper, colors="C0", error_width=12, alpha=0.3):
    """Plot horizontal error bars with consistent styling."""
    colors = [colors] * len(xs) if isinstance(colors, str) else colors
    for ii, (x, y, err_l, err_u) in enumerate(zip(xs, ys, error_lower, error_upper)):
        marker, _, bar = plt.errorbar(
            x=x, y=y, xerr=np.array((err_l, err_u))[:, None], ls="none", color=colors[ii], zorder=1
        )
        plt.setp(bar[0], capstyle="round")
        marker.set_fillstyle("none")
        bar[0].set_alpha(alpha)
        bar[0].set_linewidth(error_width)
