import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from glest.helpers import calibration_curve as calibration_curve


def grouping_diagram(
    frac_pos,
    counts,
    mean_scores,
    bins,
    ax: plt.Axes = None,
    plot_calibration: bool = True,
    plot_bins: bool = True,
    plot_cbar: bool = True,
    plot_hist: bool = True,
    plot_legend: bool = True,
    fig_kw: dict = None,
    scatter_kw: dict = None,
    calibration_kw: dict = None,
    hist_kw: dict = None,
    bin_kw: dict = None,
    legend_kw: dict = None,
):
    frac_pos = np.array(frac_pos)
    counts = np.array(counts)
    mean_scores = np.array(mean_scores)
    bins = np.array(bins)

    assert frac_pos.shape == counts.shape == mean_scores.shape
    assert bins.shape[0] == frac_pos.shape[0] + 1

    # Scatter color
    norm = Normalize(vmin=1, vmax=None)
    sm = ScalarMappable(norm=norm, cmap='flare')
    color = sm.to_rgba(counts.flat)

    # Scatter sizes
    norm = Normalize(vmin=1, vmax=100, clip=True)
    sizes = 15+ 20*norm(counts.flat)

    # Default parameters
    _fig_kw = dict(
        figsize=(3, 3),
    )
    _scatter_kw = dict(
        edgecolor='white',
        linewidth=0.3,
        color=color,
        s=sizes,
        label='Subgroups',
    )
    _calibration_kw = dict(
        marker='.',
        color='black',
        markersize=5,
        label='Calibration curve',
    )
    _hist_kw = dict(
        edgecolor='black',
        linewidth=0.2,
        color='#dfa0b3',
    )
    _bin_kw = dict(
        lw=0.2,
        ls="--",
        color="grey",
        zorder=-1,
    )
    _legend_kw = dict(
        framealpha=0,
        loc='lower center',
        bbox_to_anchor=(0.5, 1.1) if plot_hist else (0.5, 1),
        ncols=2,
    )

    # Update default parameters with input
    if calibration_kw is not None:
        _calibration_kw.update(calibration_kw)
    if hist_kw is not None:
        _hist_kw.update(hist_kw)
    if scatter_kw is not None:
        _scatter_kw.update(scatter_kw)
    if fig_kw is not None:
        _fig_kw.update(fig_kw)
    if bin_kw is not None:
        _bin_kw.update(bin_kw)
    if legend_kw is not None:
        _legend_kw.update(legend_kw)

    # Create or retrieve existing figure
    if ax is None:
        fig, ax = plt.subplots(1, 1, **_fig_kw)
    else:
        fig = ax.figure

    # Main axis
    p1 = ax.scatter(mean_scores.flat, frac_pos.flat, **_scatter_kw)

    ax.set_aspect('equal')
    ticks = [0, 0.25, 0.5, 0.75, 1]
    ax.set(
        xticks=ticks,
        yticks=ticks,
        xlabel='Predicted probability',
        ylabel='Fraction of positives',
        xlim=(-0.03, 1.03),
        ylim=(-0.03, 1.03),
    )

    if plot_bins:
        for x in bins:
            p_bin = ax.axvline(x, **_bin_kw)

    if plot_calibration:
        ax.plot([0, 1], [0, 1], ls="--", lw=1, color="black", zorder=0)
        prob_bins, mean_bins = calibration_curve(frac_pos, counts, mean_scores)
        p2, = ax.plot(mean_bins, prob_bins, **_calibration_kw)

    # Histogram on upper axis
    divider = make_axes_locatable(ax)
    if plot_hist:
        ax_hist = divider.append_axes("top", size="10%", pad=0.0)
        ax_hist.set_xlim(ax.get_xlim())
        ax_hist.get_xaxis().set_visible(False)
        ax_hist.get_yaxis().set_visible(False)
        ax_hist.spines["right"].set_visible(False)
        ax_hist.spines["top"].set_visible(False)
        ax_hist.spines["left"].set_visible(False)
        ax_hist.hist(mean_scores.flat, bins=bins, weights=counts.flat, **_hist_kw)

    # Colorbar on right axis
    if plot_cbar:
        ax_cb = divider.append_axes("right", size="4%", pad=0.05)
        ax_cb.set_title('Count', loc='left')
        Colorbar(ax_cb, mappable=sm, spacing='proportional')

    # Legend on top of the figure
    if plot_legend:
        handles_labels = {
            p1: p1.get_label(),
        }
        if plot_bins:
            handles_labels[p_bin]= 'Bin edges'
        if plot_calibration:
            handles_labels[p2] = p2.get_label()
        ax.legend(
            handles=handles_labels.keys(),
            labels=handles_labels.values(),
            **_legend_kw
        )

    return fig
