import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


def grouping_diagram(
    c_hat,
    r_hat,
    n_in_leaf,
    f,
    leaf_ids,
    groups="all",
    ax: plt.Axes = None,
    plot_calibration=True,
    plot_cbar: bool = True,
    plot_hist: bool = True,
    plot_legend: bool = True,
):
    """
    Plot a grouping diagram for residuals.
    Parameters
    ----------
    c_hat : array-like
        Predicted probabilities.
    r_hat : array-like
        Predicted residuals.
    n_in_leaf : array-like
        Number of samples in each leaf.
    f : callable
        Function to compute the grouping diagram.
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the grouping diagram.
    """
    # Scatter color
    norm = Normalize(vmin=1, vmax=None)
    sm = ScalarMappable(norm=norm, cmap="viridis")
    color = sm.to_rgba(leaf_ids.flat)

    # Scatter sizes
    norm = Normalize(vmin=1, vmax=100, clip=True)

    # Default parameters
    _fig_kw = dict(
        figsize=(4, 4),
    )
    _scatter_kw = dict(
        edgecolor="white",
        linewidth=0.3,
        color=color,
        label="Subgroups",
        alpha=0.5,
    )
    _calibration_kw = dict(
        color="black",
        linewidth=5,
        label="Calibration curve",
        zorder=3,
    )
    _hist_kw = dict(
        edgecolor="black",
        linewidth=0.2,
        color="#dfa0b3",
    )
    _bin_kw = dict(
        lw=0.2,
        ls="--",
        color="grey",
        zorder=-1,
    )
    _legend_kw = dict(
        framealpha=0,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.1) if plot_hist else (0.5, 1),
        ncols=2,
    )

    # Update default parameters with input# Create or retrieve existing figure
    if ax is None:
        fig, ax = plt.subplots(1, 1, **_fig_kw)
    else:
        fig = ax.figure

    f_star_hat = r_hat + c_hat

    ax.set_aspect("equal")
    ticks = [0, 0.25, 0.5, 0.75, 1]
    ax.set(
        xticks=ticks,
        yticks=ticks,
        xlabel="Predicted probability",
        ylabel="Fraction of positives",
        xlim=(0, 1.0),
        ylim=(0, 1.0),
    )
    ax.xaxis.label.set_fontsize(16)
    ax.yaxis.label.set_fontsize(16)

    divider = make_axes_locatable(ax)
    # Colorbar on right axis
    if plot_cbar:
        ax_cb = divider.append_axes("right", size="4%", pad=0.05)
        ax_cb.set_title("Group", loc="left")
        Colorbar(ax_cb, mappable=sm, spacing="proportional")
    legend_handles = []
    legend_labels = []
    if plot_calibration:
        ax.plot([0, 1], [0, 1], ls="--", lw=1, color="black", zorder=0)
        # prob_bins, mean_bins = calibration_curve(y, f, n_bins=100)
        sort_idx = np.argsort(f)
        (line,) = ax.plot(f[sort_idx], c_hat[sort_idx], **_calibration_kw)
        legend_handles.append(line)
        legend_labels.append("Calibration curve")

    if groups == "all" or groups is None:
        for i, leaf in enumerate(np.unique(leaf_ids)):
            mask = leaf_ids == leaf
            n_leaf = n_in_leaf[i]
            if np.sum(mask) > 0:
                # Sort by f values to create a proper curve
                f_leaf = f[mask]
                f_star_hat_leaf = f_star_hat[mask]
                sort_idx = np.argsort(f_leaf)
                # Get the color for this leaf from the colormap
                leaf_color = sm.to_rgba(leaf)
                # Make line width proportional to number of samples
                line_width = +3 * (n_leaf / np.max(n_in_leaf))
                ax.plot(
                    f_leaf[sort_idx],
                    f_star_hat_leaf[sort_idx],
                    color=leaf_color,
                    alpha=0.7,
                    linewidth=line_width,
                )
    else:
        for i, leaf in enumerate(np.unique(leaf_ids)):
            mask = leaf_ids == leaf
            n_leaf = n_in_leaf[i]
            if np.sum(mask) > 0:
                f_leaf = f[mask]
                f_star_hat_leaf = f_star_hat[mask]
                sort_idx = np.argsort(f_leaf)

            if leaf in groups:
                # Plot selected groups with color and add to legend
                leaf_color = sm.to_rgba(leaf)
                line_width = 1 + 5 * (n_leaf / np.max(n_in_leaf))
                (line,) = ax.plot(
                    f_leaf[sort_idx],
                    f_star_hat_leaf[sort_idx],
                    color=leaf_color,
                    alpha=0.7,
                    linewidth=line_width,
                    zorder=2,
                )
                legend_handles.append(line)
                legend_labels.append(f"Group {leaf}")
            else:
                # Plot non-selected groups in grey (background)
                line_width = +3 * (n_leaf / np.max(n_in_leaf))
                ax.plot(
                    f_leaf[sort_idx],
                    f_star_hat_leaf[sort_idx],
                    color="grey",
                    alpha=0.3,
                    linewidth=line_width,
                    zorder=1,
                )

    if legend_handles and plot_legend:
        ax.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),
            ncols=2,
            framealpha=0,
        )

    return fig
