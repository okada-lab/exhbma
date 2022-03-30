from typing import List, Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from exhbma import ExhaustiveLinearRegression

DEFAULT_TITLESIZE = 30
DEFAULT_LABELSIZE = 24
DEFAULT_TICKLABELSIZE = 18


def feature_posterior(
    model: ExhaustiveLinearRegression,
    figsize: Optional[Tuple[float, float]] = None,
    color: str = "gray",
    title: Optional[str] = None,
    titlesize: float = DEFAULT_LABELSIZE,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    labelsize: float = DEFAULT_LABELSIZE,
    xticklabels=None,
    xrot: float = 90,
    yticklabels: List[float] = [0, 0.2, 0.4, 0.6, 0.8, 1],
    ticklabelsize: float = DEFAULT_TICKLABELSIZE,
    hlines: Optional[List[float]] = None,
):
    """
    Plot posterior probability that each feature is included in the model.

    Parameters
    ----------
    model: ExhaustiveLinearRegression
        A trained ExhaustiveLinearRegression model.
    """
    prob = model.feature_posteriors_
    index = np.arange(model.n_features_in_)

    if figsize is None:
        figsize = (len(prob), 8)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ax.bar(index, prob, color=color)

    # Set upper limits slightly over the maximum for visibility.
    ax.set_ylim([0, 1.05])

    # Set labels
    if title is not None:
        ax.set_title(title, fontsize=titlesize)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=labelsize)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=labelsize)

    # Set tick labels
    if xticklabels is not None:
        assert len(xticklabels) == model.n_features_in_
        xticks = np.arange(len(xticklabels))
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, fontsize=ticklabelsize, rotation=xrot)
    if yticklabels is not None:
        yticks = yticklabels
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels, fontsize=ticklabelsize)

    if hlines is not None:
        for hline in hlines:
            ax.axhline(hline, c="red", linestyle="--", linewidth=3)

    return fig, ax


def weight_diagram(
    model: ExhaustiveLinearRegression,
    n_top: int = 100,
    figsize: Optional[Tuple[float, float]] = None,
    column_downward: bool = True,
    cmap: str = "RdBu_r",
    vmax: Optional[float] = None,
    vmin: Optional[float] = None,
    title: Optional[str] = None,
    titlesize: float = DEFAULT_LABELSIZE,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    cbarlabel: Optional[str] = None,
    labelsize: float = DEFAULT_LABELSIZE,
    yticklabels: Optional[List[str]] = None,
    ticklabelsize: float = DEFAULT_TICKLABELSIZE,
):
    """
    Plot weight diagram of trained exhaustive search model.

    Parameters
    ----------
    model: ExhaustiveLinearRegression
        A trained ExhaustiveLinearRegression model.
    """

    coefs = np.zeros_like(model.indicators_, dtype=float)
    for i in range(len(model.models_)):
        coefs[i, np.array(model.indicators_[i]) == 1] = np.array(
            model.models_[i].coefficient
        )
    mcoefs = np.ma.masked_where(np.array(model.indicators_) == 0, coefs)

    if figsize is None:
        figsize = (24, model.n_features_in_)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    sorted_index = np.argsort(model.log_likelihoods_)[::-1][:n_top]
    if vmax is None and vmin is None:
        vrange = np.abs(coefs[sorted_index, :]).max()
        vmax, vmin = vrange, -vrange
    else:
        if vmax is None and vmin is not None:
            assert vmin < 0
            vmax = -vmin
        if vmax is not None and vmin is None:
            assert vmax > 0
            vmin = -vmax

    mesh = ax.pcolormesh(
        np.transpose(mcoefs[sorted_index])[:: 1 - 2 * int(column_downward)],
        cmap=cmap,
        vmax=vmax,
        vmin=vmin,
    )

    # Set labels
    if title is not None:
        ax.set_title(title, fontsize=titlesize)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=labelsize)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=labelsize)
    if cbarlabel is not None:
        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label(cbarlabel, fontsize=labelsize)

    # Set tick labels
    if yticklabels is not None:
        yticks = np.arange(len(yticklabels)) + 0.5
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels, fontsize=ticklabelsize)

    xticks = np.append([1], np.arange(20, n_top + 1, 20))
    xticklabels = [str(x) for x in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=ticklabelsize)

    return fig, ax


def sigma_posterior(
    model: ExhaustiveLinearRegression,
    plot_range: float = 10,
    figsize: Optional[Tuple[float, float]] = None,
    cmap=None,
    title: Optional[str] = None,
    titlesize: float = DEFAULT_LABELSIZE,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    cbarlabel: Optional[str] = None,
    labelsize: float = DEFAULT_LABELSIZE,
    xticks=None,
    xticklabels=None,
    yticks=None,
    yticklabels=None,
    ticklabelsize: float = DEFAULT_TICKLABELSIZE,
):
    """
    Plot posterior probability of hyperparameter (sigma_w, sigma_epsilon).

    Parameters
    ----------
    model: ExhaustiveLinearRegression
        A trained ExhaustiveLinearRegression model.
    """
    log_prior = np.log([p.prob for p in model.sigma_noise_points]).reshape(
        -1, 1
    ) + np.log([p.prob for p in model.sigma_coef_points]).reshape(1, -1)
    log_likelihood = np.array(model.log_likelihood_over_sigma_)
    distribution = log_prior + log_likelihood - model.log_likelihood_
    threshold = distribution.max() - plot_range
    mdistribution = np.ma.masked_where(distribution < threshold, distribution)

    if figsize is None:
        xvals = np.log10([p.position for p in model.sigma_coef_points])
        xmin, xmax = min(xvals), max(xvals)
        yvals = np.log10([p.position for p in model.sigma_noise_points])
        ymin, ymax = min(yvals), max(yvals)
        figsize = ((xmax - xmin) * 3 + 2, (ymax - ymin) * 3)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    if cmap is None:
        cmap_list = [[0, "White"], [1, "#000000"]]
        cmap = LinearSegmentedColormap.from_list("mycmap", cmap_list)

    mesh = ax.pcolormesh(mdistribution, cmap=cmap)

    # Set labels
    if title is not None:
        ax.set_title(title, fontsize=titlesize)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=labelsize)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=labelsize)
    if cbarlabel is not None:
        cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
        cbar.set_label(cbarlabel, fontsize=labelsize)

    # Set tick labels
    if xticklabels is None:
        xticks, xticklabels = _make_log_labels(
            values=[p.position for p in model.sigma_coef_points]
        )
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, fontsize=ticklabelsize)
    if yticks is None:
        yticks, yticklabels = _make_log_labels(
            values=[p.position for p in model.sigma_noise_points]
        )
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels, fontsize=ticklabelsize)

    return fig, ax


def _make_log_labels(values) -> Tuple[List[float], List[str]]:
    vmin, vmax = min(values), max(values)
    ticks, ticklabels = [], []
    for i in range(int(np.log10(vmin)) - 1, int(np.log10(vmax)) + 1):
        val = 10 ** i
        if vmin <= val <= vmax:
            fmt = f"$10^{{{i}}}$" if i not in [0, 1] else str(10 ** i)
            index = np.searchsorted(values, val)
            ticks.append(index + 0.5)
            ticklabels.append(fmt)
    return ticks, ticklabels
