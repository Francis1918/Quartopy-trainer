# -*- coding: utf-8 -*-

"""
Python 3
17 / 09 / 2025
@author: z_tjona


"I find that I don't understand things unless I try to program them."
-Donald E. Knuth

"Either mathematics is too big for the human mind or the human mind is more than a machine."
-Kurt Godël
"""

import plotly.graph_objects as go
import plotly.colors
import numpy as np
from os import path
from datetime import datetime

# Default plotly color palette
_COLORS = plotly.colors.qualitative.Plotly


def plot_win_rate(
    *args: tuple[str | int, list[float]],
    SMOOTHING_WINDOW: int = 5,
    FREQ_EPOCH_SAVING: int = 1,
    FOLDER_SAVE: str = "./",
    FIG_NAME=lambda epoch: f"{datetime.now().strftime('%Y%m%d_%H%M')}-win_rate_{epoch:04d}.html",
    DISPLAY_PLOT: bool = False,
    fig_num: int = 1,
    position: tuple[int, int] | None = (500, 600),
    experiment_name: str = "",
):
    """Plot win rate over epochs for multiple rivals.

    Parameters
    ----------
    *args : tuple[str | int, list[float]]
        Variable number of (rival_name, win_rates) tuples to plot
    SMOOTHING_WINDOW : int
        Window size for moving average smoothing. If -1 or <=1, no smoothing is applied (default: 5)
    FREQ_EPOCH_SAVING : int
        If -1, no saving. Otherwise, save figure every n epochs (default: 1)
    FOLDER_SAVE : str
        Directory path to save figures (default: "./")
    FIG_NAME : callable
        Lambda function that generates filename given epoch number
    DISPLAY_PLOT : bool
        Whether to display the plot in the browser (default: False)
    fig_num : int
        Kept for signature compatibility (default: 1)
    position : tuple[int, int], optional
        Kept for signature compatibility
    experiment_name : str
        Experiment name to include in figure title (default: "")
    """
    fig = go.Figure()

    last_win_rates_len = 0
    for idx, (rival_name, win_rates) in enumerate(args):
        color = _COLORS[idx % len(_COLORS)]
        # Convert hex color to rgb tuple for fill
        rgb = plotly.colors.hex_to_rgb(color)

        epochs_arr = np.arange(len(win_rates))
        win_rates_arr = np.array(win_rates)
        last_win_rates_len = len(win_rates)

        # Scatter plot of raw data
        fig.add_trace(go.Scatter(
            x=epochs_arr,
            y=win_rates_arr,
            mode="markers",
            marker=dict(size=5, opacity=0.3, color=color),
            name=f"vs {rival_name}",
            legendgroup=str(rival_name),
            showlegend=True,
        ))

        if SMOOTHING_WINDOW > 1 and len(win_rates) >= SMOOTHING_WINDOW:
            smoothed = np.convolve(
                win_rates_arr,
                np.ones(SMOOTHING_WINDOW) / SMOOTHING_WINDOW,
                mode="valid",
            )
            offset = SMOOTHING_WINDOW - 1
            smoothed_epochs = np.arange(offset // 2, len(smoothed) + offset // 2)

            window_stds = []
            for i in range(len(smoothed)):
                window_data = win_rates_arr[i : i + SMOOTHING_WINDOW]
                window_stds.append(np.std(window_data))
            window_stds_arr = np.array(window_stds)

            # Smoothed line
            fig.add_trace(go.Scatter(
                x=smoothed_epochs,
                y=smoothed,
                mode="lines",
                line=dict(width=2, color=color),
                name=f"vs {rival_name} (smooth)",
                legendgroup=str(rival_name),
                showlegend=False,
            ))

            # Upper bound (invisible)
            fig.add_trace(go.Scatter(
                x=smoothed_epochs,
                y=smoothed + window_stds_arr,
                mode="lines",
                line=dict(width=0),
                legendgroup=str(rival_name),
                showlegend=False,
                hoverinfo="skip",
            ))

            # Lower bound with fill to upper
            fig.add_trace(go.Scatter(
                x=smoothed_epochs,
                y=smoothed - window_stds_arr,
                fill="tonexty",
                mode="lines",
                line=dict(width=0),
                fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.2)",
                legendgroup=str(rival_name),
                showlegend=False,
                hoverinfo="skip",
            ))

    fig.update_layout(
        title="Win rate of the epoch player vs rivals",
        xaxis_title="Training epochs",
        yaxis_title="Win rate",
        hovermode="closest",
        template="plotly_white",
        showlegend=True,
    )

    # Save the figure at regular intervals
    if last_win_rates_len % FREQ_EPOCH_SAVING == 0 and FREQ_EPOCH_SAVING != -1:
        fig.write_html(path.join(FOLDER_SAVE, FIG_NAME(last_win_rates_len)))

    if DISPLAY_PLOT:
        fig.show()


def plot_loss(
    loss_data: dict[str, list[float | int]],
    FREQ_EPOCH_SAVING: int = 200,
    FOLDER_SAVE: str = "./",
    FIG_NAME=lambda epoch: f"{datetime.now().strftime('%Y%m%d_%H%M')}-loss_{epoch:04d}.html",
    DISPLAY_PLOT: bool = False,
    fig_num: int = 2,
    position: tuple[int, int] | None = (0, 600),
    experiment_name: str = "",
):
    """
    Plot average loss per epoch with standard deviation error bands.

    Parameters
    ----------
    loss_data : dict[str, list[float | int]]
        Dictionary with 'loss_values' (list of all iteration losses) and
        'epoch_values' (list of iteration indices marking epoch boundaries)
    FREQ_EPOCH_SAVING : int
        If -1, no saving. Otherwise, save figure every n epochs (default: 200)
    FOLDER_SAVE : str
        Directory path to save figures (default: "./")
    FIG_NAME : callable
        Lambda function that generates filename given epoch number
    DISPLAY_PLOT : bool
        Whether to display the plot in the browser (default: False)
    fig_num : int
        Kept for signature compatibility (default: 2)
    position : tuple[int, int], optional
        Kept for signature compatibility
    experiment_name : str
        Experiment name to include in figure title (default: "")
    """
    epoch_values = loss_data["epoch_values"]
    loss_values = loss_data["loss_values"]

    # Calculate mean and std for each epoch
    n_epochs = len(epoch_values)
    epoch_means = []
    epoch_stds = []

    for i in range(n_epochs):
        start_idx = epoch_values[i]
        end_idx = epoch_values[i + 1] if i + 1 < n_epochs else len(loss_values)
        epoch_losses = loss_values[start_idx:end_idx]

        if len(epoch_losses) > 0:
            epoch_means.append(np.mean(epoch_losses))
            epoch_stds.append(np.std(epoch_losses))
        else:
            epoch_means.append(np.nan)
            epoch_stds.append(np.nan)

    epoch_means = np.array(epoch_means)
    epoch_stds = np.array(epoch_stds)
    epochs = np.arange(n_epochs)

    fig = go.Figure()

    # Upper bound (invisible)
    fig.add_trace(go.Scatter(
        x=epochs,
        y=epoch_means + epoch_stds,
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Lower bound with fill
    fig.add_trace(go.Scatter(
        x=epochs,
        y=epoch_means - epoch_stds,
        fill="tonexty",
        mode="lines",
        line=dict(width=0),
        fillcolor="rgba(99,110,250,0.3)",
        name="\u00b11 std dev",
    ))

    # Mean line with markers
    fig.add_trace(go.Scatter(
        x=epochs,
        y=epoch_means,
        mode="lines+markers",
        line=dict(width=2, color="#636EFA"),
        marker=dict(size=4),
        name="Mean loss",
    ))

    fig.update_layout(
        title=f"Training Loss up to epoch {n_epochs}",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        hovermode="closest",
        template="plotly_white",
    )

    # Save the figure at regular intervals
    if n_epochs % FREQ_EPOCH_SAVING == 0 and FREQ_EPOCH_SAVING != -1:
        fig.write_html(path.join(FOLDER_SAVE, FIG_NAME(n_epochs)))

    if DISPLAY_PLOT:
        fig.show()


def plot_contest_results(epochs_results: list[dict[int, dict[str, int]]]):
    """[LEGACY] Calculate win rate by rival and plot it."""

    _n_epochs = len(epochs_results)
    _n_rivals = _n_epochs

    win_rate = np.full((_n_epochs, _n_rivals), np.nan)

    for player_id, player_results in enumerate(epochs_results):
        for rival_id, result_vs_rival in player_results.items():
            _total = (
                result_vs_rival["wins"]
                + result_vs_rival["draws"]
                + result_vs_rival["losses"]
            )
            _w_rate = (
                result_vs_rival["wins"] + 0.5 * result_vs_rival["draws"]
            ) / _total

            win_rate[player_id, rival_id] = _w_rate

    fig = go.Figure(data=go.Heatmap(
        z=win_rate,
        colorscale="Viridis",
        zmin=0,
        zmax=1,
        colorbar=dict(title="Win Rate"),
    ))

    fig.update_layout(
        title="Win Rate by Epoch vs Rival",
        xaxis_title="Rival",
        yaxis_title="Epoch",
        template="plotly_white",
    )

    fig.show()
