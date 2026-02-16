# -*- coding: utf-8 -*-

"""
Python 3
04 / 12 / 2025
@author: z_tjona


"I find that I don't understand things unless I try to program them."
-Donald E. Knuth

"Either mathematics is too big for the human mind or the human mind is more than a machine."
-Kurt Godël
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.env_bootstrap import bootstrap_quartopy_path

bootstrap_quartopy_path(PROJECT_ROOT)
from quartopy import Board
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import torch
import random
from datetime import datetime
from os import path


def _board_piece_index_grid(board: Board) -> np.ndarray:
    """
    Convert a quartopy Board to a 4x4 grid of piece indices.
    Empty cells are represented as -1.
    """
    if hasattr(board, "to_matrix"):
        m = np.array(board.to_matrix())
    else:
        m = np.array(board.encode())

    # Expected one-hot layout: (1, 16, 4, 4) or (16, 4, 4)
    if m.ndim == 4 and m.shape[0] == 1:
        m = m[0]

    if m.ndim == 3 and m.shape[0] == 16:
        occupied = m.sum(axis=0) > 0
        piece_idx = m.argmax(axis=0)
        return np.where(occupied, piece_idx, -1)

    # Fallback: already a 4x4 grid
    if m.ndim == 2 and m.shape == (4, 4):
        return m.astype(int)

    # Unknown shape fallback
    return np.full((4, 4), -1, dtype=int)


def _add_board_trace(fig: go.Figure, board: Board, row: int, col: int) -> None:
    """Add a board visualization to a subplot using Plotly heatmap + annotations."""
    grid = _board_piece_index_grid(board)
    occupancy = (grid >= 0).astype(float)

    fig.add_trace(
        go.Heatmap(
            z=occupancy[::-1],  # Flip so row 0 is at top
            colorscale=[[0, "white"], [1, "lightgray"]],
            showscale=False,
            hoverinfo="skip",
        ),
        row=row,
        col=col,
    )

    # Add text annotations for piece indices
    for r in range(4):
        for c in range(4):
            v = grid[r, c]
            label = "." if v < 0 else str(int(v))
            color = "gray" if v < 0 else "red"
            fig.add_annotation(
                x=c,
                y=3 - r,  # Flip y to match visual
                text=label,
                showarrow=False,
                font=dict(size=10, color=color),
                xref=f"x{col if col > 1 else ''}",
                yref=f"y{((row - 1) * fig._grid_ref[0].__len__() + col) if row > 1 or col > 1 else ''}",
            )


def plot_boards_comp(
    *boards_pair: tuple[Board, Board],
    q_place: torch.Tensor,
    q_select: torch.Tensor,
    fig_num: int = 3,
    DISPLAY_PLOT: bool = True,
    MAX_BOARDS: int = 6,
    position: tuple[int, int] | None = (500, 0),
    experiment_name: str = "",
    FREQ_EPOCH_SAVING: int = -1,
    FOLDER_SAVE: str = "./",
    FIG_NAME=lambda epoch: f"{datetime.now().strftime('%Y%m%d_%H%M')}-boards_comp_{epoch:04d}.html",
    current_epoch: int = 0,
) -> None:
    """Plot pairs of boards side by side in a 2xn subplot grid (transposed).

    Parameters
    ----------
    *boards_pair : tuple[Board, Board]
        Variable number of board pairs to compare. Typically (state, next_state).
    q_place : torch.Tensor
        Q-values for placement actions
    q_select : torch.Tensor
        Q-values for selection actions
    fig_num : int
        Kept for signature compatibility (default: 3)
    DISPLAY_PLOT : bool
        Whether to display the plot in the browser (default: True)
    MAX_BOARDS : int
        Maximum number of board pairs to display. If more pairs provided,
        randomly samples MAX_BOARDS pairs (default: 6)
    position : tuple[int, int], optional
        Kept for signature compatibility
    experiment_name : str
        Experiment name to include in figure title (default: "")
    FREQ_EPOCH_SAVING : int
        If -1, no saving. Otherwise, save figure every n epochs (default: -1)
    FOLDER_SAVE : str
        Directory path to save figures (default: "./")
    FIG_NAME : callable
        Lambda function that generates filename given epoch number
    current_epoch : int
        Current epoch number for saving (default: 0)
    """
    n = len(boards_pair)
    if n == 0:
        return

    # Limit to MAX_BOARDS random samples
    if n > MAX_BOARDS:
        indices = random.sample(range(n), MAX_BOARDS)
        boards_pair = tuple(boards_pair[i] for i in sorted(indices))
        n = MAX_BOARDS

    # Build subplot titles
    subplot_titles = []
    for i, (b1, b2) in enumerate(boards_pair):
        t1 = getattr(b1, "name", f"state_{i}")
        t2 = getattr(b2, "name", f"next_state_{i}")
        subplot_titles.extend([t1, t2])

    # Reorder titles: all row1 first, then all row2
    titles_row1 = [subplot_titles[i * 2] for i in range(n)]
    titles_row2 = [subplot_titles[i * 2 + 1] for i in range(n)]

    fig = make_subplots(
        rows=2,
        cols=n,
        subplot_titles=titles_row1 + titles_row2,
        vertical_spacing=0.15,
        horizontal_spacing=0.05,
    )

    for i, (b1, b2) in enumerate(boards_pair):
        grid1 = _board_piece_index_grid(b1)
        occupancy1 = (grid1 >= 0).astype(float)
        grid2 = _board_piece_index_grid(b2)
        occupancy2 = (grid2 >= 0).astype(float)

        fig.add_trace(
            go.Heatmap(
                z=occupancy1[::-1],
                colorscale=[[0, "white"], [1, "lightgray"]],
                showscale=False,
                hoverinfo="skip",
            ),
            row=1,
            col=i + 1,
        )

        fig.add_trace(
            go.Heatmap(
                z=occupancy2[::-1],
                colorscale=[[0, "white"], [1, "lightgray"]],
                showscale=False,
                hoverinfo="skip",
            ),
            row=2,
            col=i + 1,
        )

        # Add piece index annotations
        for r in range(4):
            for c in range(4):
                for row_idx, grid in [(1, grid1), (2, grid2)]:
                    v = grid[r, c]
                    label = "." if v < 0 else str(int(v))
                    color = "gray" if v < 0 else "red"

                    # Calculate axis reference names
                    ax_idx = (row_idx - 1) * n + (i + 1)
                    xref = "x" if ax_idx == 1 else f"x{ax_idx}"
                    yref = "y" if ax_idx == 1 else f"y{ax_idx}"

                    fig.add_annotation(
                        x=c,
                        y=3 - r,
                        text=label,
                        showarrow=False,
                        font=dict(size=10, color=color),
                        xref=xref,
                        yref=yref,
                    )

    # Hide axis ticks on all subplots
    fig.update_xaxes(showticklabels=False, showgrid=False)
    fig.update_yaxes(showticklabels=False, showgrid=False)

    fig.update_layout(
        title_text=f"Board Comparisons - Epoch {current_epoch}",
        showlegend=False,
        height=600,
        width=max(300 * n, 600),
        template="plotly_white",
    )

    # Save the figure at regular intervals
    if current_epoch % FREQ_EPOCH_SAVING == 0 and FREQ_EPOCH_SAVING != -1:
        fig.write_html(path.join(FOLDER_SAVE, FIG_NAME(current_epoch)))

    if DISPLAY_PLOT:
        fig.show()


def plot_Qv_progress(
    q_values_history: dict[str, list[torch.Tensor]],
    rewards: torch.Tensor,
    fig_num: int = 4,
    DISPLAY_PLOT: bool = True,
    done_v: torch.Tensor | None = None,
    PLOT_TYPE: str = "time_series",
    position: tuple[int, int] | None = (0, 0),
    experiment_name: str = "",
    FREQ_EPOCH_SAVING: int = -1,
    FOLDER_SAVE: str = "./",
    FIG_NAME=lambda epoch: f"{datetime.now().strftime('%Y%m%d_%H%M')}-qv_progress_{epoch:04d}.html",
    current_epoch: int = 0,
) -> None:
    """Plot Q-value progression over epochs for each sample in the batch.

    Parameters
    ----------
    q_values_history : dict[str, list[torch.Tensor]]
        Dictionary with keys 'q_place' and 'q_select', each containing a list of
        tensors (one per epoch) with Q-values for each sample
    rewards : torch.Tensor
        Target rewards for each sample (batch_size,)
    fig_num : int
        Kept for signature compatibility (default: 4)
    DISPLAY_PLOT : bool
        Whether to display the plot in the browser (default: True)
    done_v : torch.Tensor, optional
        Boolean tensor indicating whether each sample is a terminal state (batch_size,).
        Terminal states are plotted with higher prominence (thicker, more opaque lines).
    PLOT_TYPE : str
        Plot type: "time_series" or "hist" (default: "time_series")
    position : tuple[int, int], optional
        Kept for signature compatibility
    experiment_name : str
        Experiment name to include in figure title (default: "")
    FREQ_EPOCH_SAVING : int
        If -1, no saving. Otherwise, save figure every n epochs (default: -1)
    FOLDER_SAVE : str
        Directory path to save figures (default: "./")
    FIG_NAME : callable
        Lambda function that generates filename given epoch number
    current_epoch : int
        Current epoch number for saving (default: 0)
    """
    if not q_values_history or len(q_values_history.get("q_place", [])) == 0:
        return

    q_place_history = q_values_history.get("q_place", [])
    q_select_history = q_values_history.get("q_select", [])

    batch_size = q_place_history[0].shape[0] if q_place_history else 0
    n_epochs = len(q_place_history)

    if batch_size == 0:
        return

    epochs = np.arange(n_epochs)

    # Split samples by reward value (round to handle decimal rewards)
    loss_indices = [i for i in range(batch_size) if round(rewards[i].item()) == -1]
    draw_indices = [i for i in range(batch_size) if round(rewards[i].item()) == 0]
    win_indices = [i for i in range(batch_size) if round(rewards[i].item()) == 1]

    subplot_titles = [
        "Q_place: R=-1", "Q_place: R=0", "Q_place: R=1",
        "Q_select: R=-1", "Q_select: R=0", "Q_select: R=1",
    ]

    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=subplot_titles,
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    # Define plot configurations: (row, col, indices, q_history, title)
    plot_configs = [
        (1, 1, loss_indices, q_place_history, "Q_place: R=-1"),
        (1, 2, draw_indices, q_place_history, "Q_place: R=0"),
        (1, 3, win_indices, q_place_history, "Q_place: R=1"),
        (2, 1, loss_indices, q_select_history, "Q_select: R=-1"),
        (2, 2, draw_indices, q_select_history, "Q_select: R=0"),
        (2, 3, win_indices, q_select_history, "Q_select: R=1"),
    ]

    if PLOT_TYPE == "time_series":
        for row, col, indices, q_history, title in plot_configs:
            for i in indices:
                q_sample = [q[i].item() for q in q_history]
                is_terminal = done_v[i].item() if done_v is not None else False

                fig.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=q_sample,
                        mode="lines",
                        line=dict(
                            width=1.5 if is_terminal else 0.5,
                        ),
                        opacity=0.3 if is_terminal else 0.15,
                        showlegend=False,
                        hoverinfo="y",
                    ),
                    row=row,
                    col=col,
                )

            # Update axes
            if row == 2:
                fig.update_xaxes(title_text="Epoch", row=row, col=col)
            if col == 1:
                fig.update_yaxes(title_text="Q-value", row=row, col=col)
            fig.update_yaxes(range=[-1.1, 1.1], row=row, col=col)

    elif PLOT_TYPE == "hist":
        HIST_BINS = 50
        HIST_RANGE = (-1.1, 1.1)

        for row, col, indices, q_history, title in plot_configs:
            if not indices:
                # Add placeholder annotation for empty groups
                ax_idx = (row - 1) * 3 + col
                xref = "x" if ax_idx == 1 else f"x{ax_idx}"
                yref = "y" if ax_idx == 1 else f"y{ax_idx}"
                fig.add_annotation(
                    text="No samples",
                    xref=xref,
                    yref=yref,
                    x=0.5,
                    y=0,
                    showarrow=False,
                )
                continue

            # Compute histograms for this reward group across epochs
            hist_data = []
            for q_epoch in q_history:
                q_subset = q_epoch[indices].detach().cpu().numpy().flatten()
                hist, _ = np.histogram(q_subset, bins=HIST_BINS, range=HIST_RANGE)
                hist_percent = (hist / hist.sum()) * 100 if hist.sum() > 0 else hist
                hist_data.append(hist_percent)

            if hist_data:
                hist_array = np.array(hist_data)
                bin_centers = np.linspace(HIST_RANGE[0], HIST_RANGE[1], HIST_BINS)

                fig.add_trace(
                    go.Heatmap(
                        z=hist_array.T,
                        x=epochs,
                        y=bin_centers,
                        colorscale="Viridis",
                        showscale=(row == 1 and col == 3),
                        colorbar=dict(title="% ", len=0.4, y=0.78) if (row == 1 and col == 3) else None,
                    ),
                    row=row,
                    col=col,
                )

            # Update axes
            if row == 2:
                fig.update_xaxes(title_text="Epoch", row=row, col=col)
            if col == 1:
                fig.update_yaxes(title_text="Q-value", row=row, col=col)

    fig.update_layout(
        title_text=f"Q-value Progress - Epoch {current_epoch}",
        height=800,
        width=1200,
        showlegend=False,
        template="plotly_white",
    )

    # Save the figure at regular intervals
    if current_epoch % FREQ_EPOCH_SAVING == 0 and FREQ_EPOCH_SAVING != -1:
        fig.write_html(path.join(FOLDER_SAVE, FIG_NAME(current_epoch)))

    if DISPLAY_PLOT:
        fig.show()
