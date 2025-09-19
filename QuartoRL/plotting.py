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

import matplotlib.pyplot as plt
import numpy as np

plt.ion()  # Enable interactive mode


def plot_contest_results(epochs_results: list[dict[int, dict[str, int]]]):
    """Calculate win rate by rival and plot it."""

    _n_epochs = len(epochs_results)
    _n_rivals = _n_epochs
    # Assuming rivals are from previous epochs

    # Win rate by epoch and rival
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

    plt.figure(1, figsize=(8, 6), clear=True)
    im = plt.imshow(win_rate, aspect="auto", interpolation="none", cmap="viridis")
    plt.colorbar(im, label="Win Rate", extend="neither")
    im.set_clim(0, 1)
    plt.xlabel("Rival")
    plt.ylabel("Epoch")
    # plt.xticks(ticks=range(_n_rivals))
    # plt.yticks(ticks=range(_n_epochs))
    plt.title("Win Rate by Epoch vs Rival")
    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)
