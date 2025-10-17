# -*- coding: utf-8 -*-

"""
CNN_bot - Bot based in CNN to play Quarto
"""

"""
Python 3
26 / 05 / 2025
@author: z_tjona

"I find that I don't understand things unless I try to program them."
-Donald E. Knuth
"""

from models.CNN1 import QuartoCNN
from models.NN_abstract import NN_abstract

from quartopy import BotAI, Piece, QuartoGame

from utils.logger import logger
import numpy as np
import torch
import os

logger.debug("Loading CNN_bot...")


class Quarto_bot(BotAI):
    @property
    def name(self) -> str:
        if hasattr(self, "model_path"):
            return f"CNN_bot|{self.model_path}"
        elif hasattr(self, "model_name"):
            return f"CNN_bot|{self.model_name}"
        else:
            return "CNN_bot|random_weights"

    def __init__(
        self,
        *,
        model_path: str | None = None,
        model: QuartoCNN | None = None,
        deterministic: bool = True,
        temperature: float = 0.1,
    ):
        """
        Initializes the CNN bot.
        ## Parameters
        ``model_path``: str | None
            Path to the pre-trained model. If None, random weights are loaded.

        ``model``: QuartoCNN | None
            An instance of QuartoCNN. If provided, it will be used instead of loading from a file.

        ``deterministic``: bool
            If True, the model will select the most probable action. Default is True.

        ``temperature``: float
            Controls the randomness of the selection. Higher values lead to more exploration.
            Only applicable if ``deterministic`` is False. Default is 0.1.

        ## Attributes
        ``DETERMINISTIC``: bool
            If True, the model will select the most probable action.
        ``TEMPERATURE``: float
            Controls the randomness of the selection. Higher values lead to more exploration. Only applicable if ``DETERMINISTIC`` is False.
        """
        super().__init__()  # aunque no hace nada
        logger.debug(f"CNN_bot initialized")
        assert (
            model_path is None or model is None
        ), "Either ``model_path`` or ``model`` must be provided, but not both."

        if model_path:
            logger.debug(f"Loading model from {model_path}")
            self.model = QuartoCNN.from_file(model_path)
            self.model_path = os.path.basename(model_path)
        elif model:
            assert isinstance(
                model, NN_abstract
            ), "Provided model must be a derived class of ``NN_abstract``."

            self.model = model
            self.model_name = model.name
            logger.debug(f"Using provided model instance {self.model_name}")

        else:
            logger.debug("Loading model with random weights")
            self.model = QuartoCNN()
        logger.debug("Model loaded successfully")

        self._recalculate = True  # Recalculate the model on each turn
        self.selected_piece: Piece
        self.board_position: tuple[int, int]
        # If True, the model will select the most probable action
        self.DETERMINISTIC: bool = deterministic

        # Controls the randomness of the selection. Higher values lead to more exploration.
        # Only applicable if ``DETERMINISTIC`` is False.
        self.TEMPERATURE: float = temperature

    # ####################################################################
    def calculate(
        self,
        game: QuartoGame,
        ith_try: int = 0,
    ):
        """Calculates the move for the bot based on the current board state and selected piece.
        ## Parameters
        ``game``: QuartoGame
            The current game instance.
        ``ith_try``: int
            The index of the current attempt to select or place a piece.
        ## Returns

        """
        if self._recalculate:
            board_matrix = game.game_board.encode()
            if isinstance(game.selected_piece, Piece):
                piece_onehot = game.selected_piece.vectorize_onehot()
                piece_onehot = piece_onehot.reshape(1, -1)  # Reshape to (1, 16)
            else:
                piece_onehot = np.zeros((1, 16), dtype=float)

            self.board_pos_onehot_cached, self.select_piece_onehot_cached = (
                self.model.predict(
                    torch.from_numpy(board_matrix).float(),
                    torch.from_numpy(piece_onehot).float(),
                    TEMPERATURE=self.TEMPERATURE,
                    DETERMINISTIC=self.DETERMINISTIC,
                )
            )
            batch_size = self.board_pos_onehot_cached.shape[0]
            assert batch_size == 1, f"Expected batch size of 1, got {batch_size}."

            self._recalculate = False  # Do not recalculate until the next turn

        # load from cached values
        _idx_piece: int = self.select_piece_onehot_cached[0, ith_try].item()  # type: ignore
        selected_piece = Piece.from_index(_idx_piece)

        _idx_board_pos: int = self.board_pos_onehot_cached[0, ith_try].item()  # type: ignore
        board_position = game.game_board.get_position_index(_idx_board_pos)

        return board_position, selected_piece

    def select(
        self,
        game: QuartoGame,
        ith_option: int = 0,
        *args,
        **kwargs,
    ) -> Piece:
        """Selects a piece for the other player."""

        _, selected_piece = self.calculate(game, ith_option)

        return selected_piece

    def place_piece(
        self,
        game: QuartoGame,
        piece: Piece,
        ith_option: int = 0,
        *args,
        **kwargs,
    ) -> tuple[int, int]:
        """Places the selected piece on the game board at a random valid position."""
        if ith_option == 0:
            self._recalculate = True
        board_position, _ = self.calculate(game, ith_option)
        return board_position
