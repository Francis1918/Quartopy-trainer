from contextlib import contextmanager
import inspect
from typing import Any


def _to_win_rate_counts(results: Any) -> dict[str, int]:
    if isinstance(results, dict):
        if {"Player 1", "Player 2", "Tie"}.issubset(results.keys()):
            return {
                "Player 1": int(results["Player 1"]),
                "Player 2": int(results["Player 2"]),
                "Tie": int(results["Tie"]),
            }
        if {"P1", "P2", "Empates"}.issubset(results.keys()):
            return {
                "Player 1": int(results["P1"]),
                "Player 2": int(results["P2"]),
                "Tie": int(results["Empates"]),
            }
        # Mapping path -> outcome (+1 / -1 / 0)
        p1 = p2 = tie = 0
        for value in results.values():
            if value == 1:
                p1 += 1
            elif value == -1:
                p2 += 1
            else:
                tie += 1
        return {"Player 1": p1, "Player 2": p2, "Tie": tie}

    if isinstance(results, list):
        p1 = p2 = tie = 0
        for item in results:
            if isinstance(item, dict):
                r = item.get("result")
                if r == "Player 1":
                    p1 += 1
                elif r == "Player 2":
                    p2 += 1
                else:
                    tie += 1
        return {"Player 1": p1, "Player 2": p2, "Tie": tie}

    return {"Player 1": 0, "Player 2": 0, "Tie": 0}


@contextmanager
def _temporary_disable_export(disable: bool):
    if not disable:
        yield
        return

    from quartopy import QuartoGame

    original = QuartoGame.export_history_to_csv

    def _no_export(self, output_folder: str, match_number: int):
        # Keep return type compatible with quartopy.play_games internals
        return f"in_memory_match_{match_number:06d}.csv"

    QuartoGame.export_history_to_csv = _no_export
    try:
        yield
    finally:
        QuartoGame.export_history_to_csv = original


def play_games_compat(
    *,
    matches: int,
    player1: Any,
    player2: Any,
    delay: float = 0,
    verbose: bool = False,
    save_match: bool = False,
    mode_2x2: bool = False,
    PROGRESS_MESSAGE: str = "Playing matches...",
    return_match_data: bool = False,
    match_dir: str = "./partidas_guardadas/",
) -> tuple[Any, dict[str, int]]:
    """
    Compatibility wrapper for quartopy.play_games across API versions.
    Returns `(match_payload, win_rate_counts)` where win_rate keys are:
    "Player 1", "Player 2", "Tie".
    """
    from quartopy import play_games

    sig = inspect.signature(play_games)
    params = sig.parameters

    # Old API: supports save_match/mode_2x2 and typically returns (data, win_rate)
    if "save_match" in params:
        kwargs: dict[str, Any] = {
            "matches": matches,
            "player1": player1,
            "player2": player2,
            "delay": delay,
            "verbose": verbose,
            "save_match": save_match,
            "PROGRESS_MESSAGE": PROGRESS_MESSAGE,
        }
        if "mode_2x2" in params:
            kwargs["mode_2x2"] = mode_2x2
        if "match_dir" in params:
            kwargs["match_dir"] = match_dir

        result = play_games(**kwargs)

        if isinstance(result, tuple) and len(result) == 2:
            payload, win_rate = result
            return payload, _to_win_rate_counts(win_rate)

        payload = result
        return payload, _to_win_rate_counts(result)

    # New API: no save_match/mode_2x2 and always writes csv by default.
    kwargs = {
        "matches": matches,
        "player1": player1,
        "player2": player2,
        "delay": delay,
        "verbose": verbose,
        "PROGRESS_MESSAGE": PROGRESS_MESSAGE,
    }
    if "match_dir" in params:
        kwargs["match_dir"] = match_dir
    if "return_match_data" in params:
        kwargs["return_match_data"] = return_match_data
    if "return_file_paths" in params:
        # Request compact summary dict when not asking match data
        kwargs["return_file_paths"] = False

    with _temporary_disable_export(disable=(not save_match)):
        payload = play_games(**kwargs)

    win_rate = _to_win_rate_counts(payload)
    return payload, win_rate
