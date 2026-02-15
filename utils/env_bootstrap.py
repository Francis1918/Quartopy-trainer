from pathlib import Path
import os
import sys

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


def _resolve_project_root(project_root: Path | None = None) -> Path:
    if project_root is not None:
        return project_root.resolve()
    return Path(__file__).resolve().parent.parent


def bootstrap_quartopy_path(project_root: Path | None = None) -> str | None:
    root = _resolve_project_root(project_root)
    env_file = root / ".env"

    if load_dotenv is not None and env_file.exists():
        load_dotenv(env_file)

    quartopy_path = os.getenv("QUARTOPY_PATH")
    candidates: list[Path] = []
    if quartopy_path:
        candidates.append(Path(quartopy_path).expanduser())

    candidates.append(root.parent / "Quartopy")

    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return candidate_str

    return None
