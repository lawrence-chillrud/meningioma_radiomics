from pathlib import Path


def lsdir(path):
    """Returns a lexicographically sorted list of all the immediate subdirectories of a given path."""
    return sorted([p for p in Path(path).iterdir() if p.is_dir()])
