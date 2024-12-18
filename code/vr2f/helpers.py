"""
Helper functions

2020/10 -- Felix Klotzsche
"""

import os
from pathlib import Path


def chkmkdir(path: str):
    """
    Create dir if it does not exist yet.

    Parameters
    ----------
    path : str
        Path of the dir to be created.

    """
    dir_path = Path(path)
    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)  # Creates all parents if needed
        print(f"Created directory: {dir_path}")
    else:
        print(f"Directory already exists: {dir_path}")


def print_msg(msg):
    n_line_marks = min([len(msg) + 20, 100])
    print("\n" + n_line_marks * "#" + "\n" + msg + "\n" + n_line_marks * "#" + "\n")


def is_interactive():
    import __main__ as main

    return not hasattr(main, "__file__")