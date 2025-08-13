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


def format_float(x, is_pval=False, rm_leading_zero=False):
    """
    Format a float as a string, with optional p-value formatting or leading zero removal.

    Parameters
    ----------
    x : float
        Number to format.
    is_pval : bool, optional
        If True, format as a p-value:
        - Returns "< .001" if `x` < 0.001.
        - Otherwise returns "= " followed by the value rounded to 3 decimal
          places, with any leading zero removed.
        Default is False.
    rm_leading_zero : bool, optional
        If True and `is_pval` is False, remove the leading zero from the
        formatted number (e.g., "0.25" becomes ".25").
        Default is False.

    Returns
    -------
    str
        Formatted number as a string.
    """
    
    if is_pval:
        string = "< .001" if x < 0.001 else "= " + f"{x:.3f}".lstrip("0")
        return string
    else:
        string = f"{x:.2f}"
        if rm_leading_zero:
            string = string.lstrip("0")
        return string