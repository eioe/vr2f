"""Helper functions

2020/10 -- Felix Klotzsche
"""
import os


def chkmkdir(path: str):
    """Create dir if it does not exist yet.

    Parameters
    ----------
    path : str
        Path of the dir to be created.
    """
    if not os.path.exists(path):
        os.mkdir(path)
        print(f'Created dir: {path}')
