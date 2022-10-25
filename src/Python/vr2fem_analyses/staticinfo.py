"""Store global variables

Keep paths, colors, etc. to access them throughout the project.
"""
import os
import sys
from pathlib import Path

MAIN_PATH = ''

class VR2FEMPATHS:
    """Store relevant paths for the project.
    """
    def __init__(self) -> None:
        self.PATH_PROJECT = Path(__file__).parents[4]
        if self.PATH_PROJECT.parts[-1] != 'vrstereofem':
            raise ValueError('staticinfo seems to be in the wrong folder. \
                              Cannot point to project folder.')
        self.DATA_SUBJECTS = Path(self.PATH_PROJECT, 'Data', 'Subjects')
        self.DATA_00_RAWFIF = Path(self.PATH_PROJECT, 'Data', 'raw_fif')
        self.DATA_01_EPO = Path(self.PATH_PROJECT, 'Data', '01_epo')
    def getPath(self, key: str) -> Path:
        """Get path.

        Parameters
        ----------
        key : str
            Identifier for the path.

        Returns
        -------
        Path
            Path object.
        """
        return getattr(self, key)

