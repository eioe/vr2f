"""Store global variables

Keep paths, colors, etc. to access them throughout the project.
"""
from pathlib import Path

MAIN_PATH = ''


class PATHS:
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
        self.DATA_02_ICA_AR = Path(self.PATH_PROJECT, 'Data', '02_ica', 'ar')
        self.DATA_02_ICA = Path(self.PATH_PROJECT, 'Data', '02_ica')
        self.DATA_02_EOGANNOT = Path(self.PATH_PROJECT, 'Data', '02_eogannot')
        self.DATA_02_POSTICA = Path(self.PATH_PROJECT, 'Data', '02_ica', 'cleaneddata'),
        self.DATA_03_AR = Path(self.PATH_PROJECT, 'Data', '03_ar')

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


class TIMINGS:
    def __init__(self) -> None:
        self.DUR_BL = 0.2

    def getTiming(self, key: str) -> float:
        """Get time info.

        Parameters
        ----------
        key : str
            Identifier for the timing.

        Returns
        -------
        float
            Time info.
        """
        return getattr(self, key)


class CONFIG:
    def __init__(self) -> None:
        self.N_JOBS = -2
