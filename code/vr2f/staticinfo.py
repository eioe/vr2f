"""
Store global variables.

Keep paths, colors, etc. to access them throughout the project.
"""

from pathlib import Path

MAIN_PATH = ""


class PATHS:
    """Store relevant paths for the project."""

    def __init__(self) -> None:
        """Initialize paths."""
        self.PATH_PROJECT = Path(__file__).parents[2]
        if self.PATH_PROJECT.parts[-1] != "vr2f":
            raise ValueError(
                "staticinfo seems to be in the wrong folder. \
                              Cannot point to project folder."
            )
        self.DATA_SUBJECTS = Path(self.PATH_PROJECT, "data", "Subjects")
        self.DATA_00_RAWFIF = Path(self.PATH_PROJECT, "data", "00_raw_fif")
        self.DATA_01_EPO = Path(self.PATH_PROJECT, "data", "01_epo")
        self.DATA_02_ICA_AR = Path(self.PATH_PROJECT, "data", "02_ica", "ar")
        self.DATA_02_ICA = Path(self.PATH_PROJECT, "data", "02_ica")
        self.DATA_02_EOGANNOT = Path(self.PATH_PROJECT, "data", "02_eogannot")
        self.DATA_02_POSTICA = Path(self.PATH_PROJECT, "data", "02_ica", "cleaneddata")
        self.DATA_03_AR = Path(self.PATH_PROJECT, "data", "03_ar")
        self.DATA_04_DECOD_SENSORSPACE = Path(self.PATH_PROJECT, "data", "04_decod", "sensorspace")
        self.DATA_ET_PREPROC = Path(self.PATH_PROJECT, "data", "eye_tracking", "01_preproc")
        self.FIGURES = Path(self.PATH_PROJECT, "results", "figures")

    def get_path(self, key: str) -> Path:
        """
        Get path.

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
    """Store timing parameters used in the experiment."""

    def __init__(self) -> None:
        """Initialize timing parameters."""
        self.DUR_BL = 0.2

    def get_timing(self, key: str) -> float:
        """
        Get time info.

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
    """Store configuration parameters."""

    def __init__(self) -> None:
        """Initialize configuration parameters."""
        self.N_JOBS = -2


class COLORS:
    """Store color values for plotting."""

    def __init__(self) -> None:
        """Initialize color values."""
        self.COLDICT = {
            "all": "#003049",
            "mono": "#9BC1BC",  # "#d62828",
            "stereo": "#4BA3C3",  # "#f77f00",
            "viewcond": "#588157",
            "happy": "#307351",
            "angry": "#840032",
            "neutral": "#002642",
            "surprised": "#E59500",
        }

class CONSTANTS:
    """Store constant values."""

    def __init__(self) -> None:
        """Initialize constant values."""
        self.SFREQ_ET = 120
        self.SFREQ_EEG = 500
        self.N_TRIALS_TRAINING = 24
