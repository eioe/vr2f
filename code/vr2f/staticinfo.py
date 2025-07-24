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
        self.DATA_ET = Path(self.PATH_PROJECT, "data", "eye_tracking")
        self.DATA_ET_PREPROC = Path(self.PATH_PROJECT, "data", "eye_tracking", "01_preproc")
        self.DATA_ET_DECOD = Path(self.DATA_ET, "03_decoding")
        self.RESULTS = Path(self.PATH_PROJECT, "results")
        self.FIGURES = Path(self.PATH_PROJECT, "results", "figures")
        self.STIMULIIMAGES = Path(self.PATH_PROJECT, "data", "images", "stimuli")

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
        self.ERP_WINDOWS = dict(P1 = (0.08, 0.12),
                                N170 = (0.13, 0.2),
                                EPN = (0.25, 0.3),
                                LPC = (0.4, 0.6))

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
        base_colors = {
            "all": "#003049",
            "mono": "#9BC1BC",  # "#d62828",
            "stereo": "#4BA3C3",  # "#f77f00",
            "viewcond": "#588157",
            "happy": "#307351",
            "angry": "#840032",
            "neutral": "#002642",
            "surprised": "#E59500",
            "angry_vs_happy": "#1A7AE6",
            "angry_vs_surprised": "#2C7B85",
            "angry_vs_neutral": "#840032",
            "happy_vs_surprised": "#F1A028",
            "happy_vs_neutral": "#8F26D9",
            "surprised_vs_neutral": "#92E61A",
            "mono_vs_stereo": "#FFD662",
            "id1_vs_id2_vs_id3": "#FFD662",
            "train_mono-test_mono": "#9BC1BC",
            "train_stereo-test_stereo": "#4BA3C3",
            "train_mono-test_stereo": "#d62828",
            "train_stereo-test_mono": "#f77f00",
            "P1": "#b3cde0",
            "N170": "#6497b1",
            "EPN": "#005b96",
            "LPC": "#03396c",
        }
        self.COLDICT = base_colors
        self.COLDICT.update({
            # "neutral_vs_happy_vs_angry_vs_surprised": self.COLDICT["all"],
            # "angry_vs_neutral": self.COLDICT["angry"],
            # "happy_vs_neutral": self.COLDICT["happy"],
            # "surprised_vs_neutral": self.COLDICT["surprised"]
        })

class CONSTANTS:
    """Store constant values."""

    def __init__(self) -> None:
        """Initialize constant values."""
        self.SFREQ_ET = 120
        self.SFREQ_EEG = 500
        self.N_TRIALS_TRAINING = 24
        self.CM = 1/2.54
        self.COND_DICT = {
            "viewcond": {1: "mono", 2: "stereo"},
            "emotion": {1: "neutral", 2: "happy", 3: "angry", 4: "surprised"},
            "avatar_id": {1: "Woman_01", 2: "Woman_04", 3: "Woman_08"},
            }
        self.ET_SACC_VFAC = 5
        self.ET_SACC_MINDUR = 3  # in samples
        self.ET_SACC_MINAMP = 2  # in dva
        self.ET_FIX_MINDUR = 0.05  # in s
        # participants with higher proportion of rejected epochs will be excluded:
        self.AR_PROP_EPOS_TO_REJECT_SUB = 0.2
        

