import os
from pathlib import Path
import numpy as np
import mne

from vr2fem_analyses.staticinfo import PATHS, CONFIG

paths = PATHS()
fpath = Path(paths.DATA_03_AR, 'cleaneddata')
os.listdir(fpath)