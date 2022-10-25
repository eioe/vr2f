import os
from os import path as op
from pathlib import Path

import mne
from vr2fem_analyses.staticinfo import VR2FEMPATHS as PATHS


paths = PATHS()

subID = "VR2FEM_S34"
eeg_data = Path(paths.DATA_SUBJECTS, subID, 'MainTask', 'EEG')
fname1 = Path(eeg_data, subID + '.vhdr')
raw1 = mne.io.read_raw_brainvision(fname1, preload=True,       
                                verbose=False)

fname2 = Path(eeg_data, subID + '_2.vhdr')
raw2 = mne.io.read_raw_brainvision(fname2, preload=True, 
                                verbose=False)

mne.concatenate_raws([raw1, raw2])

mne.export.export_raw(Path(eeg_data, 'test.vhdr'), raw1)