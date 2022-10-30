"""Check and clean raw epoch files

Open up each epoch file once and do some checks: 
- does the ERP (across all chans) look healthy
- skim through the data to find obv. flaws + make comments
- plot the PDS
- remove & interpolate bad channels
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mne

from vr2fem_analyses.staticinfo import PATHS, TIMINGS, CONFIG
from vr2fem_analyses.staticinfo import TIMINGS
from vr2fem_analyses import preprocess, helpers


def main(sub_nr: int = None):
    paths = PATHS()
    timings = TIMINGS()
    config = CONFIG()
    path_data = Path(paths.DATA_01_EPO, 'erp')
    # load data
    sub_list_str = [s.split('-epo')[0] for s in os.listdir(path_data)]

    if sub_nr is not None:
        sub_list_str = [sub_list_str[sub_nr]]  

    for subID in sub_list_str:

        fname = Path(paths.DATA_01_EPO, 'erp', f'{subID}-epo.fif')
        epochs = mne.read_epochs(fname)

        freq_res = 1
        sfreq = epochs.info['sfreq']
        nfft = (2 ** np.ceil(np.log2(sfreq / freq_res))).astype(int)
        picks = mne.pick_types(epochs.info,
                               meg=False,
                               eeg=True,
                               eog=True,
                               exclude='bads')
        epochs.compute_psd(method='welch',
                           fmin=0,
                           fmax=45,
                           n_fft=nfft,
                           picks=picks,
                           n_jobs=config.N_JOBS).plot()

        # you can select the bad channels here:
        epochs.plot(n_epochs=15,
                    n_channels=64,
                    picks=picks)

        # or give as an input here:
        epochs.info['bads'] += []

        # plot evoked
        (epochs.copy()
               .apply_baseline((-timings.DUR_BL, 0))
               .average()
               .plot(picks='all'))

        # check psd again after rejection
        epochs.compute_psd(method = 'welch',
                           fmin=0,
                           fmax=45,
                           n_fft = nfft,
                           picks=picks,
                           n_jobs=config.N_JOBS).plot()

        bads = epochs.info['bads']

        epochs = epochs.interpolate_bads(reset_bads=False)

        # save:
        fpath = Path(paths.DATA_01_EPO, 'erp', 'clean')
        fpath.mkdir(exist_ok=True)
        fname = Path(fpath, f'{subID}-epo.fif')
        epochs.save(fname, overwrite=True)

        # now also reject those chans from the copy for the ICA
        fname_ica = Path(paths.DATA_01_EPO, 'ica', f'{subID}-epo.fif')
        epochs_ica = mne.read_epochs(fname_ica)
        epochs_ica.info['bads'] = bads
        epochs_ica.interpolate_bads(reset_bads=True)
        fpath_ica = Path(paths.DATA_01_EPO, 'ica', 'clean')
        fpath_ica.mkdir(exist_ok=True)
        fname_ica = Path(fpath_ica, f'{subID}-epo.fif')
        epochs_ica.save(fname_ica, overwrite=True)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        helpers.print_msg("Running Job Nr. " + sys.argv[1])
        JOB_NR = int(sys.argv[1])
    else:
        JOB_NR = None
    main(JOB_NR)

