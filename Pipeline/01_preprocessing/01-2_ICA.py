import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import create_eog_epochs
import autoreject

from vr2fem_analyses.staticinfo import PATHS
from vr2fem_analyses.staticinfo import TIMINGS
from vr2fem_analyses import preprocess, helpers

def main(sub_nr: int):
    paths = PATHS()
    timings = TIMINGS()
    path_data = Path(paths.DATA_01_EPO, 'erp')
    # load data
    sub_list_str = [s.split('-epo')[0] for s in os.listdir(path_data)]

    if sub_nr is not None:
        sub_list_str = [sub_list_str[sub_nr]]  

    for subID in sub_list_str:
        
        fname = Path(paths.DATA_01_EPO, 'ica', f'{subID}-epo.fif')
        data_forICA = mne.read_epochs(fname)

        # AR needs a montage
        easycap_montage = mne.channels.make_standard_montage('easycap-M1')
        data_forICA.set_montage(easycap_montage)

        # clean it with autoreject local to remove bad epochs for better ICA fit:
        data_forAR = data_forICA.copy().apply_baseline((-timings.DUR_BL, 0))
        # AR does not perform well on non-baseline corrected data

        rn_ch_dict = {
            'AF7': 'IO1',
            'AF8': 'IO2',
            'FT9': 'LO1',
            'FT10': 'LO2'
            }
        data_forAR.rename_channels(rn_ch_dict) 
        print('renaming eog channels.')

        EOG_chans = {ch: 'eog' for ch in list(rn_ch_dict.values()) + ['Fp1', 'Fp2']}
        data_forAR.set_channel_types(EOG_chans)

        _, ar, reject_log = preprocess.clean_with_ar_local(subID,
                                                        data_forAR,
                                                        ar_from_disc=False,
                                                        save_to_disc=True,
                                                        ar_path=paths.DATA_02_ICA_AR)

        # Get ICA weights
        ica = preprocess.get_ica_weights(subID,
                                        data_forICA[~reject_log.bad_epochs],
                                        picks=None,
                                        reject=None,
                                        method='picard',
                                        fit_params=None,
                                        ica_from_disc=False,
                                        save_to_disc=True,
                                        ica_path=paths.DATA_02_ICA)

if __name__ == '__main__':
    if (len(sys.argv) > 1):
        helpers.print_msg('Running Job Nr. ' + sys.argv[1])
        JOB_NR = int(sys.argv[1])
    else:
        JOB_NR = None
    main(JOB_NR)