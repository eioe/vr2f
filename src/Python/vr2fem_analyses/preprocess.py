from pathlib import Path
import sys
import pandas as pd
import numpy as np
import mne
from mne.preprocessing import create_eog_epochs
import autoreject
from vr2fem_analyses.staticinfo import PATHS, CONFIG


def clean_with_ar_local(subID,
                        data_,
                        n_jobs=None,
                        ar_from_disc=False,
                        save_to_disc=True,
                        ar_path=None,
                        rand_state=42):

    paths = PATHS()
    config = CONFIG()
    if n_jobs is None:
        n_jobs = config.N_JOBS

    if ar_path is None:
        req_path = [p for p in [ar_from_disc, save_to_disc] if p]
        if len(req_path) > 0:
            message = 'If you want to read from or write to disk, you need to provide the according path.' + \
                      '(Argument "ar_path")'
            raise ValueError(message)
    else:
        if paths.DATA_02_ICA_AR == ar_path:
            append = '-preICA-ar'
        elif paths.DATA_02_AR == ar_path:
            append = f'-postICA-ar'
        else:
            message = f'I only can write or read AR files in these folders:\n' + \
                      paths.DATA_02_ICA_AR + '\n' + \
                      paths.DATA_02_AR
            raise ValueError(message)

    if ar_from_disc:
        fname_ar = Path(ar_path, f'{subID}{append}.fif')
        ar = autoreject.read_auto_reject(fname_ar)
        epo_clean, reject_log = ar.transform(data_, return_log=True)

    else:
        picks = mne.pick_types(data_.info, meg=False, eeg=True, stim=False,
                               eog=False)
        ar = autoreject.AutoReject(n_interpolate=np.array([2,4,8,16]),
                                   consensus= np.linspace(0.1, 1.0, 11),
                                   picks=picks, 
                                   n_jobs=n_jobs,
                                   random_state = rand_state,
                                   verbose='tqdm')
        epo_clean, reject_log = ar.fit_transform(data_, return_log=True)

    if save_to_disc:
        # Save results of AR to disk:
        fpath_ar = Path(ar_path)
        fname_ar = Path(fpath_ar, f'{subID}{append}.fif')
        fpath_ar.mkdir(parents=True, exist_ok=True)
        ar.save(fname_ar, overwrite=True)
        # externally readable version of reject log
        rejlog_df = pd.DataFrame(reject_log.labels,
                                 columns=reject_log.ch_names,
                                 dtype=int)
        rejlog_df['badEpoch'] = reject_log.bad_epochs
        fpath_rejlog = Path(ar_path, 'rejectlogs')
        fname_rejlog = Path(fpath_rejlog, f'{subID}{append}-rejectlog.csv')
        fpath_rejlog.mkdir(parents=True, exist_ok=True)
        rejlog_df.to_csv(fname_rejlog, float_format="%i")
        if 'postICA' in append:
            fpath_cleaneddata = Path(ar_path, 'cleaneddata')
            fname = Path(fpath_cleaneddata, f'{subID}-postAR-epo.fif')
            epo_clean.save(fname)


    return epo_clean, ar, reject_log



def get_ica_weights(subID,
                    data_=None,
                    picks=None,
                    reject=None,
                    method='picard',
                    fit_params=None,
                    ica_from_disc=False,
                    save_to_disc=True,
                    ica_path=None):
    ### Load ICA data
    if ica_from_disc:
        ica = mne.preprocessing.read_ica(fname=Path(ica_path, subID + '-ica.fif'))
    else:
        data_.drop_bad(reject=reject)
        ica = mne.preprocessing.ICA(method=method,
                                    fit_params=fit_params)
        ica.fit(data_,
                picks=picks)

        if save_to_disc:
            ica.save(fname=Path(ica_path, subID + '-ica.fif'),
                     overwrite=True)
    return ica