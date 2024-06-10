"""Reject ICA components."""
import os
import sys
from pathlib import Path

import mne
import numpy as np

from vr2f import helpers
from vr2f.preprocess import (
    calc_bipolar_eog,
    clean_with_ar_local,
    get_eog_epochs_clean,
    get_ica_weights,
)
from vr2f.staticinfo import CONFIG, PATHS


def get_raw_and_events(sub_id: str, path_data: Path):
    """
    Load data, events, and event dict.

    Parameters
    ----------
    sub_id : str
        Subject ID (e.g., 'VR2FEM_S03')
    path_data : Path
        _description_

    Returns
    -------
    mne.io.Raw
        Raw object
    np.ndarray
        Event array; shape: (n_events, 3)
    dict
        Event dict

    """
    fname_inp = Path(path_data, sub_id + "-raw.fif")
    raw = mne.io.read_raw_fif(fname_inp)
    events, event_id = mne.events_from_annotations(raw)
    return raw, events, event_id

def main(sub_nr: int | None = None):
    """Run main."""
    paths = PATHS()
    config = CONFIG()

    # load data
    sub_list_str = [s.split("-ica")[0] for s in os.listdir(paths.DATA_02_ICA) if s.startswith("VR2FEM_")]
    sub_list_str.sort()

    if sub_nr is not None:
        sub_list_str = [sub_list_str[sub_nr]]

    for sub_id in sub_list_str:
        # read raw
        path_data = paths.DATA_00_RAWFIF
        raw, events, ev_dict_full = get_raw_and_events(sub_id, path_data)

        # read epochs
        fname_ica = Path(paths.DATA_01_EPO, "ica", f"{sub_id}-epo.fif")
        data_ica = mne.read_epochs(fname_ica)
        fname_erp = Path(paths.DATA_01_EPO, "erp", f"{sub_id}-epo.fif")
        data_erp = mne.read_epochs(fname_erp)

        # read bad epochs from pre-ica AR object:
        _, ar, reject_log = clean_with_ar_local(
            sub_id,
            data_ica,  # doesn't matter as we load from disk
            ar_from_disc=True,
            save_to_disc=False,
            ar_path=paths.DATA_02_ICA_AR,
        )

        bad_epos_idx_preica = reject_log.bad_epochs

        # get ICA
        ica = get_ica_weights(
            sub_id,
            data_ica[~bad_epos_idx_preica],  # doesn't matter as we load from disk
            reject=None,
            method="picard",
            fit_params=None,
            ica_from_disc=True,
            save_to_disc=False,
            ica_path=paths.DATA_02_ICA,
        )
        # detect EOG indices usiing ICA and raw


        print("***************************************")
        print("computing EOG Annot")
        print("***************************************")
        # Define an intervall of more or less clean EOG data (t0: start, t1:end)
        # Normally, using the first and last event works quite well.

        t0 = events[0][0] / raw.info["sfreq"]
        t1 = events[-1][0] / raw.info["sfreq"]

        raw = calc_bipolar_eog(raw)

        epochs_veog, epochs_heog = get_eog_epochs_clean(
            raw,
            subID=sub_id,
            t0=t0,
            t1=t1,
            manual_mode=False,
            eogannot_from_disc=False,
            save_eogannot_to_disc=False,
            eogannot_path=paths.DATA_02_EOGANNOT,
        )

        threshold = 0.75
        indices_veog, scores_veog = ica.find_bads_eog(
            epochs_veog, ch_name="VEOG", measure="correlation", threshold=threshold
        )

        indices_heog, scores_heog = ica.find_bads_eog(
            epochs_heog, ch_name="HEOG", measure="correlation", threshold=threshold
        )

        ica.plot_scores(
            [scores_veog, scores_heog],
            labels=["VEOG", "HEOG"],
            exclude=indices_veog + indices_heog,
        )
        exclude = list(np.unique(indices_veog + indices_heog))

        ica.exclude = exclude

        ica.plot_components(range(0, 32), inst=data_ica)
        ica.plot_components(range(32, 64), inst=data_ica)

        ica.plot_sources(inst=data_ica)

        # check the psd how the rejection worked----------------------------
        data_ica_2 = data_ica.copy()
        data_ica_2.load_data()
        ica.apply(data_ica_2)

        freq_res = 1
        nfft = (2 ** np.ceil(np.log2(data_erp.info["sfreq"] / freq_res))).astype(int)
        data_ica_2.compute_psd(
            method="welch", fmin=0, fmax=45, picks="all", n_fft=nfft, n_jobs=config.N_JOBS
        ).plot()
        mne.viz.plot_sensors(data_ica_2.info, show_names=True)
        data_ica_2.plot(n_epochs=15, n_channels=64, picks="all", block=True)

        # after making sure that all is fine: -----------------
        # Now we apply these ICA weights to the original epochs:
        data_postica = ica.apply(data_erp.copy().load_data())

        fpath_postica = Path(paths.DATA_02_POSTICA)
        fpath_postica.mkdir(exist_ok=True)
        fname_postica = Path(fpath_postica, f"{sub_id}-postICA-epo.fif")
        data_postica.save(fname_postica, overwrite=True)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        helpers.print_msg("Running Job Nr. " + sys.argv[1])
        JOB_NR = int(sys.argv[1])
    else:
        JOB_NR = None
    main(JOB_NR)
