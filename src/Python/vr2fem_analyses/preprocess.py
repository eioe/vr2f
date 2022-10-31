from pathlib import Path
import pandas as pd
import numpy as np
import mne
from mne.preprocessing import create_eog_epochs
import autoreject
from vr2fem_analyses.staticinfo import PATHS, CONFIG


def clean_with_ar_local(
    subID,
    data_,
    n_jobs=None,
    ar_from_disc=False,
    save_to_disc=True,
    ar_path=None,
    rand_state=42,
):

    paths = PATHS()
    config = CONFIG()
    if n_jobs is None:
        n_jobs = config.N_JOBS

    if ar_path is None:
        req_path = [p for p in [ar_from_disc, save_to_disc] if p]
        if len(req_path) > 0:
            message = (
                "If you want to read from or write to disk, you need to provide the according path."
                + '(Argument "ar_path")'
            )
            raise ValueError(message)
    else:
        if paths.DATA_02_ICA_AR == ar_path:
            append = "-preICA-ar"
        elif paths.DATA_03_AR == ar_path:
            append = "-postICA-ar"
        else:
            message = (
                "I only can write or read AR files in these folders:\n"
                + paths.DATA_02_ICA_AR
                + "\n"
                + paths.DATA_03_AR
            )
            raise ValueError(message)

    if ar_from_disc:
        fname_ar = Path(ar_path, f"{subID}{append}.fif")
        ar = autoreject.read_auto_reject(fname_ar)
        epo_clean, reject_log = ar.transform(data_, return_log=True)

    else:
        picks = mne.pick_types(data_.info, meg=False, eeg=True, stim=False, eog=False)
        ar = autoreject.AutoReject(
            n_interpolate=np.array([2, 4, 8, 16]),
            consensus=np.linspace(0.1, 1.0, 11),
            picks=picks,
            n_jobs=n_jobs,
            random_state=rand_state,
            verbose="tqdm",
        )
        epo_clean, reject_log = ar.fit_transform(data_, return_log=True)

    if save_to_disc:
        # Save results of AR to disk:
        fpath_ar = Path(ar_path)
        fname_ar = Path(fpath_ar, f"{subID}{append}.fif")
        fpath_ar.mkdir(parents=True, exist_ok=True)
        ar.save(fname_ar, overwrite=True)
        # externally readable version of reject log
        rejlog_df = pd.DataFrame(
            reject_log.labels, columns=reject_log.ch_names, dtype=int
        )
        rejlog_df["badEpoch"] = reject_log.bad_epochs
        fpath_rejlog = Path(ar_path, "rejectlogs")
        fname_rejlog = Path(fpath_rejlog, f"{subID}{append}-rejectlog.csv")
        fpath_rejlog.mkdir(parents=True, exist_ok=True)
        rejlog_df.to_csv(fname_rejlog, float_format="%i")
        if "postICA" in append:
            fpath_cleaneddata = Path(ar_path, "cleaneddata")
            fname = Path(fpath_cleaneddata, f"{subID}-postAR-epo.fif")
            epo_clean.save(fname)
    return epo_clean, ar, reject_log


def get_ica_weights(
    subID,
    data_=None,
    picks=None,
    reject=None,
    method="picard",
    fit_params=None,
    ica_from_disc=False,
    save_to_disc=True,
    ica_path=None,
):
    # Load ICA data
    if ica_from_disc:
        ica = mne.preprocessing.read_ica(fname=Path(ica_path, subID + "-ica.fif"))
    else:
        data_.drop_bad(reject=reject)
        ica = mne.preprocessing.ICA(method=method, fit_params=fit_params)
        ica.fit(data_, picks=picks)

        if save_to_disc:
            ica.save(fname=Path(ica_path, subID + "-ica.fif"), overwrite=True)
    return ica


def get_eog_epochs_clean(
    data_raw,
    subID=None,
    t0=None,
    t1=None,
    manual_mode=False,
    eogannot_from_disc=False,
    save_eogannot_to_disc=True,
    eogannot_path=None,
):
    """Robust way to get epochs centered around EOG events.
    Produce EOG epochs from raw data by epoching to "EOG events" (peaks in the EOG). Based on
    `mne.preprocessing.create_eog_epochs`, but more robust.
    Specify either a (list of) start time(s) t0 and a (list of) end time(s) t1 of stretches in the data with somewhat
    clean EOG signal. Alternatively, open a plot of the data and mark it manually by annotating it as "clean_eog"
    (`manual_mode`). You can also save these manual annotations to disc and load them in again later.
    Parameters
    ----------
    data_raw : MNE raw object.
        Must contain `VEOG`and `HEOG` channel of type `eog`.
    t0 : int | list | array
        Start time(s) of clean EOG intervall(s).
        Ignored if `manual_mode` is `True`.
    t1 : int | list | array
        End time(s) of clean EOG intervall(s).
        Ignored if `manual_mode` is `True`.
    manual_mode : bool
        Set to `True` if you want to manually mark stretches of clean EOG data or load according annotations from disc.
    eogannot_from_disc : bool
        Load annotations from disc.
        Ignored if `manual_mode` is `False`.
    save_eogannot_to_disc : bool
        Save annotations for later reuse.
        Ignored if `manual_mode` is `False`.
    eogannot_path : str
        Filename to/from which the annotations shall be saved/loaded.
        Ignored if `manual_mode` is `False`.
    Returns
    -------
    epochs_veog : mne Epochs
        Epochs of length 1s, centered around VEOG events.
    epochs_heog : mne Epochs
        Epochs of length 1s, centered around HEOG events.
    """

    if not isinstance(t0, (list, np.ndarray)):
        t0 = [t0]
    if not isinstance(t1, (list, np.ndarray)):
        t1 = [t1]

    if manual_mode:
        mark_data = data_raw.copy()

        if eogannot_from_disc:
            fname = Path(eogannot_path, subID + "-eog-annot.fif")
            annot = mne.read_annotations(fname)
            mark_data.set_annotations(annot)
        else:
            # Mark stretches (at least 1) of data with clean EOG
            mark_data.pick_types(eog=True, eeg=False).filter(
                l_freq=1, h_freq=None, picks=["eog"], verbose=False
            ).plot(
                duration=180,
                scalings={"eog": 850e-6},
                # remove_dc=True,
                block=True,
            )
            if save_eogannot_to_disc:
                fname = Path.join(eogannot_path, subID + "-eog-annot.fif")
                mark_data.annotations.save(fname, overwrite=True)

        # Calculate clean epochs locked to EOG-peaks (!name the annotation `clean_eeg`!):
        raws = [
            idx for idx in mark_data.annotations if idx["description"] == "clean_eog"
        ]
        t0 = [raw["onset"] for raw in raws]
        t1 = [raw["onset"] + raw["duration"] for raw in raws]

    raw_eog = data_raw.copy().crop(t0[0], t1[0])
    if len(t0) > 1:
        for t0_, t1_ in zip(t0[1:], t1[1:]):
            raw_eog.append(data_raw.copy().crop(t0_, t1_))

    # Heuristically determine a robust threshold for EOG peak detection
    tmp = (
        raw_eog.copy()
        .load_data()
        .pick_channels(["VEOG", "HEOG"])
        .filter(l_freq=1, h_freq=10, picks=["eog"], verbose=False)
    )

    tmp_epo = mne.make_fixed_length_epochs(tmp, preload=True)

    mean_threshs = np.mean(tmp_epo.get_data().squeeze().ptp(axis=0) / 4, axis=1)
    thresh_dict = {ch: thresh for ch, thresh in zip(tmp_epo.ch_names, mean_threshs)}

    epochs_veog = create_eog_epochs(
        raw_eog,
        ch_name="VEOG",
        baseline=(None, None),
        thresh=thresh_dict["VEOG"],
        verbose=False,
    )

    print(f"Created {len(epochs_veog)} VEOG epochs.")
    if len(epochs_veog) < 50:
        print(
            "Not really a lot. This might be a problem! Consider marking longer stretches of clean EOG data."
        )
        print("########################")

    epochs_heog = create_eog_epochs(
        raw_eog,
        ch_name="HEOG",
        baseline=(None, None),
        thresh=thresh_dict["HEOG"],
        verbose=False,
    )

    print(f"Created {len(epochs_heog)} HEOG epochs.")
    if len(epochs_heog) < 50:
        print(
            "Not really a lot. This might be a problem! Consider marking longer stretches of clean EOG data."
        )
        print("########################")
    return epochs_veog, epochs_heog


def calc_bipolar_eog(data):
    """calculate and add bipolar EOG chans.

    Calculates chans VEOG and HEOG from Fp1/2-IO1/2 and LO1-LO2. Adds the two
    bipolar EOG chans to the data set (in place).

    Parameters
    ----------
    data : Raw or Epochs
        Data; needs to have chans Fp1, Fp2, LO1, LO2, IO1, IO2.

    Returns
    -------
    Raw or Epochs
        Modified data (now with two additional EOG chans.)
    """
    # ; works in place:
    data.load_data()
    dataL = data.get_data(['Fp1']) - data.get_data(['IO1'])
    dataR = data.get_data(['Fp2']) - data.get_data(['IO2'])
    if isinstance(data, mne.io.BaseRaw):
        axis = 0
    elif isinstance(data, mne.BaseEpochs):
        axis = 1
    else:
        raise ValueError('Invalid type. Only support raw or epochs.')
    dataVEOG = np.stack((dataL, dataR), axis=axis).mean(axis=axis)

    dataHEOG = data.get_data(['LO1']) - data.get_data(['LO2'])
    dataEOG = np.concatenate((dataVEOG, dataHEOG), axis=axis)
    info = mne.create_info(ch_names=['VEOG', 'HEOG'], sfreq=data.info['sfreq'], ch_types=['eog', 'eog'])
    if isinstance(data, mne.io.BaseRaw):
        newdata = mne.io.RawArray(dataEOG, info=info)
    elif isinstance(data, mne.BaseEpochs):
        newdata = mne.EpochsArray(dataEOG, info=info)
    else:
        raise ValueError('Invalid type. Only support raw or epochs.')

    data.add_channels([newdata], force_update_info=True)
    return data
