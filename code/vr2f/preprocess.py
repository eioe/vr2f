from pathlib import Path

import autoreject
import mne
import numpy as np
import pandas as pd
from mne.preprocessing import create_eog_epochs

from vr2f.staticinfo import CONFIG, PATHS


def clean_with_ar_local(
    subID,
    data_,
    n_jobs=None,
    ar_from_disc=False,
    save_to_disc=True,
    ar_path=None,
    rand_state=42,
):
    """
    Clean MNE epoch data using local AutoReject (AR) parameters.

    This function applies the AutoReject algorithm (Jas et al., 2017) to 
    clean EEG epochs. 
    It can either read a precomputed AutoReject model from disk or fit 
    a new one on the provided data. Optionally, the results can be saved 
    to disk, including the fitted AutoReject object, rejection logs, and 
    cleaned data (for post-ICA cleaning).

    Parameters
    ----------
    subID : str
        Subject identifier used for file naming when saving/loading AR models.
    data_ : mne.Epochs
        MNE Epochs object containing EEG data to be cleaned.
    n_jobs : int or None, optional
        Number of parallel jobs to run. If None, defaults to value in `CONFIG().N_JOBS`.
    ar_from_disc : bool, default=False
        If True, load an existing AutoReject model from disk instead of fitting a new one.
    save_to_disc : bool, default=True
        If True, save the AutoReject model, rejection logs, and optionally cleaned data to disk.
    ar_path : str or Path or None, optional
        Path to the directory where AR files should be read/written. 
        Must be provided if `ar_from_disc` or `save_to_disc` is True.
        Valid options are:
        - `PATHS().DATA_02_ICA_AR` (pre-ICA AutoReject)
        - `PATHS().DATA_03_AR` (post-ICA AutoReject)
    rand_state : int, default=42
        Random state used for reproducibility in AutoReject fitting.

    Returns
    -------
    epo_clean : mne.Epochs
        Cleaned MNE Epochs object after AutoReject processing.
    ar : autoreject.AutoReject
        The fitted AutoReject object (either loaded from disk or newly computed).
    reject_log : autoreject.RejectLog
        Reject log containing details of rejected epochs and channels.
    
    Notes
    -----
    - When `save_to_disc=True`:
        - The AutoReject object is saved as a `.fif` file.
        - Rejection logs are stored as `.csv` in a `rejectlogs/` subdirectory.
        - Cleaned data (only if post-ICA) is stored in `cleaneddata/` subdirectory.

    References
    ----------
    Mainak Jas, Denis Engemann, Federico Raimondo, Yousra Bekhti, and Alexandre Gramfort. 
    (2017). "Autoreject: Automated artifact rejection for MEG and EEG data." 
    NeuroImage, 159, 417–429. https://doi.org/10.1016/j.neuroimage.2017.06.034

    See Also
    --------
    AutoReject documentation: https://autoreject.github.io
    """
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
    elif ar_path == paths.DATA_02_ICA_AR:
        append = "-preICA-ar"
    elif ar_path == paths.DATA_03_AR:
        append = "-postICA-ar"
    else:
        message = (
            "I only can write or read AR files in these folders:\n" + paths.DATA_02_ICA_AR + "\n" + paths.DATA_03_AR
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
        rejlog_df = pd.DataFrame(reject_log.labels, columns=reject_log.ch_names, dtype=int)
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
    """
    Compute or load Independent Component Analysis (ICA) weights for EEG/MEG data.

    This function either loads an existing ICA decomposition from disk or 
    computes a new ICA on the provided MNE Epochs or Raw object. Optionally, 
    the fitted ICA can be saved to disk for later reuse.

    Parameters
    ----------
    subID : str
        Subject identifier used for file naming when saving/loading ICA solutions.
    data_ : mne.Epochs or mne.io.Raw | None
        Data to fit ICA on. Required if `ica_from_disc=False`.
    picks : str, list, slice, or None, optional
        Channels to include in ICA fitting. If None, defaults to EEG/MEG channels.
        See `mne.pick_types` for details.
    reject : dict or None, optional
        Rejection parameters to drop bad segments before ICA fitting.
        Follows the format used in MNE (e.g., `{'eeg': 150e-6}`).
    method : str, default='picard'
        ICA fitting method. Options include `'fastica'`, `'picard'`, `'infomax'`, etc.
        See `mne.preprocessing.ICA` documentation for available methods.
    fit_params : dict or None, optional
        Additional parameters passed to the ICA fitting algorithm.
    ica_from_disc : bool, default=False
        If True, load an existing ICA object from disk instead of computing a new one.
    save_to_disc : bool, default=True
        If True, save the computed ICA object to disk at `ica_path`.
    ica_path : str or Path or None, optional
        Path to directory for reading/writing ICA files. Must be provided if 
        `ica_from_disc=True` or `save_to_disc=True`.

    Returns
    -------
    ica : mne.preprocessing.ICA
        The fitted or loaded ICA object containing spatial filters and unmixing matrix.
    
    References
    ----------
    Hyvärinen, A., & Oja, E. (2000). Independent component analysis: 
    algorithms and applications. Neural Networks, 13(4–5), 411–430.
    https://doi.org/10.1016/S0893-6080(00)00026-5

    MNE-Python documentation: https://mne.tools/stable/generated/mne.preprocessing.ICA.html
    """

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
    """
    Robust way to get epochs centered around EOG events.
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
            mark_data.pick_types(eog=True, eeg=False).filter(l_freq=1, h_freq=None, picks=["eog"], verbose=False).plot(
                duration=180,
                scalings={"eog": 850e-6},
                # remove_dc=True,
                block=True,
            )
            if save_eogannot_to_disc:
                fname = Path.join(eogannot_path, subID + "-eog-annot.fif")
                mark_data.annotations.save(fname, overwrite=True)

        # Calculate clean epochs locked to EOG-peaks (!name the annotation `clean_eeg`!):
        raws = [idx for idx in mark_data.annotations if idx["description"] == "clean_eog"]
        t0 = [raw["onset"] for raw in raws]
        t1 = [raw["onset"] + raw["duration"] for raw in raws]

    raw_eog = data_raw.copy().crop(t0[0], t1[0])
    if len(t0) > 1:
        for t0_, t1_ in zip(t0[1:], t1[1:], strict=False):
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
    thresh_dict = {ch: thresh for ch, thresh in zip(tmp_epo.ch_names, mean_threshs, strict=False)}

    epochs_veog = create_eog_epochs(
        raw_eog,
        ch_name="VEOG",
        baseline=(None, None),
        thresh=thresh_dict["VEOG"],
        verbose=False,
    )

    print(f"Created {len(epochs_veog)} VEOG epochs.")
    if len(epochs_veog) < 50:
        print("Not really a lot. This might be a problem! Consider marking longer stretches of clean EOG data.")
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
        print("Not really a lot. This might be a problem! Consider marking longer stretches of clean EOG data.")
        print("########################")
    return epochs_veog, epochs_heog


def calc_bipolar_eog(data):
    """
    calculate and add bipolar EOG chans.

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
    dataL = data.get_data(["Fp1"]) - data.get_data(["IO1"])
    dataR = data.get_data(["Fp2"]) - data.get_data(["IO2"])
    if isinstance(data, mne.io.BaseRaw):
        axis = 0
    elif isinstance(data, mne.BaseEpochs):
        axis = 1
    else:
        raise ValueError("Invalid type. Only support raw or epochs.")
    dataVEOG = np.stack((dataL, dataR), axis=axis).mean(axis=axis)

    dataHEOG = data.get_data(["LO1"]) - data.get_data(["LO2"])
    dataEOG = np.concatenate((dataVEOG, dataHEOG), axis=axis)
    info = mne.create_info(ch_names=["VEOG", "HEOG"], sfreq=data.info["sfreq"], ch_types=["eog", "eog"])
    if isinstance(data, mne.io.BaseRaw):
        newdata = mne.io.RawArray(dataEOG, info=info)
    elif isinstance(data, mne.BaseEpochs):
        newdata = mne.EpochsArray(dataEOG, info=info)
    else:
        raise ValueError("Invalid type. Only support raw or epochs.")

    data.add_channels([newdata], force_update_info=True)
    return data
