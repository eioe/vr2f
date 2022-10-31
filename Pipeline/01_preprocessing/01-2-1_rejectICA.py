import os
from pathlib import Path
import numpy as np
import mne

from vr2fem_analyses.staticinfo import PATHS, CONFIG
from vr2fem_analyses.preprocess import (
    clean_with_ar_local,
    get_eog_epochs_clean,
    get_ica_weights,
    calc_bipolar_eog,
)


def get_raw_and_events(subID: str, path_data: Path):
    """Load data, events, and event dict.

    Parameters
    ----------
    subID : str
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
    fname_inp = Path(path_data, subID + "-raw.fif")
    raw = mne.io.read_raw_fif(fname_inp)
    events, event_id = mne.events_from_annotations(raw)
    return raw, events, event_id


paths = PATHS()
config = CONFIG()

# load data
sub_list_str = [s.split("-ica")[0] for s in os.listdir(paths.DATA_02_ICA)]


subID = sub_list_str[9]

# read raw
fpath_raw = paths.DATA_00_RAWFIF
path_data = paths.DATA_00_RAWFIF
raw, events, ev_dict_full = get_raw_and_events(subID, path_data)

# read epochs
fname_ica = Path(paths.DATA_01_EPO, "ica", f"{subID}-epo.fif")
data_ica = mne.read_epochs(fname_ica)
fname_erp = Path(paths.DATA_01_EPO, "erp", f"{subID}-epo.fif")
data_erp = mne.read_epochs(fname_erp)

# read bad epochs from pre-ica AR object:
_, ar, reject_log = clean_with_ar_local(
    subID,
    data_ica,  # doesn't matter as we load from disk
    ar_from_disc=True,
    save_to_disc=False,
    ar_path=paths.DATA_02_ICA_AR,
)

bad_epos_idx_preica = reject_log.bad_epochs

# get ICA
ica = get_ica_weights(
    subID,
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
    subID=subID,
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

ica.plot_sources(inst=data_ica)  # [~bad_epos_idx_preica])

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

# after making sure that all is fine: -----------------
# Now we apply these ICA weights to the original epochs:
data_postica = ica.apply(data_erp.copy().load_data())

ha = calc_bipolar_eog(data_postica.copy())
ha.plot(picks="all")

fpath_postica = Path(paths.DATA_02_POSTICA)
fpath_postica.mkdir(exist_ok=True)
fname_postica = Path(fpath_postica, f'{subID}-postICA-epo.fif')
data_postica.save(fname_postica, overwrite=True)
