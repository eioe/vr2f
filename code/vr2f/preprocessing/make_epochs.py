"""
Epoch the data.

Crop the raw data into epochs (time-locked to the onset of the face stimuli).
We label the epochs with useful names, according to our marker logic
(https://www.notion.so/mind-body-emotion/LSL-Marker-coding-in-file-067912a0220d4c4b8fb599d0e4b68fc5). We extract epochs
for the main analyses ('erp') and another copy ('ica') which we will use to
determine the ICA decomposition. Raw data for ERP epochs is filtered [0.1, 40]
Hz, for ICA epochs [1.0, 40]Hz.


2022/10 -- Felix Klotzsche
"""

import os
import re
from pathlib import Path

import mne

from vr2f.staticinfo import PATHS


def get_data_and_events(sub_id: str, path_data: Path):
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


def extract_epochs(  # noqa: PLR0913
    raw_data,
    events,
    event_id_,
    tmin,
    tmax,
    l_freq,
    h_freq,
    baseline=None,
    bad_epos=None,
    n_jobs=1,
):
    """
    Cut out epochs around markers.

    Parameters
    ----------
    raw_data : _type_
        _description_
    events : _type_
        _description_
    event_id_ : _type_
        _description_
    tmin : _type_
        _description_
    tmax : _type_
        _description_
    l_freq : _type_
        _description_
    h_freq : _type_
        _description_
    baseline : _type_, optional
        _description_, by default None
    bad_epos : _type_, optional
        _description_, by default None
    n_jobs : int, optional
        _description_, by default 1

    Returns
    -------
    _type_
        _description_

    """
    # filter the data:
    if (l_freq is not None) or (h_freq is not None):
        filtered = raw_data.load_data().filter(l_freq=l_freq, h_freq=h_freq, n_jobs=n_jobs, verbose=False)
    else:
        filtered = raw_data.load_data()

    epos_ = mne.Epochs(
        filtered,
        events,
        event_id=event_id_,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        preload=False,
    )
    if bad_epos is not None:
        epos_.drop(bad_epos, "BADRECORDING")
    return epos_


def check_epo_numbers(sub_id: str, n_epos: int):
    """
    Check whether the subject has the expected number of epochs.

    Parameters
    ----------
    sub_id : str
        Subject ID (e.g., 'VR2FEM_S03')
    n_epos : int
        Number of epochs/events in data.

    """
    subs_with_717_exception = ("VR2FEM_S03", "VR2FEM_S06", "VR2FEM_S19", "VR2FEM_S23", "VR2FEM_S32")
    expected = 717 if sub_id in subs_with_717_exception else 720
    if n_epos != expected:
        raise ValueError(f"Subject {sub_id} has {n_epos} epochs, but should have {expected}.")


def main():
    """Run main."""
    paths = PATHS()
    path_data = paths.DATA_00_RAWFIF
    # load data
    sub_list_str = [s.split("-raw")[0] for s in os.listdir(path_data)]

    event_info = {
        "mono/id1/neutral": 111,
        "mono/id1/happy": 112,
        "mono/id1/angry": 113,
        "mono/id1/surprised": 114,
        "mono/id2/neutral": 121,
        "mono/id2/happy": 122,
        "mono/id2/angry": 123,
        "mono/id2/surprised": 124,
        "mono/id3/neutral": 131,
        "mono/id3/happy": 132,
        "mono/id3/angry": 133,
        "mono/id3/surprised": 134,
        "stereo/id1/neutral": 211,
        "stereo/id1/happy": 212,
        "stereo/id1/angry": 213,
        "stereo/id1/surprised": 214,
        "stereo/id2/neutral": 221,
        "stereo/id2/happy": 222,
        "stereo/id2/angry": 223,
        "stereo/id2/surprised": 224,
        "stereo/id3/neutral": 231,
        "stereo/id3/happy": 232,
        "stereo/id3/angry": 233,
        "stereo/id3/surprised": 234,
    }

    for sub_id in sub_list_str:
        raw, events, ev_dict_full = get_data_and_events(sub_id, path_data)

        # trim event_dict to relevant events
        ev_dict_stripped = {
            int(re.findall(r"\d+", ev)[0]): ev_dict_full[ev] for ev in ev_dict_full if len(re.findall(r"\d+", ev)) > 0
        }
        event_dict = {k: ev_dict_stripped[event_info[k]] for k in event_info}
        events_faceonset = [ev for ev in events if ev[2] in event_dict.values()]
        # drop training trials
        events_faceonset = events_faceonset[24:]

        # make epochs for ERP analysis: [0.1 40]
        epos_erp = extract_epochs(
            raw,
            events_faceonset,
            event_dict,
            tmin=-0.3,
            tmax=1.0,
            l_freq=0.1,
            h_freq=40,
        )

        check_epo_numbers(sub_id, len(epos_erp.events))

        fpath = Path(paths.DATA_01_EPO, "erp")
        fpath.mkdir(parents=True, exist_ok=True)

        fname = Path(fpath, f"{sub_id}-epo.fif")
        epos_erp.save(fname, overwrite=True)

        # make epochs for ICA [1.0 40]
        epos_ica = extract_epochs(raw, events_faceonset, event_dict, tmin=-0.3, tmax=1.0, l_freq=1, h_freq=40)

        fpath = Path(paths.DATA_01_EPO, "ica")
        fpath.mkdir(parents=True, exist_ok=True)
        fname = Path(fpath, f"{sub_id}-epo.fif")
        epos_ica.save(fname, overwrite=True)

        # make unfiltered epochs:
        epos_unfiltered = extract_epochs(
            raw, events_faceonset, event_dict, tmin=-0.3, tmax=1.0, l_freq=None, h_freq=None
        )

        fpath = Path(paths.DATA_01_EPO, "unfiltered")
        fpath.mkdir(parents=True, exist_ok=True)
        fname = Path(fpath, f"{sub_id}-epo.fif")
        epos_unfiltered.save(fname, overwrite=True)


if __name__ == "__main__":
    main()
