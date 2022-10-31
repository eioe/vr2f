"""Clean raw EEG data.

EEG raw data is often a bit of a mess due to things that happen during data
acquisition. We use this script to clean the raw data files in a very subject
specific manner. We inspected all files for obvious flaws, noted them down,
(https://www.notion.so/mind-body-emotion/EEG-9aeb50340570466e9c77fbc2f07b7a0f)
and fix them here--where possible. You're welcome. ;)


2022/10 -- Felix Klotzsche
"""

import os
from pathlib import Path

import mne

from vr2fem_analyses.staticinfo import PATHS
from vr2fem_analyses.preprocess import calc_bipolar_eog


def setMontageChanTypes(raw: mne.io.Raw) -> mne.io.Raw:
    """Set a monatage and change chan type of EOG chans.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw data, is probably changed in place

    Returns
    -------
    mne.io.Raw
        Raw data with montage and EOG chan types set
    """
    rn_ch_dict = {"AF7": "IO1",
                  "AF8": "IO2",
                  "FT9": "LO1",
                  "FT10": "LO2"}
    raw.rename_channels(rn_ch_dict)
    print("renaming facial eog channels.")

    EOG_chans = {ch: "eog" for ch in list(rn_ch_dict.values())}
    raw.set_channel_types(EOG_chans)

    easycap_montage = mne.channels.make_standard_montage("easycap-M1")
    raw.set_montage(easycap_montage, on_missing='warn')

    return raw


def removeAnnotationsOfFirstNTrials(raw: mne.io.Raw, N: int) -> mne.io.Raw:
    """Remove the events for the first N trials.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw instance to be cleaned. Will be changed in-place.
    N : int
        Number of trials to be removed (from the beginning).

    Returns
    -------
    mne.io.Raw
        Cleaned raw.
    """
    n_ev2skip = N
    counter = 0
    idx_del = []
    for idx, a in enumerate(raw.annotations):
        sdesc = a["description"]
        if "Stimulus" in sdesc and counter < n_ev2skip:
            idx_del.append(idx)
            print(f"Removing event: {sdesc}")
            if sdesc == "Stimulus/S 16":  # trial end
                counter += 1
    raw.annotations.delete(idx_del)
    return raw


def removeAnnotationsOfLastNTrials(raw: mne.io.Raw, N: int) -> mne.io.Raw:
    """Remove the events for the first N trials.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw instance to be cleaned. Will be changed in-place.
    N : int
        Number of trials to be removed (from the end).

    Returns
    -------
    mne.io.Raw
        Cleaned raw.
    """
    n_ev2skip = N
    counter = 0
    idx_del = []
    for idx in range(1, len(raw.annotations)):
        sdesc = raw.annotations[-idx]["description"]
        if "Stimulus" in sdesc and counter < n_ev2skip:
            idx_del.append(-idx)
            print(f"Removing event: {sdesc}")
            if sdesc == "Stimulus/S 11":  # trial start
                counter += 1
    raw.annotations.delete(idx_del)
    return raw


def clean_subjects(sub_list_str: list, paths: PATHS):
    """Clean the original raw data from the most severe problems.

    Parameters
    ----------
    sub_list_str : list
        List of subject identifiers.
    paths : PATHS
        Object with path information.
    """
    for subID in sub_list_str:
        match subID:
            case "VR2FEM_S02":
                # Concatenate 3 files into 1:
                eeg_data = Path(paths.DATA_SUBJECTS, subID, "MainTask", "EEG")
                fname1 = Path(eeg_data, subID + ".vhdr")
                raw1 = mne.io.read_raw_brainvision(fname1, preload=True, verbose=False)

                fname2 = Path(eeg_data, subID + "_2.vhdr")
                raw2 = mne.io.read_raw_brainvision(fname2, preload=True, verbose=False)

                fname3 = Path(eeg_data, subID + "_3.vhdr")
                raw3 = mne.io.read_raw_brainvision(fname3, preload=True, verbose=False)

                mne.concatenate_raws([raw1, raw2, raw3])  # modifies raw1 in-place

                # Set montage and chan types for EOG chans:
                raw1 = setMontageChanTypes(raw1)

                # I manually checked that the concatenated file is ok, so no checks done here

                ## Export back to BV format does not work yet due to bug in pybv
                # mne.export.export_raw(Path(eeg_data, 'test.vhdr'), raw2)
                # path_olds = Path(eeg_data, 'single_files')
                # chkmkdir(path_olds)
                # shutil.move(fname1, path_olds)
                # shutil.move(fname2, path_olds)

                # so we store as .fif directly
                fname_out = Path(paths.DATA_00_RAWFIF, f"{subID}-raw.fif")
                raw1.save(fname_out, overwrite=True)
                print(f"Saved file to: {fname_out}")

            case "VR2FEM_S03":
                eeg_data = Path(paths.DATA_SUBJECTS, subID, "MainTask", "EEG")
                fname1 = Path(eeg_data, subID + ".vhdr")
                raw1 = mne.io.read_raw_brainvision(fname1, preload=True, verbose=False)
                # delete last 4 events & we lost 3 events due to bad connection:
                raw1 = removeAnnotationsOfLastNTrials(raw1, 4)

                # Set montage and chan types for EOG chans:
                raw1 = setMontageChanTypes(raw1)

                # so we store as .fif directly
                fname_out = Path(paths.DATA_00_RAWFIF, f"{subID}-raw.fif")
                raw1.save(fname_out, overwrite=True)
                print(f"Saved file to: {fname_out}")

            case "VR2FEM_S12":
                # delete first 3 events
                eeg_data = Path(paths.DATA_SUBJECTS, subID, "MainTask", "EEG")
                fname1 = Path(eeg_data, subID + ".vhdr")
                raw1 = mne.io.read_raw_brainvision(fname1, preload=True, verbose=False)
                raw1 = removeAnnotationsOfFirstNTrials(raw1, 3)

                # Set montage and chan types for EOG chans:
                raw1 = setMontageChanTypes(raw1)

                # so we store as .fif directly
                fname_out = Path(paths.DATA_00_RAWFIF, f"{subID}-raw.fif")
                raw1.save(fname_out, overwrite=True)
                print(f"Saved file to: {fname_out}")

            case "VR2FEM_S16":
                eeg_data = Path(paths.DATA_SUBJECTS, subID, "MainTask", "EEG")
                fname1 = Path(eeg_data, subID + ".vhdr")
                raw1 = mne.io.read_raw_brainvision(fname1, preload=True, verbose=False)

                # delete bad annotation/event (S133) at sample 2319057
                # (annot. #3156):
                raw1.annotations.delete(3156)

                # Set montage and chan types for EOG chans:
                raw1 = setMontageChanTypes(raw1)

                # so we store as .fif directly
                fname_out = Path(paths.DATA_00_RAWFIF, f"{subID}-raw.fif")
                raw1.save(fname_out, overwrite=True)
                print(f"Saved file to: {fname_out}")

            case "VR2FEM_S18":
                # delete first 1 events
                eeg_data = Path(paths.DATA_SUBJECTS, subID, "MainTask", "EEG")
                fname1 = Path(eeg_data, subID + ".vhdr")
                raw1 = mne.io.read_raw_brainvision(fname1, preload=True, verbose=False)
                raw1 = removeAnnotationsOfFirstNTrials(raw1, 1)

                # Set montage and chan types for EOG chans:
                raw1 = setMontageChanTypes(raw1)

                # so we store as .fif directly
                fname_out = Path(paths.DATA_00_RAWFIF, f"{subID}-raw.fif")
                raw1.save(fname_out, overwrite=True)
                print(f"Saved file to: {fname_out}")

            case "VR2FEM_S19":
                eeg_data = Path(paths.DATA_SUBJECTS, subID, "MainTask", "EEG")
                fname1 = Path(eeg_data, subID + ".vhdr")
                raw1 = mne.io.read_raw_brainvision(fname1, preload=True, verbose=False)
                # delete first 2 events & we lost 3 events due to bad connection:
                raw1 = removeAnnotationsOfFirstNTrials(raw1, 2)

                # Set montage and chan types for EOG chans:
                raw1 = setMontageChanTypes(raw1)

                # so we store as .fif directly
                fname_out = Path(paths.DATA_00_RAWFIF, f"{subID}-raw.fif")
                raw1.save(fname_out, overwrite=True)
                print(f"Saved file to: {fname_out}")

            case "VR2FEM_S21":
                eeg_data = Path(paths.DATA_SUBJECTS, subID, "MainTask", "EEG")
                fname1 = Path(eeg_data, subID + ".vhdr")
                raw1 = mne.io.read_raw_brainvision(fname1, preload=True, verbose=False)
                # delete first 2 events & we lost some events due to bad connection:
                raw1 = removeAnnotationsOfFirstNTrials(raw1, 2)

                # Set montage and chan types for EOG chans:
                raw1 = setMontageChanTypes(raw1)

                # so we store as .fif directly
                fname_out = Path(paths.DATA_00_RAWFIF, f"{subID}-raw.fif")
                raw1.save(fname_out, overwrite=True)
                print(f"Saved file to: {fname_out}")

            case "VR2FEM_S23":
                eeg_data = Path(paths.DATA_SUBJECTS, subID, "MainTask", "EEG")
                fname1 = Path(eeg_data, subID + ".vhdr")
                raw1 = mne.io.read_raw_brainvision(fname1, preload=True, verbose=False)
                # delete first 2 events & we lost some events due to bad connection:
                raw1 = removeAnnotationsOfFirstNTrials(raw1, 2)

                # Set montage and chan types for EOG chans:
                raw1 = setMontageChanTypes(raw1)

                # so we store as .fif directly
                fname_out = Path(paths.DATA_00_RAWFIF, f"{subID}-raw.fif")
                raw1.save(fname_out, overwrite=True)
                print(f"Saved file to: {fname_out}")

            case "VR2FEM_S24":
                # delete first 1 event
                eeg_data = Path(paths.DATA_SUBJECTS, subID, "MainTask", "EEG")
                fname1 = Path(eeg_data, subID + ".vhdr")
                raw1 = mne.io.read_raw_brainvision(fname1, preload=True, verbose=False)
                raw1 = removeAnnotationsOfFirstNTrials(raw1, 1)

                # Set montage and chan types for EOG chans:
                raw1 = setMontageChanTypes(raw1)

                # so we store as .fif directly
                fname_out = Path(paths.DATA_00_RAWFIF, f"{subID}-raw.fif")
                raw1.save(fname_out, overwrite=True)
                print(f"Saved file to: {fname_out}")

            case "VR2FEM_S27":
                eeg_data = Path(paths.DATA_SUBJECTS, subID, "MainTask", "EEG")
                fname1 = Path(eeg_data, subID + ".vhdr")
                raw1 = mne.io.read_raw_brainvision(fname1, preload=True, verbose=False)
                # delete first 2 events & we lost some events due to bad connection:
                raw1 = removeAnnotationsOfFirstNTrials(raw1, 4)

                # Set montage and chan types for EOG chans:
                raw1 = setMontageChanTypes(raw1)

                # so we store as .fif directly
                fname_out = Path(paths.DATA_00_RAWFIF, f"{subID}-raw.fif")
                raw1.save(fname_out, overwrite=True)
                print(f"Saved file to: {fname_out}")

            case "VR2FEM_S26":
                eeg_data = Path(paths.DATA_SUBJECTS, subID, "MainTask", "EEG")
                fname1 = Path(eeg_data, subID + ".vhdr")
                raw1 = mne.io.read_raw_brainvision(fname1, preload=True, verbose=False)
                # delete first 2 events & we lost some events due to bad connection:
                raw1 = removeAnnotationsOfFirstNTrials(raw1, 5)

                # Set montage and chan types for EOG chans:
                raw1 = setMontageChanTypes(raw1)

                # so we store as .fif directly
                fname_out = Path(paths.DATA_00_RAWFIF, f"{subID}-raw.fif")
                raw1.save(fname_out, overwrite=True)
                print(f"Saved file to: {fname_out}")

            case "VR2FEM_S29":
                eeg_data = Path(paths.DATA_SUBJECTS, subID, "MainTask", "EEG")
                fname1 = Path(eeg_data, subID + ".vhdr")
                raw1 = mne.io.read_raw_brainvision(fname1, preload=True, verbose=False)
                # delete first 2 events & we lost some events due to bad connection:
                raw1 = removeAnnotationsOfFirstNTrials(raw1, 2)

                # Set montage and chan types for EOG chans:
                raw1 = setMontageChanTypes(raw1)

                # so we store as .fif directly
                fname_out = Path(paths.DATA_00_RAWFIF, f"{subID}-raw.fif")
                raw1.save(fname_out, overwrite=True)
                print(f"Saved file to: {fname_out}")

            case "VR2FEM_S34":
                # Concatenate 2 files into 1:
                eeg_data = Path(paths.DATA_SUBJECTS, subID, "MainTask", "EEG")
                fname1 = Path(eeg_data, subID + ".vhdr")
                raw1 = mne.io.read_raw_brainvision(fname1, preload=True, verbose=False)

                fname2 = Path(eeg_data, subID + "_2.vhdr")
                raw2 = mne.io.read_raw_brainvision(fname2, preload=True, verbose=False)

                mne.concatenate_raws([raw1, raw2])  # modifies raw1 in-place

                # Set montage and chan types for EOG chans:
                raw1 = setMontageChanTypes(raw1)

                # I manually checked that the concatenated file is ok, so no checks done here

                ## Export back to BV format does not work yet due to bug in pybv
                # mne.export.export_raw(Path(eeg_data, 'test.vhdr'), raw2)
                # path_olds = Path(eeg_data, 'single_files')
                # chkmkdir(path_olds)
                # shutil.move(fname1, path_olds)
                # shutil.move(fname2, path_olds)

                # so we store as .fif directly
                fname_out = Path(paths.DATA_00_RAWFIF, f"{subID}-raw.fif")
                raw1.save(fname_out, overwrite=True)
                print(f"Saved file to: {fname_out}")

            # all others we store directly as .fif
            case _:
                eeg_data = Path(paths.DATA_SUBJECTS, subID, "MainTask", "EEG")
                fname1 = Path(eeg_data, subID + ".vhdr")
                raw1 = mne.io.read_raw_brainvision(fname1, preload=True, verbose=False)

                # Set montage and chan types for EOG chans:
                raw1 = setMontageChanTypes(raw1)

                # so we store as .fif directly
                fname_out = Path(paths.DATA_00_RAWFIF, f"{subID}-raw.fif")
                raw1.save(fname_out, overwrite=True)
                print(f"Saved file to: {fname_out}")


def main():
    """Run main."""
    paths = PATHS()

    sub_list_str = os.listdir(paths.DATA_SUBJECTS)

    # step through subjects one by one

    clean_subjects(sub_list_str=sub_list_str, paths=paths)
    return


if __name__ == "__main__":
    main()
