import os
from multiprocessing import Pool
from pathlib import Path

from vr2f.staticinfo import COLORS, CONFIG, PATHS


def copy_trackers(sub_id, path_subs=None):
    if path_subs is None:
        paths = PATHS()
        path_subs = paths.DATA_SUBJECTS
    print(f"Copying trackers for {sub_id}")
    dest = Path(path_subs, sub_id, "MainTask", "Unity", "S001", "trackers")

    source = "keeper:NEUROHUM/Subprojects/VRstereofem/" + str(Path(*[p for p in dest.parts[-7:]]))

    cmd = f"rclone copy {source} {dest} --include '*_eye_*.csv'"

    os.system(cmd)


def copy_to_keeper(source_file: str, dest_folder: str, keeper_root = "keeper:NEUROHUM/Subprojects/VRstereofem/"):
    """Copy a file via rclone to keeper."""
    source = keeper_root + source_file
    cmd = f"rclone copy {source} {dest_folder}"
    os.system(cmd)  # noqa: S605
    print(cmd)


def copy_all_trackers():
    paths = PATHS()

    path_subs = paths.DATA_SUBJECTS
    sub_list_str = sorted(os.listdir(path_subs))

    with Pool(14) as p:
        p.map(copy_trackers, sub_list_str)
