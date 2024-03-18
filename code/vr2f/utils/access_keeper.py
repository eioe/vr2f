import os
from pathlib import Path

from vr2fem_analyses.staticinfo import COLORS, CONFIG, PATHS

paths = PATHS()

path_subs = paths.DATA_SUBJECTS
sub_list_str = sorted(os.listdir(path_subs))

subID = sub_list_str[0]

# Parallelize the following loop

# Copy the tracker

# for subID in sub_list_str:
#     print(f"Copying trackers for {subID}")
#     dest = Path(path_subs, subID, "MainTask", "Unity", "S001", "trackers")

#     source = (
#         "keeper:NEUROHUM/Subprojects/VRstereofem/"
#         + str(Path(*[p for p in dest.parts[-7:]]))
#     )

#     cmd = f"rclone copy {source} {dest} --include '*_eye_*.csv'"

#     os.system(cmd)


def copy_trackers(subID, path_subs=path_subs):
    print(f"Copying trackers for {subID}")
    dest = Path(path_subs, subID, "MainTask", "Unity", "S001", "trackers")

    source = "keeper:NEUROHUM/Subprojects/VRstereofem/" + str(Path(*[p for p in dest.parts[-7:]]))

    cmd = f"rclone copy {source} {dest} --include '*_eye_*.csv'"

    os.system(cmd)


# copy the trackers for all subjects in parallel using a pool of 14 workers
from multiprocessing import Pool

with Pool(14) as p:
    p.map(copy_trackers, sub_list_str)
