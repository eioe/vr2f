import os
from pathlib import Path

from vr2fem_analyses.staticinfo import CONFIG, PATHS, COLORS


paths = PATHS()

path_subs = paths.DATA_SUBJECTS
sub_list_str = sorted(os.listdir(path_subs))

subID = sub_list_str[0]


for subID in sub_list_str:
    dest = Path(path_subs, subID, "MainTask", "Unity", "S001")

    source = (
        "keeper:NEUROHUM/Subprojects/VRstereofem/"
        + str(Path(*[p for p in dest.parts[-6:]]))
        + "/trial_results.csv"
    )

    cmd = f"rclone copy {source} {dest}"

    os.system(cmd)