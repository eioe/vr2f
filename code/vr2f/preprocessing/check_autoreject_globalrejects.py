"""
Check AutoReject (AR) rejection rates and flag subjects exceeding a threshold.

This script scans the post-ICA AR directory defined in :class:`vr2f.staticinfo.PATHS`
and evaluates, per subject, whether the proportion of rejected epochs exceeds the
project-level threshold in :class:`vr2f.staticinfo.CONSTANTS`.
"""

import pandas as pd
from pathlib import Path

from vr2f.staticinfo import PATHS, CONSTANTS

paths = PATHS()
constants = CONSTANTS()

fpath = Path(paths.DATA_03_AR)
fpath_rejlogs = Path(fpath, "rejectlogs")
fpath_rejdata = Path(fpath, "rejecteddata")
files = sorted(fpath.glob("*-ar.fif"))
files_rejlogs = sorted(fpath_rejlogs.glob("*.csv"))

print(f"Too many rejected trials (more than {constants.AR_PROP_EPOS_TO_REJECT_SUB * 100}%):")

for rejlog, data in zip(files_rejlogs, files, strict=True):
  if not rejlog.stem.split('-')[0] == data.stem.split('-')[0]:
    raise ValueError(f"Mismatch between {rejlog} and {data}")
  rej_log_df = pd.read_csv(rejlog)
  is_rejected = rej_log_df.badEpoch.sum() > 720 * constants.AR_PROP_EPOS_TO_REJECT_SUB
  print(f"{rejlog.stem.split('-')[0]}: {is_rejected}")


