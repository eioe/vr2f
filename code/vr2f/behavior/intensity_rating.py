"""
Functionalities relevant for analyzing the data from the intensity rating task.
"""
import os
import pandas as pd
from pathlib import Path
from vr2f.staticinfo import PATHS



def get_intensity_ratings():
  paths = PATHS()
  path_subs = paths.DATA_SUBJECTS
  sub_list_str = sorted(os.listdir(path_subs))

  df_ratings = pd.DataFrame()
  for sub_id in sub_list_str:
    if sub_id in ["VR2FEM_S18"]:  # No data for this participant
      continue
    path_in = Path(path_subs, sub_id, "AvatarReview", "Unity", "S001")
    fname = Path(path_in, "trial_results.csv")
    df = pd.read_csv(fname)
    df = (df.rename(columns={"ppid": "sub_id",
                            "selected_manikin": "intensity",
                            "emo": "emotion",
                            "stereo": "viewcond"})
            .loc[:, ["sub_id", "trial_num", "viewcond", "avatar", "emotion",
                    "intensity", "response_time"]]
            .replace({"emotion": {1: "neutral", 2: "happy", 3: "angry", 4: "surprised"},
                      "viewcond": {0: "mono", 1: "stereo"}})
          )
    df_ratings = df if df_ratings.empty else pd.concat([df_ratings, df], axis=0)
  return df_ratings
