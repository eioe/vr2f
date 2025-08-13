import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import gaussian_filter

from vr2f.staticinfo import CONSTANTS, PATHS

def plot_image(av_name, emotion, ax, invert=False, alpha=1):
  paths = PATHS()
  img = plt.imread(Path(paths.STIMULIIMAGES, f"{av_name}_{emotion.capitalize()}.png"))
  cutval = 205
  yOffset = 40
  im2 = img[cutval + yOffset : -cutval, cutval + int(yOffset/2) : -(cutval+int(yOffset/2)), :]
  height, width, _ = im2.shape
  # Overlay the image onto the plot, centered on the axes
  origin = "lower" if invert else "upper"
  ax.imshow(im2, extent=[-5, 5, -5, 5], alpha=alpha, origin=origin)


def get_saliency_map(fixation_df, query_str):
  # Calculate  continuous saliency maps
  # https://link.springer.com/article/10.3758/s13428-012-0226-9#Sec6 

  # create all possible combinations of theta and phi (binned in 0.1 dva)
  q_vert = np.round(np.arange(-5, 5.1, 0.1), 1)
  q_hor = q_vert.copy()
  px_df = pd.DataFrame({"center_phi_rounded": np.repeat(q_hor, len(q_vert)),
        "center_theta_rounded": np.tile(q_vert, len(q_hor))})

  fixations_binned = (fixation_df
                  .assign(center_phi_rounded = lambda x: x["center_phi"].round(1))
                  .assign(center_theta_rounded = lambda x: x["center_theta"].round(1))
                  .query(query_str)
                  .groupby(["center_phi_rounded", "center_theta_rounded"], as_index=True)
                  .size()
                  .rename("fixation_count")
                  .to_frame()
                  .reset_index()
                  .merge(px_df, how="right", on=["center_phi_rounded", "center_theta_rounded"])
                  .fillna(0)
          )

  # convolve with a gaussian kernel (sd=1; see leMeur & Baccion, 2013)
  saliency_map = (fixations_binned
                  .pivot_table(index="center_phi_rounded",
                          columns="center_theta_rounded",
                          values="fixation_count")
  )

  saliency_map = gaussian_filter(saliency_map, sigma=1)
  return(saliency_map)