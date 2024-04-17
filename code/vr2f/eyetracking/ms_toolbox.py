from pathlib import Path

import numpy as np
import pandas as pd

from vr2f.staticinfo import PATHS


def vecvel(data, srate, vel_type=2):
  """
  Compute velocity from eye-tracking data.

  Parameters
  ----------
  data : np.array
      Eye-tracking data with 2 columns with vertical and
      horizontal coordinates.
  srate : int
      Sampling rate.
  vel_type : int, optional
      Type of velocity computation. The default is 2.

  Returns
  -------
  vel : np.array
      Velocity data. Same format as input data.

  """
  n = len(data)
  vel = np.zeros((n, 2))

  if vel_type == 2:
    vel[2:(n-2),:] = srate * (data[4:,] + data[3:(n-1),] - data[1:(n-3),] - data[:(n-4),])/6
    vel[1,:] = srate * (data[2,:] - data[0,:])/2
    vel[n-2,:] = srate * (data[n-1,:] - data[n-3,:])/2
  else:
    vel[1:(n-2),:] = srate * (data[2:,] - data[:(n-1),])/2

  return vel


def microsacc(data, srate, vfac = 5, mindur = 3):  # noqa: C901, PLR0915
  """
  Identify microsaccades based on velocity thresholding and computes their characteristics.

  Parameters
  ----------
  data : np.ndarray
      The input data containing eye position signals. Expected to be an Nx2 numpy array
      where N is the number of samples, the first column is the horizontal (X) component,
      and the second column is the vertical (Y) component.
  srate : float
      The sampling rate of the data in Hz.
  vfac : float, optional
      The velocity factor used for thresholding. Determines the multiplication factor for the median-based threshold.
      Default is 5.
  mindur : int, optional
      The minimum duration of a microsaccade in samples. Microsaccades shorter than this duration are not considered.
      Default is 3.

  Returns
  -------
  Tuple[pd.DataFrame, List[float]]
      A tuple containing two elements:
      - A pandas DataFrame with the detected microsaccades. Columns include 'idx_onset'
      (onset index), 'idx_offset' (offset index), 'vpeak' (peak velocity), 'vec_x' (horizontal
      displacement), 'vec_y' (vertical displacement), 'amp_x' (horizontal amplitude), and 'amp_y' (vertical amplitude).
      - A list with two elements representing the radius thresholds for the horizontal and vertical
      components, respectively.

  Raises
  ------
  Exception
      If the computed standard deviation for the horizontal or vertical velocity is less than a
      very small threshold (indicating no movement), an exception is raised to indicate that the
      velocity calculation might be incorrect.

  Notes
  -----
  The function computes the velocity of eye movements and uses a velocity-based threshold to
  identify (micro)saccades. It then calculates various parameters of these (micro)saccades, such as
  peak velocity and amplitude.

  """
  # Compute velocity
  vel = vecvel(data, srate)

  medx = np.median(vel[:,0])
  msdx = np.sqrt(np.median((vel[:,0] - medx)**2))
  medy = np.median(vel[:,1])
  msdy = np.sqrt(np.median((vel[:,1] - medy)**2))

  if msdx < 1e-10:
    msdx = np.sqrt(np.mean(vel[:,0]**2) - (np.mean(vel[:,0]))**2)
    if msdx < 1e-10:
      raise Exception("msdx < realmin in microsacc.R")
  if msdy < 1e-10:
    msdy = np.sqrt(np.mean(vel[:,1]**2) - (np.mean(vel[:,1]))**2)
    if msdy < 1e-10:
      raise Exception("msdy < realmin in microsacc.R")

  radiusx = vfac * msdx
  radiusy = vfac * msdy
  radius = [radiusx, radiusy]

  # Apply test criterion: elliptic treshold
  test = (vel[:,0]/radiusx)**2 + (vel[:,1]/radiusy)**2
  indx = np.where(test>1)[0]

  # Determine saccades
  n = len(indx)
  nsac = 0
  sac = []
  dur = 0
  onset = 0
  i = 0

  while i < n-1:
    if (indx[i+1] - indx[i]) == 1:
      dur += 1
    else:
      if dur >= mindur:
        nsac += 1
        offset = i
        sac.append([int(indx[onset]), int(indx[offset]), 0, 0, 0, 0, 0])
      onset = i + 1
      dur = 1
    i = i + 1

  # Check minimum duration for last microsaccade
  if dur >= mindur:
    nsac = nsac + 1
    offset = i
    sac.append([int(indx[onset]), int(indx[offset]), 0, 0, 0, 0, 0])

  if nsac > 0:
    # Compute peak velocity, horiztonal and vertical components
    for s in range(nsac):
      # Onset and offset for saccades
      onset = sac[s][0]
      offset = sac[s][1]
      idx = range(onset, offset+1)
      # Saccade peak velocity (vpeak)
      vpeak = max(np.sqrt(vel[idx,0]**2 + vel[idx,1]**2))
      sac[s][2] = vpeak
      # Saccade vector (dx,dy)
      dx = data[offset,0] - data[onset,0]
      dy = data[offset,1] - data[onset,1]
      sac[s][3] = dx
      sac[s][4] = dy
      # Saccade amplitude (dX,dY)
      minx = np.min(data[idx,0])
      maxx = np.max(data[idx,0])
      miny = np.min(data[idx,1])
      maxy = np.max(data[idx,1])
      ix1 = np.argmin(data[idx,0])
      ix2 = np.argmax(data[idx,0])
      iy1 = np.argmin(data[idx,1])
      iy2 = np.argmax(data[idx,1])
      dist_x = np.sign(ix2-ix1)*(maxx-minx)
      dist_y = np.sign(iy2-iy1)*(maxy-miny)
      sac[s][5] = dist_x
      sac[s][6] = dist_y

    sac_arr = np.array(sac)
    # Convert to DataFrame
    sac_df = pd.DataFrame(sac_arr,
                          columns = ["idx_onset", "idx_offset", "vpeak", "vec_x",
                                     "vec_y", "amp_x", "amp_y"])
    sac_df = sac_df.astype({"idx_onset": int, "idx_offset": int})

  else:
    sac_df = pd.DataFrame(columns = ["idx_onset", "idx_offset", "vpeak", "vec_x",
                                     "vec_y", "amp_x", "amp_y"])
  return sac_df, radius



def import_demo_data():
  paths = PATHS()
  fpath = Path(paths.DATA_ET, "ms_toolbox_demodata")
  fname = "f01.005.dat"
  data = np.loadtxt(Path(fpath, fname))

  fname_res = "f01.005_results.csv"
  results = pd.read_csv(Path(fpath, fname_res), header=None,
                        names=["idx_onset", "idx_offset", "vpeak", "vec_x",
                               "vec_y", "amp_x", "amp_y"])
  # adjust indices to 0-based
  results["idx_onset"] = results["idx_onset"] - 1
  results["idx_offset"] = results["idx_offset"] - 1
  return data, results


def test_microsacc():
  dat, res = import_demo_data()
  idx = range(3000,4500)
  xl = dat[idx,1:3]
  sac, radius = microsacc(xl, 500)
  
  if not np.allclose(sac.copy().to_numpy().flatten(),
                     res.copy().to_numpy().flatten()):
    raise ValueError("Test failed.")
  else:
    print("Test succeeded.")
