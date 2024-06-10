import numpy as np

def spherical2cart(theta, phi, input_in_degree=False):
  if input_in_degree:
    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)
  y = np.sin(phi) * -1
  x = np.sin(theta) * np.cos(phi)
  z = np.cos(theta) * np.cos(phi)
  return x, y, z

def gazevec_intersec_xyplane(vec, z, tolerance=1e-12):
  if (vec[2] < 0):
    raise ValueError("Gaze vector must be pointing forward.")
  if (vec[2] < tolerance):
    raise ValueError("Gaze vector is parallel to the xy-plane.")
  r = z / vec[2]
  x = r * vec[0]
  y = r * vec[1]
  return x, y

  
def angle_in_plane(x, y):
  ang = np.arctan2(x, y)
  if ang < 0:
    ang = 2*np.pi + ang
  return ang

def angle_from_spherical(theta, phi, input_in_degree=True, output_in_degree=False):
  x, y, z = spherical2cart(theta, phi, input_in_degree=input_in_degree)
  x, y = gazevec_intersec_xyplane([x, y, z], 1)
  if output_in_degree:
    return angle_in_plane(x, y) * 180 / np.pi
  return angle_in_plane(x, y)
