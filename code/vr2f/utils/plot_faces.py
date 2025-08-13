"""Plot stimuli faces as PDF files."""
import matplotlib.pyplot as plt

from pathlib import Path

from vr2f.staticinfo import CONSTANTS, PATHS


def plot_faces_pdf():
  """Plot stimuli faces as PDFs."""
  cm = CONSTANTS().CM
  cond_dict = CONSTANTS().COND_DICT
  paths = PATHS()

  # choose avatar IDs
  av_names = (
      cond_dict["avatar_id"][2],
      cond_dict["avatar_id"][2],
      cond_dict["avatar_id"][1],
      cond_dict["avatar_id"][1],
      cond_dict["avatar_id"][3],
      cond_dict["avatar_id"][3],
  )
  # choose the respective emotions
  emotions = ("angry", "neutral", "happy", "neutral", "angry", "surprised")

  for av_name, emotion in zip(av_names, emotions, strict=True):
      fig, ax = plt.subplots(1, figsize=(6 * cm, 6 * cm))
      path_images = Path(paths.STIMULIIMAGES)
      img = plt.imread(Path(path_images, f"{av_name}_{emotion.capitalize()}.png"))
      cutval = 205
      im2 = img[cutval + 35 : -cutval, cutval + 35 : -cutval, :]
      height, width, _ = im2.shape
      # Overlay the image onto the plot, centered on the axes
      ax.imshow(im2, alpha=1, aspect="auto")
      # hide the axes and the ticks and labels
      ax.axis("off")
      # remove the white space around the image
      plt.tight_layout()
      # set background to transparent
      fig.patch.set_facecolor("black")
      fig.patch.set_alpha(1.0)
      ax.patch.set_facecolor("black")
      ax.patch.set_alpha(1.0)
      # save as pdf
      fpath = Path(paths.STIMULIIMAGES, "pdf")
      fpath.mkdir(parents=True, exist_ok=True)
      fname = Path(fpath, f"{av_name}_{emotion.capitalize()}.pdf")
      plt.savefig(
          fname,
          dpi=300,
          # bbox_inches='tight',  # noqa: ERA001
          pad_inches=0,
          transparent=True,
      )



if __name__ == "__main__":
  plot_faces_pdf()
