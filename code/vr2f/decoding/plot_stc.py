import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import mne
from vr2f.staticinfo import PATHS


def parse_args():
    parser = argparse.ArgumentParser(description="Make Screenshot.")
    parser.add_argument("--hemi", default="rh", choices=["lh", "rh"], help="Hemisphere to plot")
    parser.add_argument(
        "--contrast",
        default="neutral_vs_happy_vs_angry_vs_surprised",
        help="Contrast to plot",
    )
    parser.add_argument("--no-save", dest="save", action="store_false",
                        default=True, help="Do not save the image(s)")
    return parser.parse_args()

def load_labels(hemi="lh"):
    """
    Load HCP MMP and aparc sub parcellation labels for the specified hemisphere.

    Returns
    -------
    labels : list of mne.Label
        List of labels loaded from the annotation.
    hemi : str
        Hemisphere for which to load labels. Must be "lh" or "rh". Default is "lh".

    """
    subjects_dir = mne.datasets.sample.data_path() / "subjects"
    mne.datasets.fetch_hcp_mmp_parcellation(subjects_dir=subjects_dir, verbose=True)
    mne.datasets.fetch_aparc_sub_parcellation(subjects_dir=subjects_dir, verbose=True)

    labels = mne.read_labels_from_annot(
        "fsaverage", "HCPMMP1_combined", hemi, subjects_dir=subjects_dir
        )

    return labels

def get_label(name, labels):
    return next(label for label in labels if name in label.name)


def save_brain_screenshot(brain, hemi, view, time, contrast, show=False):
    paths = PATHS()
    img = brain.screenshot(mode="rgba")
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.axis("off")  # Hide axes
    ax.set_title("")
    if show:
        plt.show()
    # save image
    path_fig = Path(paths.FIGURES, "decod", "stc", contrast, view)
    path_fig.mkdir(parents=True, exist_ok=True)

    img_file = path_fig / f"contrast-{contrast.replace('_', '-')}_time-{time}_view-{view}_hemi-{hemi}.png"
    fig.savefig(img_file, transparent=True, bbox_inches='tight', pad_inches=0)
    print(f"Saved brain screenshot to {img_file}")


def plot_areas(brain, labels, col_dict):
    for label in labels:
        brain.add_label(label, borders=False, alpha=0.5, color=col_dict[label.name])
    brain.show()


def main():
    args = parse_args()
    hemi = args.hemi
    contrast = args.contrast
    save_img = args.save

    times = [0.1, 0.17, 0.26, 0.5]  # P1, N170, EPN, LPC
    views = ["lat", "par"]

    subjects_dir = mne.datasets.sample.data_path() / "subjects"
    path_data = Path("Data").resolve()
    paths = PATHS()


    path_stcs = Path(f"{paths.DATA_04_DECOD_SENSORSPACE}/{contrast}/roc_auc_ovr/patterns/src_timecourses/mean/")
    for f in path_stcs.iterdir():
        print(f)

    fname = f"stc_{contrast.replace('_', '-')}"

    stc = mne.read_source_estimate(str(path_stcs / fname))
    brain = stc.plot(
        subject="fsaverage",
        hemi=hemi,
        surface="inflated",
        views="lat",
        initial_time=0,
        # cortex="white",
        time_viewer=True,
        smoothing_steps=10,
        time_unit="s",
        subjects_dir=subjects_dir,
        alpha=0.9,
        colormap="RdBu_r",
        # transparent=False,
        show_traces=False,
        background=(1, 1, 1, 0),
        colorbar=False,
        title="",
        clim={"kind": "percent", "lims": (97.5, 99, 99.9)},  # (90, 95, 99.9)},  # },
        verbose=False,
    )

    labels = load_labels(hemi)
    ffa_label = get_label("Ventral Stream", labels)
    pvc_label = get_label("Primary Visual Cortex", labels)
    evc_label = get_label("Early Visual Cortex", labels)
    mtp_label = get_label("MT+", labels)
    spc_label = get_label("Superior Parietal Cortex", labels)
    label_kws = dict(
        borders=True,
        alpha=0.8,
    )
    labels = [ffa_label, pvc_label, evc_label, mtp_label, spc_label]
    col_dict = {
        ffa_label.name: "magenta",
        pvc_label.name: "cyan",
        evc_label.name: "yellow",
        mtp_label.name: "green",
        spc_label.name: "blue",
    }
    for view in views:
        if view == "par":
            for label in labels:
                brain.add_label(label, **label_kws, color=col_dict[label.name])

        for time in times:
            brain.set_time(time)
            brain.show_view(view)
            print(f"Showing: {view} view at time {time}")
            print(f"Contrast: {contrast}")
            input("Press enter to move on.")
            if save_img:
                save_brain_screenshot(brain, hemi, view, time, contrast)

        brain.set_time(0)
        plot_areas(brain, labels, col_dict)
        if save_img:
            save_brain_screenshot(brain, hemi, view, 0, "areas")

if __name__ == "__main__":
    main()

