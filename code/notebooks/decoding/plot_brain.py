import mne
import pyvista 

pyvista.OFF_SCREEN = False

print("Fuck my life")

Brain = mne.viz.get_brain_class()

subjects_dir = mne.datasets.sample.data_path() / "subjects"
mne.datasets.fetch_hcp_mmp_parcellation(subjects_dir=subjects_dir, verbose=True)

mne.datasets.fetch_aparc_sub_parcellation(subjects_dir=subjects_dir, verbose=True)


labels = mne.read_labels_from_annot(
    "fsaverage", "HCPMMP1", "lh", subjects_dir=subjects_dir
)
brain = Brain(
    "fsaverage",
    "lh",
    "inflated",
    subjects_dir=subjects_dir,
    cortex="low_contrast",
    background="white",
    size=(800, 600),
    show=False
)


brain.add_label(aud_label, borders=False)

print("Still fucked")

brain.save_image("mybrain.png")
brain.close()
