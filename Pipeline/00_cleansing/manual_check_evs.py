"""Manual check events

Similar content to `manual_check_raws.ipynb`, was used to inspect the raws (esp. in terms of event numbers, etc.). Pretty dirty but might have useful things. 

2022/10  --  Felix Klotzsche
"""

import os
from os import path as op
from pathlib import Path
from tabnanny import verbose
import numpy as np
import pandas as pd

from logging import exception, warning

import mne
from vr2fem_analyses.staticinfo import VR2FEMPATHS as PATHS

paths = PATHS()
subID_problematic = ['VR2FEM_S09']
sub_path = paths.DATA_SUBJECTS
sub_list_str = os.listdir(sub_path)
mega_df = pd.DataFrame(columns=['id'])
ev_coll = dict()
ev_dict_coll = dict()
for subID in sub_list_str:
    if subID in subID_problematic: continue
    p_data = Path(sub_path, subID, 'MainTask', 'EEG')
    fname = Path(p_data, subID + '.vhdr')
    raw = mne.io.read_raw_brainvision(fname, preload=False, verbose=False)
    evs, ev_dict = mne.events_from_annotations(raw, verbose=False)
    ev_coll[subID] = evs
    ev_dict_coll[subID] = ev_dict
    df = pd.DataFrame(evs, columns=['time', 'dur', 'id'])
    df = df.groupby('id').size().reset_index(name=subID)
    # df = df[(df['id']>110) & (df['id']<235)]
    if np.any(df.loc[(df['id']>110) & (df['id']<235), subID] != 31):
        mega_df = pd.merge(mega_df, df, how='outer')


subID = 'VR2FEM_S03'
sub_evs = ev_coll[subID]

# check whether every stim onset is followed by a stim offset event:
idx_stimon = np.argwhere((sub_evs[:,2]>110) & (sub_evs[:,2]<235))

if (np.any(sub_evs[idx_stimon + 1, 2] != 13)):
    print(f'{subID}:    NOt each stim onset followed by offset event.')
    idx_bad_stimons = np.argwhere(sub_evs[idx_stimon + 1, 2].flatten() != 13)
    idx_bads = idx_stimon[idx_bad_stimons.squeeze()]

    if idx_bads.ndim == 1:
        idx_bads = [idx_bads]

    for t in idx_bads:
        print(sub_evs[t[0]-5:t[0]+5,:])



stim_onsets = sub_evs[idx_stimon, :]
stim_offsets = sub_evs[(sub_evs[:,2]==13), :]
if (len(stim_onsets) != len(stim_offsets)): 
    raise ValueError(f'{subID}:    Unequal number of stim on- and offsets.')
diffs = (stim_offsets[:,0] - stim_onsets[:,0])  # [:-1,0])


# check if there's a `trial_events.csv` for every subject

for subID in sub_list_str:
    if subID in subID_problematic: continue
    p_data = Path('..', 'Data', 'Subjects', subID, 'MainTask', 'Unity', 'S001')
    ffiles = os.listdir(p_data)
    if not 'trial_results.csv' in ffiles:
        print(subID)


subID = 'VR2FEM_S03'
p03 = Path('..', 'Data', 'Subjects', subID, 'MainTask', 'EEG')

fname_old = Path(p03, 'VR2FEM_03.vhdr')
fname_new = Path(p03, 'VR2FEM_S03.vhdr')
mne_bids.copyfiles.copyfile_brainvision(fname_old, fname_new, verbose=True)
