# Study VR2F: **EEG-decodability of facial expressions and their stereoscopic depth cues in immersive virtual reality**

`[Last update: August 18, 2025]`

***
    Period:     2024-03 - 2025-08
    Status:     preprint | submitted
    Author(s):  Felix Klotzsche
    Contact:    klotzsche@cbs.mpg.de

***

<!--📖 **Publication:**  [Klotzsche, et al. (2025, ...)](https://...) -->

💽 **Data:** https://doi.org/10.17617/3.KJGEZQ 

📑 **Preprint:** [ tbd ]  

![fig1: exp design](/vr2f/resources/images/fig1.png)
## Abstract
Face perception typically occurs in three-dimensional space, where stereoscopic depth cues enrich the perception of facial features. Yet, most neurophysiological research on face processing relies on two-dimensional displays, potentially overlooking the role of stereoscopic depth information. Here, we combine immersive virtual reality (VR), electroencephalography (EEG), and eye tracking to examine the neural representation of faces under controlled manipulations of stereoscopic depth. Thirty-four participants viewed computer-generated faces with neutral, happy, angry, and surprised expressions in frontal view under monoscopic and stereoscopic viewing conditions. Using time-resolved multivariate decoding, we show that EEG signals in immersive VR conditions can reliably differentiate facial expressions. Stereoscopic depth cues elicited a distinct and decodable neural signature, confirming the sensitivity of our approach to depth-related processing. Yet, expression decoding remained robust across depth conditions, indicating that under controlled frontal viewing, the neural encoding of facial expressions is invariant to binocular depth cues. Eye tracking showed that expression-related gaze patterns contained comparable information but did not account for neural representations, while depth information was absent in gaze patterns—consistent with dissociable representational processes. Our findings demonstrate the feasibility of EEG-based neural decoding in fully immersive VR as a tool for investigating face perception in naturalistic settings and provide new evidence for the stability of expression representations across depth variations in three-dimensional viewing conditions.

![fig2: decoding results](/vr2f/resources/images/fig2.png)

## Instructions

### Install vr2f research code as package

In case, there is no project-related virtual / conda environment yet, create one for the project:

```shell
conda create -n vr2f python=3.10
```

And activate it:

```shell
conda activate vr2f
```

Then install the code of the research project as python package:

```shell
# assuming your current working dircetory is the project root
pip install -e .
```
**Note**: The `-e` flag installs the package in editable mode,
i.e., changes to the code will be directly reflected in the installed package.

## Get the data
Download the data set (or parts of it) from [EDMOND](https://doi.org/10.17617/3.CQ2VXX):  
https://doi.org/10.17617/3.CQ2VXX   
There are data-readme files on EDMOND which explain which files/folders you might want to work with (the entire data set is quite spacious).
Put it into the hierarchy next to the `code/` folder in a folder called `data/`. Your working tree should look like this:  
<details>
<summary> click here </summary>

![working tree screenshot](/vr2f/resources/images/workingtree.png)
</details>


## Run analyses
### EEG
---
#### Preprocessing
In case, you want to start with the raw, continuous files, you should use the data sets in `data/eeg/00_raw_fif`.  To run my preprocessing, you'd run:  
 1. [make_epochs.py](./code/vr2f/preprocessing/make_epochs.py)
 2. [check_epochs.py](./code/vr2f/preprocessing/check_epochs.py)
 3. [run_ica.py](./code/vr2f/preprocessing/run_ica.py)
 4. [reject_ica.py](./code/vr2f/preprocessing/reject_ica.py)
 5. [run_autoreject.py](./code/vr2f/preprocessing/run_autoreject.py)  

#### Decoding
For the **decoding**, you should use [decoding.py](./code/vr2f/decoding/decoding.py) from the command line (or use the [HPC script](./code/HPC/DECOD_SS.sh)).  
Example:
```bash
python3.10 ./code/vr2f/decoding/decoding.py 0 emotion all
```
Will run the decoding of the emotional expressions in trials from both (all) viewing conditions (mono and stereo) from the first participant. 
<details>
<summary>
Arguments
</summary>

| Pos | Name              | Type | Choices / Values                                       | Description              |
| --- | ----------------- | ---- | ------------------------------------------------------ | ------------------------ |
| 0   | `participant_idx` | int  | `0–33`                                                 | Participant index.       |
| 1   | `contrast`        | str  | `emotion`, `emotion_pairwise`, `viewcond`, `avatar_id` | What to decode.          |
|     |                   |      | · `emotion` – multiclass facial expression             |                          |
|     |                   |      | · `emotion_pairwise` – all binary pairs of expressions |                          |
|     |                   |      | · `viewcond` – depth condition (mono vs stereo)        |                          |
|     |                   |      | · `avatar_id` – stimulus identity                      |                          |
| 2   | `viewcond`        | str  | `mono`, `stereo`, `all`                                | Trial subset to include. |

</details>
<br>

For **cross-decoding** (only implemented for decoding the emotional expression), run [crossdecoding.py](./code/vr2f/decoding/crossdecoding.py) from the command line (or via HPC). It only takes a single CL argument -> participant index (0–34). 

### Source reconstruction
To calculate the source activation timecourses, run [decoding_calc_source_timecourses.py](vr2f/code/vr2f/decoding/decoding_calc_source_timecourses.py)  
(1 CL argument -> participant idx). 

![fig 3: source reconstruction results](/vr2f/resources/images/fig3.png)

#### Plot the results & calculate stats
Check out the notebook [decoding_plotters.ipynb](vr2f/code/notebooks/decoding/decoding_plotters.ipynb).  
For nice & interactive 3Dplots of the sources, use [plot_stc.py](vr2f/code/vr2f/decoding/plot_stc.py).

---
### Behavior
Use the notebook [behavior_analysis.ipynb](vr2f/code/notebooks/behavior/behavior_analysis.ipynb). 


---
### Eye Tracking
Use and follow instructions in the notebooks in [/code/notebooks/eyetracking](vr2f/code/notebooks/eyetracking).  

![fig 4: eye tracking results](/vr2f/resources/images/fig4.png)

---


## Publications

[tbc]

## Contributors/Collaborators
[Felix Klotzsche*](https://bsky.app/profile/flxklotz.bsky.social "Follow on Bluesky"),
[Ammara Nasim](https://www.uni-bamberg.de/allgpsych/wissenschaftliche-mitarbeitende/ammara-nasim/ "University website")
[Simon M. Hofmann](https://bsky.app/profile/smnhfmnn.bsky.social "Follow on Bluesky"),
[Arno Villringer](https://www.cbs.mpg.de/employees/villringer "Institute's webpage"),
[Vadim V. Nikulin](https://www.cbs.mpg.de/employees/nikulin "Institute's webpage"),
[Werner Sommer](https://www.psychology.hu-berlin.de/de/mitarbeiter/4489 "University website"),
[Michael Gaebler](https://www.michaelgaebler.com "Personal webpage")  
*\* corresponding author*

This study was conducted at the [Max Planck Institute for Human Cognitive and Brain Sciences](https://www.cbs.mpg.de/en "Go the institute website")
as part of the [NEUROHUM project](https://neurohum.cbs.mpg.de "Go the project site").

[![NEUROHUM Logo](https://neurohum.cbs.mpg.de/assets/institutes/headers/cbsneurohum-desktop-en-cc55f3158c5428ca969719e99df1c4f636a0662c1c42e409d476328092106060.svg)](https://neurohum.cbs.mpg.de "Go the project site")