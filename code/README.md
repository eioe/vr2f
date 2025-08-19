# vrstereofem – **code**

    Last update:    August 18, 2025
    Author:         Felix Klotzsche

***


## HPC

SLURM scripts for running relevant processing steps on a compute cluster. 


## notebooks
For some analysis and esp. visualization steps, we used Jupyter notebooks.  
- behavioral results
- plotting decoding performance 
- plotting spatial patterns of decoders
- stats on decoding
- eye tracking analyses
- ...

## RScripts
Only 1 script which reproduces a behavioral figure with the data from Gilbert et al. (2019). 

## vr2f
Main part of the package. Contains workhorse scripts.


## Configs

Paths to data, parameter settings, etc. are stored in the config script: `./code/vr2f/staticinfo.py`


## Python

To install the research code as package, run the following code in the project root directory:

```shell
pip install -e "."
```

