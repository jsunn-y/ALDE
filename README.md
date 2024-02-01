# ALDE
Active Learning-assisted Directed Evolution (ALDE) is a package for simulating and performing active learning on combinatorial site-saturation libraries for protein engineering.

## Installation
To download, clone this repository
```
git clone https://github.com/jsunn-y/ALDE.git
```
To run ALDE, the relevant anaconda environment can be installed from `alde.yml`. To build this environment, run
```
cd ./ALDE
conda env create -f alde.yml
conda activate ALDE
```

## Production Runs
ALDE can be executed using the following command:
```
python execute_production.py
```

## Simulation Runs
To reproduce the computational simulations on complete landscapes:
```
python execute_simulation.py
```
## Tuning the execution script.
We reccomend running the scripts with default parameters to reproduce to results from our study. However, the following variables can be tuned in the execution script.

| Variable| Default Value | Description | 
|:-----------|:-------:|:----------------:|
| encoding | GB1_onehot |defines the design objective: the encoding for the protein, and the dataset and labels to use | 
| batch size | 96 | number of samples to query in each round of active learning |
| n_pseudorand_init | 96 | number of initial random samples | 
| budget | 384 | number of samples to query after the initial random intialization |
