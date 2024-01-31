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
