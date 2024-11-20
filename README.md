# ALDE
[![DOI](https://zenodo.org/badge/687191886.svg)](https://doi.org/10.5281/zenodo.14194551)
![image](figs/ALDE_workflow.png)
Active Learning-assisted Directed Evolution (ALDE) is a package for simulating and performing active learning on combinatorial site-saturation libraries for protein engineering.

## Installation
To download, clone this repository
```
git clone https://github.com/jsunn-y/ALDE.git
```
To run ALDE, the relevant anaconda environment can be installed from `alde.yml`. To build this environment, run
```
cd ./ALDE
mkdir results
conda env create -f alde.yml
conda activate ALDE
```
The encodings and fitness data used in our study can be downloaded from [here](https://zenodo.org/records/12196802), and should be unzipped to replace the empty `data` folder. If you are using your own training data, you can skip this download. Install time should be on the order of seconds to minutes.

## Production Runs
Production runs should be used for a wet-lab ALDE campaign. It can also be used to reproduce the results from the protoglobin (ParPgb) wet-lab ALDE campaign demonstrated in our study.

First generate the design space (all variants and encodings to search across) for the combinatorial library by specifying `nsites` (number of residues being simultanesouly mutated) and `name` (name of the project) within `generate_domain.py`. Scripts for the protoglobin wet-lab campaign in our study are given as an example. Then run the script:
```
python generate_domain.py --name=ParPgb --nsites=5
```
Outputs from `generate_domain.py` will appear in the folder `/data/{name}`. The two outputs are `all_combos.csv`, which contains the order of the combos in the design space (domain) as strings and `onehot_x.pt`, which is a torch tensor containing the respective onehot encodings in the same order as the list of combos. For a given ALDE campaign, generating the domain only needs to be executed once. Afterward, training data should be uploaded into that folder as `fitness.csv`, where a 'Combo' column specifies the protein sequence at the mutated residues and a separate column specifies the respective fitness value. Note that unseen labels in the domain are filled in as 0 values as placeholders, so the calculated regret is meaningless for the production runs. Run time for a typical computer with a GPU or several CPUs will be on the order of minutes.

For every round of training and prediction, ALDE can be executed using the following command:
```
python execute_production.py --name=ParPgb \
--data_csv=fitness_round1_training.csv \
--obj_col=Diff \
--output_path=results/ParPgb_production/round1/ \
--batch_size=90 \
--seed_index=0
```
Within the argparser, `name` must be specified to correspond to the relevant data folder. The data should be loaded as a dataframe from `/data/{name}/{data_csv}` containing sequences and their corresponding fitness values. `obj_col` should specify the column containing fitness values to be optimized. In this csv, the sequence column should be labeled as 'Combo'. By default, predictions will be made using onehot encodings, for 4 different models and 3 different acquisition functions. The `path` variable should be updated to where the results will be saved. `batch_size` specifies the number of samples to query for the next round of screening. Script for the protoglobin wet-lab campaign in our study is given as an example.

Relevant output will be found in the `results` path and will end with `indices.pt`, corresponding to the indices of the training and newly queried variants, in order. These indices can then be mapped to the `all_combos.csv` file to determine the sequences of the corresponding variants to be further tested. An example of doing this for the ParPgb data is given in `output2variants.ipynb`.

## Simulation Runs

To reproduce the computational simulations on complete landscapes (GB1 and TrpB) from our study:
```
python execute_simulation.py
```
ALDE will be simulated for 2 combinatorially complete datasets, 4 encodings, 3 models, and 3 acquisition functions, as outlined in our study.

To reproduce the baseline with a single round of ALDE, run the following command:
```
python execute_simulation.py --n_pseudorand_init=384 --budget=96 --output_path=results/384+96_baseline/
```

## Results Files
The general format for results file prefix is: `{model name}-DO-{dropout rate}-{kernel}-{acquisition function}-{end layer dimensions of architecture}_{index for the random seed}`. Different encodings and datasets will be organized in separate folders. Each ALDE campaign (for a given encoding, model, and acquisition function in the prefix) will produce the following results files as torch tensors:

| File Suffix | Description | 
|:-----------|:----------------:|
| `indices.pt` | One output for the campaign. The indices for the queried protein sequences, which can be referenced by the order given in the fitness dataframes: `_fitness.csv` file for simulation runs, or by the `all_combos.csv` file for production runs|
| `_{n}mu.pt` | Posterior mean values from the model, for all samples in the design space. Outputted each time a model has been trained in the campaign, where n indicates the number of queried samples used to train the model. Used during anlysis of uncertainty quantification. For the production runs, only the samples from the training set are valid.|
| `_{n}sigma.pt` | Posterior standard deviation values from the model, for all samples in the design space. Outputted each time a model has been trained in the campaign, where n indicates the number of queried samples used to train the model. Used during anlysis of uncertainty quantification. For the production runs, only the samples from the training set are valid. |

## Tuning the execution script
We recommend running the scripts with default parameters to reproduce to results from our study. However, the following variables can be tuned in the execution script.

| Variable| Default Value | Description | 
|:-----------|:-------:|:----------------:|
| protein | GB1 |defines the design objective: the dataset and labels to use: supports 'GB1' or 'TrpB' for simulation runs| 
| encoding | onehot |defines how the protein sequences are featurized: supports 'AAindex', 'onehot', 'georgiev', or 'ESM2' for simulation runs| 
| batch size | 96 | number of samples to query in each round of active learning |
| n_pseudorand_init | 96 | number of initial random samples | 
| budget | 384 | number of total samples to query, not including the initial random samples | 
| path | results/5x96_simulations/ | path to save results |
| runs | 70 | number of times to repeat the simulation from random initialization |
| index | 0 | index of the first run, which determines which seeds are used from rndseed.txt |
| kernel | RBF | the kernel for models with GPs, only supports radial basis function (RBF) |
| mtype | DNN_ENSEMBLE | model tpye: supports one of 'BOOSTING_ENSEMBLE', 'GP_BOTORCH', 'DNN_ENSEMBLE', and 'DKL_BOTORCH,' which is an ensemble of boosting models, a Gaussian process, an ensemble of neural networks, and a deep kernel, respectively|
| acq_fn | GREEDY | acquisition function: supports of 'GREEDY', 'UCB', and 'TS,' which are greedy, upper confidence bound, and thompson sampling, respectively|
| lr | 0.001 | learning rate, which affects the training of all models except 'BOOSTING_ENSEMBLE' |
| num_simult_jobs | 1 | number of simulations to perform simultaneously, for multiprocessing |
| arc | [encoding_dim, 30, 30, 1] | architecture of the model, where the first element is the encoding dimension, middle elements are hidden layer sizes, and the final layer is size 1. BOOSTING_ENSEMBLE and GP_BOTORCH do not have hidden layers|
| fname |  | filename for saving results, autogenerated |
| xi | 4 | $\xi$ term, only applies to upper confidence bound |
| dropout | 0 | training dropout rate |
| mcdrouput | 0 | test time dropout rate, not currently supported  |
| train_iter | 300 | number of iterations for training, which affects the training of all models except 'BOOSTING_ENSEMBLE'  |
| verbose | 2 | how much to report to the terminal  |

## Analysis
The raw outputs from the ALDE simulations can be summarized into a dataframe using `analysis/tabulate_results.ipynb`. Example output from tabulation of the simulations are provided in `analysis/all_results.csv`. Afterward, example visualizations of the tabulated results can be performed with `analysis/visualization.ipynb` and `analysis/ParPgb/trajectory_visualization.ipynb`.

The complete processed fitness data from our wet-lab campaign is provided in `analysis/ParPgb_fitness_all.csv`.
