# DDN-public:heterogneity-exp
Code for heterogeneity experiments with DDNs and ESNs. 

## Dependencies
pip install [requirements.txt](requirements.txt)

## Hyperparameter Optimisation
Code for NARMA hyperparameter optimisation using CMA-ES can be seen in 
[heterogeneity_experiment_NARMA.py](heterogeneity_experiment_NARMA.py), with the following parameters: 

- `-d`/`--delay`: use delay
- `-k <nr of clusters>`/`--clusters <nr of clusters>`: Number of sub-reservoirs to be used
- `-nr <nr of neurons>`/`--neurons <nr of neurons>`: Number of neurons to be used in the reservoir
- `-dd`/`--distributed_decay`: if added, optimize both mean and variance for decay/leak parameter, otherwise only mean, 
  with variance set fixed to 0.
- `-cd`/`--cluster_decay`: if added, decay/leak parameters are optimized for each sub-reservoir separately. Otherwise,
  single set of decay parameter for the whole network is used.
- `-fd`/`--fixed_delays`: if added, delays are left to their initial distributions and not optimized during evolution
- `-s`/`--suffix`: String is added to the end of the savefile. Useful to differentiate between repetitions of the 
  same experiments

Similarly, for the Mackey Glass hyperparameter optimization 
([heterogeneity_experiment_MG.py](heterogeneity_experiment_MG.py)), in addition to the previous parameters we have

- `-b`/`--bcm`: Include unsupervised local plasticity using BCM.
- `-e`/`--error-margin`: Validation error margin for computing blind prediction horizon.
- `-t`/`--tau_range`: Range from which random Mackey-Glass tau value is sampled during validation. To maintain a 
  constant tau value throughout all evaluations, simply limit the range to one value.
- `-n`/`--exponent_range`: Same as above, but for the Mackey-Glass exponent.

## Examples
[random_ddn_example.py](examples/random_ddn_example.py) shows how to create and use a DDN or ESN, how to train a 
and evaluate a readout layer, and how to visually simulate a network.

## Testing optimized networks
To test the best hyperparameter sets from an optimization run, use [testOptimized_MG.py](testOptimized_MG.py) `path` 
with the following parameters:
- `-r`/`--resamples`: number of times a reservoir is sampled from the best hyperparameter set for each tau.
- `-t`/`--testsamples`: number of time points in each test sequence.
- `-s`/`--testsequences`: number of sequences for each network evaluation.