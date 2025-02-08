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
- `cd`/`--cluster_decay`: if added, decay/leak parameters are optimized for each sub-reservoir separately. Otherwise,
  single set of decay parameter for the whole network is used.
- `s`/`--suffix`: String is added to the end of the savefile. Useful to differentiate between repetitions of the 
  same experiments

## Examples
[random_ddn_example.py](examples/random_ddn_example.py) shows how to create and use a DDN or ESN, how to train a 
and evaluate a readout layer, and how to visually simulate a network.