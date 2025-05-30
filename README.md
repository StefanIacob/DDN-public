# dist-delay-net
This repository contains the code, data and analysis discussed in the distance-based delay networks
paper [Exploiting Signal Propagation Delays to Match Task Memory Requirements in Reservoir Computing](https://doi.org/10.3390/biomimetics9060355).

Analysis for the paper [Memory–Non-Linearity Trade-Off in Distance-Based Delay Networks](https://www.mdpi.com/2313-7673/9/12/755) 
is based on the same evolved networks, 
and can be found in [IPC-Analysis.ipynb](analysis-notebooks/IPC-Analysis.ipynb), with examples on how to generate 
the paper figures. 

## Branches
- Analysis for the initial [2022 paper on DDNs](https://link.springer.com/chapter/10.1007/978-3-031-21753-1_21) is shown 
in the [2022-paper branch](https://github.com/StefanIacob/DDN-public/tree/2022-paper) as it requires an older 
version of the code.
- Analysis for the [2023 paper on ADDNs](https://link.springer.com/chapter/https://ieeexplore.ieee.org/abstract/document/10191901) is shown 
in the [2023-paper branch](https://github.com/StefanIacob/DDN-public/tree/2023-paper) as it requires an older 
version of the code.

## Dependencies
pip install requirements.txt

## Hyperparameter Optimisation
Code for CMA-ES hyperparameter optimisation is included for two benchmark tasks

### NARMA
Optimizing DDNs or baseline ESNs for NARMA tasks can be done by running [NARMA_experiment.py](NARMA_experiment.py) with
the appropriate parameters. Results from previous optimization runs can be found
in [NARMA-10_results_23](results/NARMA-10_results_23) and [NARMA-30_results_23](results/NARMA-30_results_23), also used in
the [analysis notebook](analysis-notebooks/MC-Analysis.ipynb) to generate paper figures.

### Mackey-Glass
Optimizing DDNs, ADDNs, adaptive ESNs or baseline ESNs for Mackey-Glass tasks can be done by
running [mg_experiment.py](mg_experiment.py) with the appropriate parameters. Results from previous optimization runs
can be found in [ADDN_further_experiments](results/ADDN-2024-papers), also used in
the [analysis notebook](analysis-notebooks/MC-Analysis.ipynb) to generate paper figures.

## Testing
Example code for how to test best optimized networks shown in [testOptimized.py](examples/testOptimized_NARMA.py)

## Paper Figures and data analysis
All paper figures can be reproduced using the included notebooks:
- [Exploiting Signal Propagation Delays to Match Task Memory Requirements in Reservoir Computing](https://www.mdpi.com/2313-7673/9/6/355): [analysis notebook](analysis-notebooks/MC-Analysis.ipynb).
- [Memory–Non-Linearity Trade-Off in Distance-Based Delay Networks](https://www.mdpi.com/2313-7673/9/12/755): 
[analysis notebook](analysis-notebooks/IPC-Analysis.ipynb)

## Visual Example
See [visual_example.py]([examples/visual_example.py]) for a DDN or ESN (either random or optimized) simulated with a
GUI, visually showing differences in network responses to various inputs. 

## IPC
Information processing capacity (IPC) is used in the
[2024 paper](https://www.mdpi.com/2313-7673/9/12/755).
To understand how this is computed, and how the [capacities tools](Capacities/capacities.py) are used for DDNs, have a 
look at 
[capacitiesDDNExample.ipynb](examples/capacitiesDDNExample.ipynb).

Code for computing IPC can be found in [Capacities](Capacities) folder, written by prof. dr. Joni Dambre, published in 
[Dambre, Joni, David Verstraeten, Benjamin Schrauwen, and Serge Massar. “Information Processing Capacity of Dynamical Systems.” Scientific Reports 2, no. 1 (July 19, 2012): 514.
](https://doi.org/10.1038/srep00514). An example for how the original capacities code is used can be seen in 
[Capacity example.ipynb](Capacities/Capacity%20example.ipynb).