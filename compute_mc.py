import pickle as pkl
import numpy as np
import argparse
from utils import genome_memory_capacity

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process saved evolution results dict path and output path")
    parser.add_argument("input_filename", type=str, help="The path to the file to process.")
    parser.add_argument("output_filename", type=str, help="output file path")
    parser.add_argument("-i", "--interval", type=int, default=10,
                        action="store", help="interval between generations for which MC is computed")

    args = parser.parse_args()
    filename = args.input_filename
    output_path = args.output_filename
    save_every = args.interval
    if output_path[-2:] != '.p':
        output_path = output_path + '.p'

    print("loading file: " + filename )
    with open(filename, 'rb') as f:
        results_dict = pkl.load(f)
    print("Dictionary keys:")
    print(results_dict.keys())
    alphas = [10e-7, 10e-5, 10e-3]
    if "alpha grid" in results_dict.keys():
        alphas = results_dict["alpha grid"]
    print("Validation scores array dimensions:")
    val_results = results_dict['validation performance']
    print(val_results.shape)
    narma_task = False
    if len(val_results.shape) == 4:
        narma_task = True

    n_sequences_unsupervised = results_dict['number of sequences']['unsupervised']
    if narma_task:
        scores = np.mean(np.min(val_results, axis=-1), axis=-1)
    else:
        scores = np.mean(val_results, axis=-1)

    ex_net = results_dict['example net']
    max_gen = val_results.shape[0]

    print("Computing MC...")
    m_cap_evo = []
    for gen in np.arange(0, max_gen, save_every):
        pop = results_dict['parameters'][gen, :, :]
        i = np.argmin(scores[gen])  # Best candidate
        print("generation" + str(gen) + ", selected candidate " + str(i))
        m_cap = genome_memory_capacity(pop[i, :], ex_net, 100, 1000, None,
                                       400, alphas, eval_reps=5, genome_reps=5,
                                       n_sequences_unsupervised=n_sequences_unsupervised)
        m_cap_evo.append(m_cap)

    m_cap_evo = np.array(m_cap_evo)
    data_to_save = {
        'memory capacities': m_cap_evo,
        'evolution results': results_dict
    }

    print("saving file: " + output_path)
    with open(output_path, 'wb') as f:
        pkl.dump(data_to_save, f)
