import sys
import os
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)
import Capacities.capacities as CAP
from utils import createNARMA10, createNARMA30
import numpy as np
import argparse
from datetime import date
import pickle as pkl

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Experiment configuration",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-n10", "--narma_10", action='store_true', help='Compute N-10 task caps')
    parser.add_argument("-n30", "--narma_30", action='store_true', help='Compute N-30 task caps')

    args = parser.parse_args()
    config = vars(args)

    tasks = {'narma_10': createNARMA10, 'narma_30': createNARMA30}
    n_tasks = 0
    for task_name in tasks:
        if config[task_name]:
            print("Computing task caps for {}".format(task_name))
            n_tasks += 1
            func = tasks[task_name]
            np.random.seed(42)
            inputs, outputs = func(30000)
            np.random.seed()
            inputs_rescaled = inputs * 4 - 1
            max_del = 60
            max_win = int(task_name[-2:]) + 2
            Citer = CAP.capacity_iterator(mindel=1, mindeg=1, maxdeg=2, minvars=1, maxvars=3, maxdel=max_del,
                                          delskip=100,
                                          m_delay=False,
                                          m_windowpos=False, m_window=False, m_powerlist=False, m_variables=False,
                                          m_degrees=False, minwindow=0, maxwindow=max_win)
            task_totalcap, task_allcaps, task_numcaps, task_nodes = Citer.collect(inputs_rescaled, outputs)
            results_dict = {
                'taskCap': task_allcaps,
                'totalCap': task_totalcap,
                'maxdel': max_del,
                'maxdeg': 2
            }
            results_filename = task_name + "_task_caps_" + str(date.today()) + ".p"
            try:
                results_path = "../results/ipc-results/" + results_filename
                print("Writing results to {}".format(results_path))
                with open(results_path, "wb") as f:
                    pkl.dump(results_dict, f)
            except Exception as e:
                results_path = "results/ipc-results/" + results_filename
                print("Failed, writing results to {}".format(results_path))
                with open(results_path, "wb") as f:
                    pkl.dump(results_dict, f)

    if n_tasks == 0:
        print('Neither N-10 nor N-30 selected, please add either -n10, -n30 or both flags to compute any task capacity.')
