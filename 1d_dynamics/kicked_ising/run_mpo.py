from funcs import *
import numpy as np
import concurrent.futures
n = 14
total_depth = 20
chi_mps_list = [2,4,8,32,64]
direction = 'mpo'
J = 0.7
g = 0.5
h = 0.5
entropies_mps_dict = {}
amp_times_mps_dict = {}
def save_mps_results(n, J, g, h, chi_mps, entropies_mps_dict):
    # Save the results to a txt file
    print(f'Saving results for MPS with chi={chi_mps}')
    with open(f'./results/MPO_n={n}_J={J}_g={g}_h={h}.txt', 'a') as f:
        entropies_mps = entropies_mps_dict[chi_mps]
        for t in range(len(entropies_mps)):
            f.write(f'depth={t+1}, S_A={entropies_mps[t]}, chi={chi_mps}\n')

for chi_mps in chi_mps_list:
    # Parallel execution
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Map each execution of compute_entanglement_entropy to a process
        results = executor.map(compute_entanglement_entropy, 
                            [n] * total_depth,  # n is constant across calls
                            range(1, total_depth + 1),  # depth varies from 1 to total_depth
                            [chi_mps] * total_depth,  # chi_mps is constant across calls
                            [direction] * total_depth,  # direction is constant
                            [J] * total_depth,  # J is constant
                            [g] * total_depth,  # g is constant
                            [h] * total_depth)  # h is constant

        entropies_mps = []
        traces_mps = []
        amp_times_mps = []
        for entropy, trace, amp_time in results:
            entropies_mps.append(entropy)
            traces_mps.append(trace)
            amp_times_mps.append(amp_time)
        entropies_mps_dict[chi_mps] = entropies_mps
        amp_times_mps_dict[chi_mps] = amp_times_mps
        
        save_mps_results(n, J, g, h, chi_mps, entropies_mps_dict)