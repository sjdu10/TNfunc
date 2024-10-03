from funcs import *
import numpy as np
import concurrent.futures
n = 14
total_depth = 20
# chi_list = [2, 4, 8, 16, 32, 64]
chi = -1
J = 0.7
g = 0.5
h = 0.5
entropies_exact_dict = {}

def save_exact_results(n, J, g, h, chi, entropies_exact_dict):
    # Save the results to a txt file
    print(f'Saving results for exact with chi={chi}')
    t_S_dict = {}
    entropies_exact = entropies_exact_dict[chi]
    for t in range(len(entropies_exact)):
        t_S_dict[t+1] = entropies_exact[t]
        
    # dump the dictionary
    import pickle
    with open(f'./results/volume_law/exact_n={n}_J={J}_g={g}_h={h}.pkl', 'wb') as f:
        pickle.dump(t_S_dict, f)
    


if type(h) == tuple:
    h_list = [0]*n
    for i in range(n):
        h_list[i] = np.random.uniform(h[0], h[1])
    hin = tuple(h_list)
else:
    hin = h
    
with concurrent.futures.ProcessPoolExecutor() as executor:
    # Map each execution of compute_entanglement_entropy to a process
    results = executor.map(compute_entanglement_entropy_exact_volume_law, 
                        [n] * total_depth,  # n is constant across calls
                        range(1, total_depth + 1),  # depth varies from 1 to total_depth
                        # [-1] * total_depth,  # chi is constant across calls
                        # [direction] * total_depth,  # direction is constant
                        [J] * total_depth,  # J is constant
                        [g] * total_depth,  # g is constant
                        [hin] * total_depth)  # h is constant

    entropies_exact = []
    traces_exact = []
    amp_times_exact = []
    for entropy_list, trace, amp_time in results:
        entropies_exact.append(entropy_list)
        traces_exact.append(trace)
        amp_times_exact.append(amp_time)
    entropies_exact_dict[chi] = entropies_exact
    
    save_exact_results(n, J, g, h, chi, entropies_exact_dict)

