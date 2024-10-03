from funcs import *
import numpy as np
import concurrent.futures
import sys
n = int(sys.argv[1])
total_depth = 50
init_depth = 1
# chi_list = [2, 4, 8, 16, 32, 64]
chi = -1
J = 0.2
g = 0.2
h = 0.2
entropies_exact_dict = {}
amp_times_exact_dict = {}
def save_exact_results(n, J, g, h, chi, entropies_exact_dict):
    # Save the results to a txt file
    print(f'Saving results for exact with chi={chi}')
    with open(f'./results/exact_n={n}_J={J}_g={g}_h={h}.txt', 'a') as f:
        entropies_exact = entropies_exact_dict[chi]
        for t in range(len(entropies_exact)):
            f.write(f'depth={t+init_depth}, S_A={entropies_exact[t]}, chi={chi}\n')


if type(h) == tuple:
    h_list = [0]*n
    for i in range(n):
        h_list[i] = np.random.uniform(h[0], h[1])
    hin = tuple(h_list)
else:
    hin = h
    
with concurrent.futures.ProcessPoolExecutor() as executor:
    # Map each execution of compute_entanglement_entropy to a process
    results = executor.map(compute_entanglement_entropy_exact, 
                        [n] * total_depth,  # n is constant across calls
                        range(init_depth, total_depth + init_depth),  # depth varies from 1 to total_depth
                        # [-1] * total_depth,  # chi is constant across calls
                        # [direction] * total_depth,  # direction is constant
                        [J] * total_depth,  # J is constant
                        [g] * total_depth,  # g is constant
                        [hin] * total_depth)  # h is constant

    entropies_exact = []
    traces_exact = []
    amp_times_exact = []
    for entropy, trace, amp_time in results:
        entropies_exact.append(entropy)
        traces_exact.append(trace)
        amp_times_exact.append(amp_time)
    entropies_exact_dict[chi] = entropies_exact
    amp_times_exact_dict[chi] = amp_times_exact
    
    save_exact_results(n, J, g, h, chi, entropies_exact_dict)

