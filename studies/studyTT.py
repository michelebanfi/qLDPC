import numpy as np
import matplotlib.pyplot as plt
import tqdm

from decoding.beliefPropagation import performBeliefPropagation
from spaceTime import spaceTimeMatrix, spacetimeSyndrome


codes = [
    "[[72, 12, 6]]",
    "[[90, 8, 10]]",
    "[[108, 8, 10]]",
    "[[144, 12, 12]]",
    "[[288, 12, 18]]",
]

trials = 10
physicalErrorRates = np.logspace(-3.2, -1.3, 8)
results = {}
for code in tqdm.tqdm(codes):
    oc = np.load(f'codes/{code}.npz')
    name = code
    results[name] = {}
    code = oc['Hx']
    Lx = oc['Lx']
    distance = oc['distance']
    n = len(code[0])
    logicalErrorRates = []
    BPs_fault_rates = []
    BPs_miscorrected_rates = []
    incorrectable_rates = []
    
    H_spacetime = spaceTimeMatrix(code, distance)
    
    for errorRate in physicalErrorRates:
        
        tot_vars = H_spacetime.shape[1] # now, the beliefs are more, space + time
        initialBeliefs = [np.log((1 - errorRate) / errorRate)] * tot_vars
        logical_error = 0
        BPs_fault = 0
        BPs_miscorrected = 0
        incorrectable = 0
        
        
        for _ in range(trials):
            
            error, syndrome = spacetimeSyndrome(code, errorRate, distance)
            
            detection, isSyndromeFound, _ = performBeliefPropagation(H_spacetime, syndrome, initialBeliefs, verbose=False)
            
            num_vars = n * distance # distance here represents n_cycles. As the paper do
            
            # since we want just the data part
            
            detection_data = detection[:num_vars]
            
            # total corrections over time
            total_correction = np.sum(detection, axis=0) % 2
            
            residual = (total_correction + error) % 2
            
            if np.any((Lx @ residual) % 2):
                logical_error += 1
            
                    
                
        
        ler = logical_error / trials
        logicalErrorRates.append(ler)
        
    results[name]['ler'] = logicalErrorRates
    
    
# dump results to file for safekeeping
np.savez("data/LERS.npz", physicalErrorRates=physicalErrorRates, results=results)

plt.figure(figsize=(10, 6))
for name in results:
    plt.plot(physicalErrorRates, results[name]['ler'], label=name, marker='o')
    plt.grid(True)
    plt.xscale('log')
    plt.xlabel('Physical error rate')
    plt.ylabel('Logical error rate')
    plt.legend()
plt.savefig("media/LERS")

# grouped stacked bars: components = BPs_fault, BPs_miscorrected, incorrectable
# names = codes  # preserve the original order used to build `results`
# code_labels = ['72', '90', '108', '144', '288']  # n values for labeling
# x = np.arange(len(physicalErrorRates))
# n_codes = len(names)
# bar_width = 0.12  # narrower bars
# group_spacing = 0.15  # space between groups

# plt.figure(figsize=(14, 6))
# for i, name in enumerate(names):
#     bps_fault = np.array(results[name]['BPs_fault'])
#     bps_misc = np.array(results[name]['BPs_miscorrected'])
#     incorrect = np.array(results[name]['incorrectable'])
#     positions = x + i * (bar_width + 0.02)  # add small gap between bars

#     if i == 0:
#         plt.bar(positions, bps_fault, bar_width, label='BPs_fault', color='tab:blue')
#         plt.bar(positions, bps_misc, bar_width, bottom=bps_fault, label='BPs_miscorrected', color='tab:orange')
#         plt.bar(positions, incorrect, bar_width, bottom=bps_fault + bps_misc, label='incorrectable', color='tab:green')
#     else:
#         plt.bar(positions, bps_fault, bar_width, color='tab:blue')
#         plt.bar(positions, bps_misc, bar_width, bottom=bps_fault, color='tab:orange')
#         plt.bar(positions, incorrect, bar_width, bottom=bps_fault + bps_misc, color='tab:green')

# # Add code labels (n values) on top of each bar group
# for i, name in enumerate(names):
#     bps_fault = np.array(results[name]['BPs_fault'])
#     bps_misc = np.array(results[name]['BPs_miscorrected'])
#     incorrect = np.array(results[name]['incorrectable'])
#     total_height = bps_fault + bps_misc + incorrect
#     positions = x + i * (bar_width + 0.02)
    
#     for j, (pos, height) in enumerate(zip(positions, total_height)):
#         if height > 0:  # only label bars with data
#             plt.text(pos, height + 0.3, code_labels[i], ha='center', va='bottom', fontsize=7, rotation=90)

# # center x-ticks under each group and format ticks
# plt.xticks(x + (n_codes - 1) * (bar_width + 0.02) / 2, [f'{r:.1e}' for r in physicalErrorRates])
# plt.xlabel('Physical error rate')
# plt.ylabel('Number of BP failures (counts)')
# plt.grid(True, axis='y', linestyle='--', alpha=0.6)
# plt.legend(loc='upper left')
# plt.tight_layout()
# plt.savefig("media/BP_failures_stacked.png")