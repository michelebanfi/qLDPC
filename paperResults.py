import numpy as np
import matplotlib.pyplot as plt
import tqdm

from decoding.beliefPropagation import performBeliefPropagationFast
from decoding.OSD import performOSD

codes = [
    "[[72, 12, 6]]",
    "[[90, 8, 10]]",
    "[[108, 8, 10]]",
    "[[144, 12, 12]]",
    "[[288, 12, 18]]",
]

np.random.seed(0)

trials = 100000
# physicalErrorRates = np.logspace(-3.2, -1.3, 8)
physicalErrorRates = [0.01, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001, 0.0009]
results_OSD = {}

names = codes  # preserve the original order used to build `results`
code_labels = ['72', '90', '108', '144', '288']  # n values for labeling
x = np.arange(len(physicalErrorRates))
n_codes = len(names)
bar_width = 0.12  # narrower bars
group_spacing = 0.15  # space between groups


for code in tqdm.tqdm(codes):
    oc = np.load(f'codes/{code}.npz')
    name = code
    results_OSD[name] = {}
    code = oc['Hx']
    Lx = oc['Lx']
    distance = oc['distance']
    n = len(code[0])
    logicalErrorRates = []
    BPs_fault_rates = []
    BPs_miscorrected_rates = []
    incorrectable_rates = []
    degeneracies = []
    
    for errorRate in physicalErrorRates:
        
        initialBeliefs = [np.log((1 - errorRate) / errorRate)] * n
        logical_error = 0
        BPs_fault = 0
        BPs_miscorrected = 0
        incorrectable = 0
        degenerateErrors = 0
        
        
        for _ in range(trials):
            
            #### CODE CAPACITY ERROR MODEL ####
            # non-trivial pythonic way to generate random bitstring with given error rate
            error1 = (np.random.random(n) < errorRate).astype(int)
            error2 = (np.random.random(n) < errorRate).astype(int)
            error = (error1 + error2) % 2
            
            syndrome = (error @ code.T) % 2
            
            #### SIMPLE PHENOMENOLOGICAL ERROR MODEL ####
            # measurementError = (np.random.random(len(syndrome)) < errorRate).astype(int)
            # syndrome = (syndrome + measurementError) % 2
            
            detection, isSyndromeFound, llrs = performBeliefPropagationFast(code, syndrome, initialBeliefs, verbose=False, maxIter=75)
            
            if not isSyndromeFound:
                # logical_error += 1
                # BPs_fault += 1
                
                detection = performOSD(code, syndrome, llrs, detection)
                
                

            # This is the XOR, between the actual error and the detected error. We are simulating the correction of the error
            residual = (detection + error) % 2
            
            syndromeLogic = (Lx @ residual) % 2
            
            osd_syndrome_check = (detection @ code.T) % 2
            is_valid_osd = np.array_equal(osd_syndrome_check, syndrome)
            
            if is_valid_osd and not np.any(syndromeLogic) and (np.array_equal(detection, error) == False):
                degenerateErrors += 1
            
            if np.any(syndromeLogic):
                logical_error += 1

                error_weight = np.sum(error)
                if error_weight < (distance // 2):
                    BPs_miscorrected += 1
                else:
                    incorrectable += 1
                                    
        
        ler = logical_error / trials
        logicalErrorRates.append(ler)
        BPs_fault_rates.append(BPs_fault)
        BPs_miscorrected_rates.append(BPs_miscorrected)
        incorrectable_rates.append(incorrectable)
        degeneracies.append(degenerateErrors)
        
    results_OSD[name]['ler'] = logicalErrorRates
    results_OSD[name]['BPs_fault'] = BPs_fault_rates
    results_OSD[name]['BPs_miscorrected'] = BPs_miscorrected_rates
    results_OSD[name]['incorrectable'] = incorrectable_rates
    results_OSD[name]['degeneracies'] = degeneracies
    
np.savez('data/BPOSD.npz', results=results_OSD)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for name in results_OSD:
    axes[0].plot(physicalErrorRates, results_OSD[name]['degeneracies'], label=name, marker='o')
    axes[0].grid(True)
    axes[0].legend()
    axes[0].set_title('Degeneracies')
    
for name in results_OSD:
    axes[1].plot(physicalErrorRates, results_OSD[name]['ler'], label=name, marker='o')
    axes[1].grid(True)
    axes[1].legend()
    axes[1].set_yscale('log')
    axes[1].set_xscale('log')
    axes[1].set_title('Logical errors')
    
plt.savefig('media/BPOSD_results.png', dpi=300)