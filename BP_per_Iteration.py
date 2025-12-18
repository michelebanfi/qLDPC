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

errorRate = 0.01

iterations = [10, 20, 30, 40, 50, 60]

code_labels = ['72', '90', '108', '144', '288']

trials = 50000

np.random.seed(0)

results = {}

for code in codes:
    oc = np.load(f"codes/{code}.npz")
    name = code
    code = oc['Hx']
    Lx = oc['Lx']
    n = len(code[0])
    results[name] = {}
    logicalErrors = []
    degeneracies = []
    
    for max_iter in iterations:
        initialBeliefs = [np.log((1 - errorRate) / errorRate)] * n
        
        logicalError = 0
        degenerateErrors = 0
        
        for _ in tqdm.tqdm(range(trials)):
            error = (np.random.random(n) < errorRate).astype(int)
            
            syndrome = (error @ code.T) % 2
            
            detection, isSyndromeFound, llrs = performBeliefPropagationFast(code, syndrome, initialBeliefs, verbose=False, maxIter=max_iter)
            
            if not isSyndromeFound:
                
                detection = performOSD(code, syndrome, llrs, detection)
                
            residual = (detection + error) % 2
            
            syndromeLogic = (Lx @ residual) % 2
            
            osdCheck = (detection @ code.T) % 2
            isValidOSD = np.array_equal(osdCheck, syndrome)
            
            if isValidOSD and not np.any(syndromeLogic) and (np.array_equal(detection, error) == False):
                degenerateErrors += 1
                
            if np.any(syndromeLogic):
                logicalError += 1
                
        logicalErrors.append(logicalError / trials)
        degeneracies.append(degenerateErrors / trials)
    
    results[name]['logicalErrors'] = logicalErrors
    results[name]['degeneracies'] = degeneracies
    results[name]['iterations'] = iterations
        
np.savez('data/BP_per_Iteration.npz', results=results)

# for each code, create a column and plot logical error rate vs iterations
fig, axes = plt.subplots(2, len(results), figsize=(6*len(results), 10))
properties = ['logicalErrors', 'degeneracies']
property_labels = ['Logical Error Rate', 'Degeneracies']

for row, (prop, label) in enumerate(zip(properties, property_labels)):
    for col, (name, data) in enumerate(results.items()):
        ax = axes[row, col]
        ax.plot(data['iterations'], data[prop], marker='o')
        ax.set_title(f'Code {name}' if row == 0 else '')
        ax.set_xlabel('Max BP Iterations')
        ax.set_ylabel(label)
        ax.set_yscale('log')
        ax.grid(True)

plt.tight_layout()
plt.savefig('media/BP_per_Iteration.png')