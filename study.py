import numpy as np
import matplotlib.pyplot as plt
import tqdm

from beliefPropagation import performBeliefPropagation


codes = [
    "[[72, 12, 6]]",
    "[[90, 8, 10]]",
    "[[108, 8, 10]]",
    "[[144, 12, 12]]",
    "[[288, 12, 18]]",
]

#trials = 50000
trials = 1000
physicalErrorRates = np.logspace(-3, -1.3, 8)
results = {}
for code in tqdm.tqdm(codes):
    oc = np.load(f'codes/{code}.npz')
    name = code
    code = oc['Hx']
    Lx = oc['Lx']
    n = len(code[0])
    logicalErrorRates = []
    
    for errorRate in physicalErrorRates:
        
        initialBeliefs = [np.log((1 - errorRate) / errorRate)] * n
        logical_error = 0
        
        for _ in range(trials):
            
            # non-trivial pythonic way to generate random bitstring with given error rate
            error = (np.random.random(n) < errorRate).astype(int)
            
            detection, isSyndromeFound = performBeliefPropagation(code, error, initialBeliefs, verbose=False)
            
            if not isSyndromeFound:
                logical_error += 1
            else:
                # This is the XOR, between the actual error and the detected error. We are simulating the correction of the error
                residual = (detection + error) % 2
                
                syndromeLogic = (Lx @ residual) % 2
                
                if np.any(syndromeLogic):
                    logical_error += 1
        
        ler = logical_error / trials
        logicalErrorRates.append(ler)
        
    results[name] = logicalErrorRates
    
# dump results to file for safekeeping
np.savez("data/LERS.npz", physicalErrorRates=physicalErrorRates, results=results)

plt.figure(figsize=(10, 6))
for name in results:
    plt.plot(physicalErrorRates, results[name], label=name, marker='o')
    plt.grid(True)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(-3, -1))
    plt.xlabel('Physical error rate')
    plt.ylabel('Logical error rate')
    plt.legend()
plt.savefig("media/LERS")