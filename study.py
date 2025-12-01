import numpy as np
import matplotlib.pyplot as plt

from beliefPropagation import performBeliefPropagation


codes = [
    "[[72, 12, 6]]",
    # "[[90, 8, 10]]",
    # "[[108, 8, 10]]",
    # "[[144, 12, 12]]",
    # "[[288, 12, 18]]",
]

trials = 10
physicalErrorRates = [0.01, 0.005, 0.001]
results = {}
for code in codes:
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
            
            detection, isSyndromeFound = performBeliefPropagation(code, error, initialBeliefs)
            
            if not isSyndromeFound:
                logical_error += 1
            else:
                residual = (detection + error) % 2
                
                # print(len(residual), residual)
                # print(Lx)
                # print(Lx.shape)
                
                syndromeLogic = (Lx @ residual) % 2
                
                if np.any(syndromeLogic):
                    logical_error += 1
        
        ler = logical_error / trials
        logicalErrorRates.append(ler)
        
    results[name] = logicalErrorRates
    
    
print(results)

for i in results:
    print(i)

plt.figure(figsize=(10, 6))
for name in results:
    plt.plot(physicalErrorRates, results[name], label= name)
plt.savefig("media/LERS")