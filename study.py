import numpy as np
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
for code in codes:
    code = np.load(f'codes/{code}.npz')['Hx']
    n = len(code[0])
    
    for errorRate in physicalErrorRates:
        
        initialBeliefs = [np.log((1 - errorRate) / errorRate)] * n
        
        for _ in range(trials):
            
            error = (np.random.random(n) < errorRate).astype(int)
            
            print(f"ERRORR __________. __ _ __ _ _s{error}")
            detection, isSyndromeFound = performBeliefPropagation(code, error, initialBeliefs)
            