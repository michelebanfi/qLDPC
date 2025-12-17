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
physicalErrorRates = [0.005] 
trials = 20000

np.random.seed(0)

stabilizer_spectra = {} 

for code_name in codes:
    oc = np.load(f'codes/{code_name}.npz')
    Hx = oc['Hx']
    Lx = oc['Lx']
    n = len(Hx[0])
    
    initialBeliefs = [np.log((1 - 0.005) / 0.005)] * n
    
    
    weights_found = []
    
    print(f"Simulating {code_name}...")
    for _ in tqdm.tqdm(range(trials)):
        error = (np.random.random(n) < 0.005).astype(int)
        syndrome = (error @ Hx.T) % 2
        
        detection, isSyndromeFound, llrs = performBeliefPropagationFast(Hx, syndrome, initialBeliefs, verbose=False)
        if not isSyndromeFound:
            detection = performOSD(Hx, syndrome, llrs, detection)
            
        residual = (detection + error) % 2
        syndromeLogic = (Lx @ residual) % 2
        
        if not np.any(syndromeLogic):
            if not np.array_equal(detection, error):
                w = np.sum(residual)
                weights_found.append(w)

    stabilizer_spectra[code_name] = weights_found

fig, axes = plt.subplots(1, len(stabilizer_spectra), figsize=(6*len(stabilizer_spectra), 5))
if len(stabilizer_spectra) == 1:
    axes = [axes]

for ax, (name, weights) in zip(axes, stabilizer_spectra.items()):
    ax.hist(weights, bins=np.arange(0, 50)-0.5, alpha=0.7, label=name, density=True, edgecolor='black')
    ax.set_title(f'Spectrum: {name}')
    ax.set_xlabel('Weight of Stabilizer Loop (Residual Error)')
    ax.set_ylabel('Probability')
    ax.set_xticks(np.arange(0, 50, 2))
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.savefig('media/spectrum.png', dpi=300)