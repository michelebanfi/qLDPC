"""
GPU-optimized version of paperResults.py
Uses batched belief propagation for significant speedup.

Install requirements:
    pip install cupy-cuda12x  # For CUDA 12.x (adjust for your version)
    # OR pip install cupy-cuda11x for CUDA 11.x
    
For Apple Silicon (M1/M2/M3), use JAX instead
"""

import numpy as np
import matplotlib.pyplot as plt
import tqdm
import time

from decoding.beliefPropagationGPU import (
    performBeliefPropagationBatch,
    generate_errors_and_syndromes_batch,
    GPU_AVAILABLE
)
from decoding.OSD import performOSD

codes = [
    "[[72, 12, 6]]",
    "[[90, 8, 10]]",
    "[[108, 8, 10]]",
    "[[144, 12, 12]]",
    "[[288, 12, 18]]",
]

np.random.seed(0)
rng = np.random.default_rng(0)

trials = 100000
physicalErrorRates = [0.01, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001, 0.0009]

# Batch size for GPU processing - adjust based on your GPU memory
# Larger = faster but more memory
BATCH_SIZE = 1000

results_OSD = {}

names = codes  # preserve the original order used to build `results`
code_labels = ['72', '90', '108', '144', '288']  # n values for labeling
x = np.arange(len(physicalErrorRates))
n_codes = len(names)
bar_width = 0.12  # narrower bars
group_spacing = 0.15  # space between groups

print(f"GPU Available: {GPU_AVAILABLE}")
print(f"Running {trials} trials with batch size {BATCH_SIZE}")
print("=" * 60)

total_start = time.time()

for code_name in tqdm.tqdm(codes, desc="Processing codes"):
    print(f"\nProcessing code: {code_name}")
    oc = np.load(f'codes/{code_name}.npz')
    name = code_name
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
        rate_start = time.time()
        
        initialBeliefs = np.array([np.log((1 - errorRate) / errorRate)] * n)
        logical_error = 0
        BPs_fault = 0
        BPs_miscorrected = 0
        incorrectable = 0
        degenerateErrors = 0
        
        # Process in batches
        num_batches = (trials + BATCH_SIZE - 1) // BATCH_SIZE
        
        for batch_idx in range(num_batches):
            batch_start = batch_idx * BATCH_SIZE
            batch_end = min((batch_idx + 1) * BATCH_SIZE, trials)
            current_batch_size = batch_end - batch_start
            
            # Generate batch of errors and syndromes
            # Code capacity error model: apply error twice and XOR
            errors1, syndromes1 = generate_errors_and_syndromes_batch(
                code, errorRate, current_batch_size, rng
            )
            errors2, syndromes2 = generate_errors_and_syndromes_batch(
                code, errorRate, current_batch_size, rng
            )
            
            # XOR the two error patterns (code capacity model)
            errors = (errors1 + errors2) % 2
            syndromes = (syndromes1 + syndromes2) % 2
            
            # Run batched BP
            detections, converged, llrs_batch = performBeliefPropagationBatch(
                code, syndromes, initialBeliefs, maxIter=75
            )
            
            # Process results for this batch
            for i in range(current_batch_size):
                error = errors[i]
                syndrome = syndromes[i]
                detection = detections[i]
                llrs = llrs_batch[i]
                isSyndromeFound = converged[i]
                
                if not isSyndromeFound:
                    # BP failed, run OSD
                    detection = performOSD(code, syndrome, llrs, detection)
                
                # This is the XOR between the actual error and the detected error
                # We are simulating the correction of the error
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
        
        rate_time = time.time() - rate_start
        print(f"  p={errorRate}: LER={ler:.6f}, degeneracies={degenerateErrors}, time={rate_time:.1f}s")
        
    results_OSD[name]['ler'] = logicalErrorRates
    results_OSD[name]['BPs_fault'] = BPs_fault_rates
    results_OSD[name]['BPs_miscorrected'] = BPs_miscorrected_rates
    results_OSD[name]['incorrectable'] = incorrectable_rates
    results_OSD[name]['degeneracies'] = degeneracies

total_time = time.time() - total_start
print(f"\n{'=' * 60}")
print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")

np.savez('data/BPOSD_GPU.npz', results=results_OSD)

# Plotting
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
    
plt.savefig('media/BPOSD_GPU_results.png', dpi=300)
print(f"Plot saved to media/BPOSD_GPU_results.png")
