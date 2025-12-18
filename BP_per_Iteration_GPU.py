"""
GPU-optimized version of BP_per_Iteration.py
Uses batched belief propagation for significant speedup.

Install requirements:
    pip install cupy-cuda12x  # For CUDA 12.x (adjust for your version)
    # OR pip install cupy-cuda11x for CUDA 11.x
    
For Apple Silicon (M1/M2/M3), use JAX instead - see BP_per_Iteration_JAX.py
"""

import numpy as np
import matplotlib.pyplot as plt
import tqdm
import time

from decoding.beliefPropagationGPU import (
    performBeliefPropagationGPU,
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

errorRate = 0.01
iterations = [10, 20, 30, 40, 50, 60, 70, 80, 90]
code_labels = ['72', '90', '108', '144', '288']
trials = 100000

# Batch size for GPU processing - adjust based on your GPU memory
# Larger = faster but more memory
BATCH_SIZE = 1000

np.random.seed(0)
rng = np.random.default_rng(0)

results = {}

print(f"GPU Available: {GPU_AVAILABLE}")
print(f"Running {trials} trials with batch size {BATCH_SIZE}")
print("=" * 60)

total_start = time.time()

for code_name in codes:
    print(f"\nProcessing code: {code_name}")
    oc = np.load(f"codes/{code_name}.npz")
    H = oc['Hx']
    Lx = oc['Lx']
    n = len(H[0])
    
    results[code_name] = {}
    logicalErrors = []
    degeneracies = []
    OSD_invocations = []
    llrs_per_iter = []
    llrs_per_iter_after_OSD = []
    
    for max_iter in iterations:
        iter_start = time.time()
        initialBeliefs = np.array([np.log((1 - errorRate) / errorRate)] * n)
        
        logicalError = 0
        degenerateErrors = 0
        OSD_invoked = 0
        iter_llrs = []
        iter_llrs_after_OSD = []
        
        # Process in batches
        num_batches = (trials + BATCH_SIZE - 1) // BATCH_SIZE
        
        for batch_idx in tqdm.tqdm(range(num_batches), desc=f"iter={max_iter}"):
            batch_start = batch_idx * BATCH_SIZE
            batch_end = min((batch_idx + 1) * BATCH_SIZE, trials)
            current_batch_size = batch_end - batch_start
            
            # Generate batch of errors and syndromes
            errors, syndromes = generate_errors_and_syndromes_batch(
                H, errorRate, current_batch_size, rng
            )
            
            # Run batched BP
            detections, converged, llrs_batch = performBeliefPropagationBatch(
                H, syndromes, initialBeliefs, maxIter=max_iter
            )
            
            # Process results (this part still sequential, but fast)
            for i in range(current_batch_size):
                error = errors[i]
                syndrome = syndromes[i]
                detection = detections[i]
                llrs = llrs_batch[i]
                isSyndromeFound = converged[i]
                
                iter_llrs.extend(llrs.tolist())
                
                if not isSyndromeFound:
                    iter_llrs_after_OSD.extend(llrs.tolist())
                    detection = performOSD(H, syndrome, llrs, detection)
                    OSD_invoked += 1
                
                residual = (detection + error) % 2
                syndromeLogic = (Lx @ residual) % 2
                
                osdCheck = (detection @ H.T) % 2
                isValidOSD = np.array_equal(osdCheck, syndrome)
                
                if isValidOSD and not np.any(syndromeLogic) and not np.array_equal(detection, error):
                    degenerateErrors += 1
                
                if np.any(syndromeLogic):
                    logicalError += 1
        
        iter_time = time.time() - iter_start
        print(f"  max_iter={max_iter}: LER={logicalError/trials:.4f}, "
              f"OSD={OSD_invoked/trials:.2%}, time={iter_time:.1f}s")
        
        logicalErrors.append(logicalError / trials)
        degeneracies.append(degenerateErrors / trials)
        OSD_invocations.append(OSD_invoked / trials)
        llrs_per_iter.append(iter_llrs)
        llrs_per_iter_after_OSD.append(iter_llrs_after_OSD)
    
    results[code_name]['logicalErrors'] = logicalErrors
    results[code_name]['degeneracies'] = degeneracies
    results[code_name]['OSD_invocations'] = OSD_invocations
    results[code_name]['iterations'] = iterations
    results[code_name]['llrs_per_iter'] = llrs_per_iter
    results[code_name]['llrs_per_iter_after_OSD'] = llrs_per_iter_after_OSD

total_time = time.time() - total_start
print(f"\n{'=' * 60}")
print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")

np.savez('data/BP_per_Iteration_GPU.npz', results=results, allow_pickle=True)

# Plotting (same as original)
fig, axes = plt.subplots(5, len(results), figsize=(6*len(results), 20))
properties = ['logicalErrors', 'degeneracies', 'OSD_invocations']
property_labels = ['Logical Error Rate', 'Degeneracies', 'OSD Invocations']

for row, (prop, label) in enumerate(zip(properties, property_labels)):
    for col, (name, data) in enumerate(results.items()):
        ax = axes[row, col]
        ax.plot(data['iterations'], data[prop], marker='o')
        ax.set_title(f'Code {name}' if row == 0 else '')
        ax.set_xlabel('Max BP Iterations')
        ax.set_ylabel(label)
        ax.set_yscale('log')
        ax.grid(True)

for col, (name, data) in enumerate(results.items()):
    ax = axes[3, col]
    llrs_data = data['llrs_per_iter']
    
    valid_data = []
    valid_positions = []
    for i, llrs in enumerate(llrs_data):
        if len(llrs) > 0:
            valid_data.append(llrs)
            valid_positions.append(data['iterations'][i])
    
    if valid_data:
        parts = ax.violinplot(valid_data, positions=valid_positions, widths=8, showmeans=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_alpha(0.7)
    
    ax.set_xlabel('Max BP Iterations')
    ax.set_ylabel('LLR Distribution (BP output)')
    ax.grid(True)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    ax = axes[4, col]
    llrs_data_after_OSD = data['llrs_per_iter_after_OSD']
    
    valid_data_after_OSD = []
    valid_positions_after_OSD = []
    for i, llrs in enumerate(llrs_data_after_OSD):
        if len(llrs) > 0:
            valid_data_after_OSD.append(llrs)
            valid_positions_after_OSD.append(data['iterations'][i])
    
    if valid_data_after_OSD:
        parts = ax.violinplot(valid_data_after_OSD, positions=valid_positions_after_OSD, widths=8, showmeans=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_alpha(0.7)
    
    ax.set_xlabel('Max BP Iterations')
    ax.set_ylabel('LLR Distribution (post-OSD)')
    ax.grid(True)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('media/BP_per_Iteration_GPU.png')
print("Plot saved to media/BP_per_Iteration_GPU.png")
