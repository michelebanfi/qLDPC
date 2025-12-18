"""
GPU-accelerated Belief Propagation using CuPy.
Falls back to NumPy if CuPy is not available.

Install CuPy: pip install cupy-cuda12x  (adjust for your CUDA version)
Or for CPU fallback with similar API: pip install numpy
"""

try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("CuPy detected - GPU acceleration enabled")
except ImportError:
    import numpy as cp
    GPU_AVAILABLE = False
    print("CuPy not found - falling back to NumPy (CPU)")

import numpy as np
from scipy.sparse import csr_matrix


def performBeliefPropagationGPU(H, syndrome, initialBelief, verbose=False, maxIter=50):
    """
    GPU-accelerated BP for a single syndrome.
    Drop-in replacement for performBeliefPropagationFast.
    """
    H_gpu = cp.asarray(H, dtype=cp.float64)
    syndrome_gpu = cp.asarray(syndrome, dtype=cp.int8)
    initialBelief_gpu = cp.asarray(initialBelief, dtype=cp.float64)
    
    num_checks, num_vars = H_gpu.shape
    
    mask = H_gpu != 0
    syndrome_sign = (1 - 2 * syndrome_gpu.astype(cp.float64)).reshape(-1, 1)
    
    Q = cp.where(mask, initialBelief_gpu, 0)
    R = cp.zeros_like(Q)
    
    CLIP_VAL = 0.9999999
    
    # Pre-compute H sparse for syndrome check (keep on CPU for efficiency)
    H_sparse = csr_matrix(np.asarray(H))
    
    for currentIter in range(maxIter):
        # Horizontal step
        tanh_Q = cp.tanh(Q * 0.5)
        tanh_Q = cp.where(mask, tanh_Q, 1.0)
        
        row_prod = cp.prod(tanh_Q, axis=1, keepdims=True)
        
        tanh_Q_safe = cp.where(cp.abs(tanh_Q) < 1e-15, 1e-15, tanh_Q)
        prod_others = row_prod / tanh_Q_safe
        
        prod_clipped = cp.clip(prod_others * syndrome_sign, -CLIP_VAL, CLIP_VAL)
        R = cp.where(mask, 2.0 * cp.arctanh(prod_clipped), 0)
        
        # Vertical step
        R_sum = cp.sum(R, axis=0)
        values = R_sum + initialBelief_gpu
        
        Q = cp.where(mask, values - R, 0)
        
        # Hard decision - move to CPU for sparse syndrome check
        candidateError = (values < 0).astype(cp.int8)
        candidateError_cpu = cp.asnumpy(candidateError) if GPU_AVAILABLE else candidateError
        
        calculateSyndrome = H_sparse.dot(candidateError_cpu) % 2
        syndrome_cpu = cp.asnumpy(syndrome_gpu) if GPU_AVAILABLE else syndrome_gpu
        
        if np.array_equal(calculateSyndrome, syndrome_cpu):
            values_cpu = cp.asnumpy(values) if GPU_AVAILABLE else values
            if verbose:
                print(f"Error found at iteration {currentIter}")
            return candidateError_cpu, True, values_cpu
    
    candidateError_cpu = cp.asnumpy(candidateError) if GPU_AVAILABLE else candidateError
    values_cpu = cp.asnumpy(values) if GPU_AVAILABLE else values
    return candidateError_cpu, False, values_cpu


def performBeliefPropagationBatch(H, syndromes, initialBelief, maxIter=50):
    """
    Batched GPU BP - process multiple syndromes in parallel.
    This is the KEY optimization: instead of running trials sequentially,
    run many trials simultaneously on the GPU.
    
    Args:
        H: Parity check matrix (num_checks, num_vars)
        syndromes: Batch of syndromes (batch_size, num_checks)
        initialBelief: Initial LLR beliefs (num_vars,)
        maxIter: Maximum BP iterations
    
    Returns:
        candidateErrors: (batch_size, num_vars)
        converged: (batch_size,) boolean array
        values: (batch_size, num_vars) final LLRs
    """
    batch_size = syndromes.shape[0]
    
    H_gpu = cp.asarray(H, dtype=cp.float64)
    syndromes_gpu = cp.asarray(syndromes, dtype=cp.int8)
    initialBelief_gpu = cp.asarray(initialBelief, dtype=cp.float64)
    
    num_checks, num_vars = H_gpu.shape
    
    # Mask: (num_checks, num_vars)
    mask = H_gpu != 0
    
    # Syndrome signs: (batch_size, num_checks, 1)
    syndrome_signs = (1 - 2 * syndromes_gpu.astype(cp.float64))[:, :, cp.newaxis]
    
    # Q: (batch_size, num_checks, num_vars)
    Q = cp.where(mask, initialBelief_gpu, 0)
    Q = cp.broadcast_to(Q, (batch_size, num_checks, num_vars)).copy()
    
    R = cp.zeros((batch_size, num_checks, num_vars), dtype=cp.float64)
    
    CLIP_VAL = 0.9999999
    
    # Track which samples have converged
    converged = cp.zeros(batch_size, dtype=cp.bool_)
    final_candidates = cp.zeros((batch_size, num_vars), dtype=cp.int8)
    final_values = cp.zeros((batch_size, num_vars), dtype=cp.float64)
    
    H_sparse = csr_matrix(np.asarray(H))
    
    for currentIter in range(maxIter):
        # Only process non-converged samples
        active = ~converged
        if not cp.any(active):
            break
        
        # Horizontal step: (batch_size, num_checks, num_vars)
        tanh_Q = cp.tanh(Q * 0.5)
        tanh_Q = cp.where(mask, tanh_Q, 1.0)
        
        # Product along variables for each check: (batch_size, num_checks, 1)
        row_prod = cp.prod(tanh_Q, axis=2, keepdims=True)
        
        tanh_Q_safe = cp.where(cp.abs(tanh_Q) < 1e-15, 1e-15, tanh_Q)
        prod_others = row_prod / tanh_Q_safe
        
        prod_clipped = cp.clip(prod_others * syndrome_signs, -CLIP_VAL, CLIP_VAL)
        R = cp.where(mask, 2.0 * cp.arctanh(prod_clipped), 0)
        
        # Vertical step
        R_sum = cp.sum(R, axis=1)  # (batch_size, num_vars)
        values = R_sum + initialBelief_gpu
        
        # Update Q
        Q = cp.where(mask, values[:, cp.newaxis, :] - R, 0)
        
        # Hard decision
        candidateErrors = (values < 0).astype(cp.int8)
        
        # Check syndromes (move to CPU for sparse operation)
        candidateErrors_cpu = cp.asnumpy(candidateErrors) if GPU_AVAILABLE else candidateErrors
        syndromes_cpu = cp.asnumpy(syndromes_gpu) if GPU_AVAILABLE else syndromes_gpu
        
        for i in range(batch_size):
            if converged[i]:
                continue
            calc_syndrome = H_sparse.dot(candidateErrors_cpu[i]) % 2
            if np.array_equal(calc_syndrome, syndromes_cpu[i]):
                converged[i] = True
                final_candidates[i] = candidateErrors[i]
                final_values[i] = values[i]
    
    # Fill in non-converged results
    not_converged = ~converged
    if cp.any(not_converged):
        not_converged_idx = cp.where(not_converged)[0]
        final_candidates[not_converged_idx] = candidateErrors[not_converged_idx]
        final_values[not_converged_idx] = values[not_converged_idx]
    
    if GPU_AVAILABLE:
        return cp.asnumpy(final_candidates), cp.asnumpy(converged), cp.asnumpy(final_values)
    return final_candidates, converged, final_values


def generate_errors_and_syndromes_batch(H, error_rate, batch_size, rng=None):
    """
    Generate a batch of random errors and their syndromes on GPU.
    
    Returns:
        errors: (batch_size, num_vars)
        syndromes: (batch_size, num_checks)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    num_checks, num_vars = H.shape
    
    # Generate errors on CPU (CuPy random can be slower for this)
    errors = (rng.random((batch_size, num_vars)) < error_rate).astype(np.int8)
    
    # Compute syndromes
    syndromes = (errors @ H.T) % 2
    
    return errors, syndromes.astype(np.int8)
