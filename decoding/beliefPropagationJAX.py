"""
JAX-accelerated Belief Propagation with JIT compilation.
Works on Apple Silicon (M1/M2/M3) with Metal, NVIDIA GPUs with CUDA, and CPU.

Install:
    pip install jax jaxlib              # CPU only
    pip install jax "jax-metal"         # Apple Silicon (macOS)
    pip install jax[cuda12]             # NVIDIA GPU
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
import numpy as np

# Check available backends
print(f"JAX devices: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")


@partial(jit, static_argnames=['maxIter'])
def bp_single_jax(H, syndrome, initialBelief, maxIter=50):
    """
    JIT-compiled BP for a single syndrome.
    """
    num_checks, num_vars = H.shape
    
    mask = H != 0
    syndrome_sign = (1 - 2 * syndrome.astype(jnp.float32)).reshape(-1, 1)
    
    Q = jnp.where(mask, initialBelief, 0.0)
    
    CLIP_VAL = 0.9999999
    
    def bp_step(carry, _):
        Q, converged, candidate, values = carry
        
        # Horizontal step
        tanh_Q = jnp.tanh(Q * 0.5)
        tanh_Q = jnp.where(mask, tanh_Q, 1.0)
        
        row_prod = jnp.prod(tanh_Q, axis=1, keepdims=True)
        tanh_Q_safe = jnp.where(jnp.abs(tanh_Q) < 1e-15, 1e-15, tanh_Q)
        prod_others = row_prod / tanh_Q_safe
        
        prod_clipped = jnp.clip(prod_others * syndrome_sign, -CLIP_VAL, CLIP_VAL)
        R = jnp.where(mask, 2.0 * jnp.arctanh(prod_clipped), 0.0)
        
        # Vertical step
        R_sum = jnp.sum(R, axis=0)
        new_values = R_sum + initialBelief
        
        new_Q = jnp.where(mask, new_values - R, 0.0)
        
        # Hard decision
        new_candidate = (new_values < 0).astype(jnp.int32)
        
        # Check syndrome
        calc_syndrome = (H.T @ new_candidate) % 2
        new_converged = jnp.array_equal(calc_syndrome, syndrome)
        
        # Keep old values if already converged
        final_Q = jnp.where(converged, Q, new_Q)
        final_candidate = jnp.where(converged, candidate, new_candidate)
        final_values = jnp.where(converged, values, new_values)
        final_converged = converged | new_converged
        
        return (final_Q, final_converged, final_candidate, final_values), None
    
    init_candidate = jnp.zeros(num_vars, dtype=jnp.int32)
    init_values = jnp.zeros(num_vars, dtype=jnp.float32)
    init_carry = (Q, False, init_candidate, init_values)
    
    (_, converged, candidate, values), _ = jax.lax.scan(
        bp_step, init_carry, None, length=maxIter
    )
    
    return candidate, converged, values


@partial(jit, static_argnames=['maxIter'])
def bp_batch_jax(H, syndromes, initialBelief, maxIter=50):
    """
    Batched JIT-compiled BP using vmap.
    Process multiple syndromes in parallel.
    
    Args:
        H: (num_checks, num_vars)
        syndromes: (batch_size, num_checks)
        initialBelief: (num_vars,)
        maxIter: int
    
    Returns:
        candidates: (batch_size, num_vars)
        converged: (batch_size,)
        values: (batch_size, num_vars)
    """
    # vmap over the syndrome axis
    batched_bp = vmap(
        lambda s: bp_single_jax(H, s, initialBelief, maxIter),
        in_axes=0
    )
    return batched_bp(syndromes)


def performBeliefPropagationJAX(H, syndrome, initialBelief, verbose=False, maxIter=50):
    """
    Drop-in replacement for performBeliefPropagationFast using JAX.
    """
    H_jax = jnp.asarray(H, dtype=jnp.float32)
    syndrome_jax = jnp.asarray(syndrome, dtype=jnp.int32)
    initialBelief_jax = jnp.asarray(initialBelief, dtype=jnp.float32)
    
    candidate, converged, values = bp_single_jax(
        H_jax, syndrome_jax, initialBelief_jax, maxIter
    )
    
    return np.asarray(candidate), bool(converged), np.asarray(values)


def performBeliefPropagationBatchJAX(H, syndromes, initialBelief, maxIter=50):
    """
    Batched BP using JAX vmap - process many syndromes in parallel.
    
    Args:
        H: numpy array (num_checks, num_vars)
        syndromes: numpy array (batch_size, num_checks)
        initialBelief: numpy array (num_vars,)
        maxIter: int
    
    Returns:
        candidates: numpy array (batch_size, num_vars)
        converged: numpy array (batch_size,) bool
        values: numpy array (batch_size, num_vars)
    """
    H_jax = jnp.asarray(H, dtype=jnp.float32)
    syndromes_jax = jnp.asarray(syndromes, dtype=jnp.int32)
    initialBelief_jax = jnp.asarray(initialBelief, dtype=jnp.float32)
    
    candidates, converged, values = bp_batch_jax(
        H_jax, syndromes_jax, initialBelief_jax, maxIter
    )
    
    return np.asarray(candidates), np.asarray(converged), np.asarray(values)


def generate_errors_and_syndromes(H, error_rate, batch_size, rng=None):
    """Generate batch of random errors and syndromes."""
    if rng is None:
        rng = np.random.default_rng()
    
    num_checks, num_vars = H.shape
    errors = (rng.random((batch_size, num_vars)) < error_rate).astype(np.int8)
    syndromes = (errors @ H.T) % 2
    
    return errors, syndromes.astype(np.int32)


# Warm-up function to JIT compile before timing
def warmup_jit(H, maxIter=50):
    """Call this once before timing to compile the JIT functions."""
    num_checks, num_vars = H.shape
    dummy_syndrome = np.zeros(num_checks, dtype=np.int32)
    dummy_belief = np.ones(num_vars, dtype=np.float32)
    
    # Single
    _ = performBeliefPropagationJAX(H, dummy_syndrome, dummy_belief, maxIter=maxIter)
    
    # Batch
    dummy_syndromes = np.zeros((10, num_checks), dtype=np.int32)
    _ = performBeliefPropagationBatchJAX(H, dummy_syndromes, dummy_belief, maxIter=maxIter)
    
    print("JIT compilation complete")


if __name__ == "__main__":
    # Quick test
    print("\nRunning quick test...")
    
    # Create a simple test matrix
    H = np.array([
        [1, 1, 0, 1, 0, 0],
        [0, 1, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 1],
    ], dtype=np.float32)
    
    error_rate = 0.1
    belief = np.log((1 - error_rate) / error_rate) * np.ones(6, dtype=np.float32)
    
    # Warm up JIT
    warmup_jit(H, maxIter=10)
    
    # Test single
    error = np.array([1, 0, 0, 0, 0, 0], dtype=np.int32)
    syndrome = (error @ H.T) % 2
    
    candidate, converged, values = performBeliefPropagationJAX(H, syndrome, belief, maxIter=10)
    print(f"Single: converged={converged}, candidate={candidate}")
    
    # Test batch
    errors, syndromes = generate_errors_and_syndromes(H, 0.1, 100)
    candidates, converged_batch, values_batch = performBeliefPropagationBatchJAX(
        H, syndromes, belief, maxIter=10
    )
    print(f"Batch: {converged_batch.sum()}/{len(converged_batch)} converged")
