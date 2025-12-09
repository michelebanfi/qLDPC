import numpy as np
from scipy.sparse import csr_matrix

from drawUtils import plotGraph

def performBeliefPropagation(H, syndrome, initialBelief, verbose=True, plotPath=None, maxIter=50):
    """
    Optimized min-sum belief propagation decoder.
    
    Uses sparse matrix operations and vectorized NumPy for significant speedup.
    """
    # Convert to sparse matrix for efficient operations
    H = np.asarray(H, dtype=np.float64)
    syndrome = np.asarray(syndrome, dtype=np.int8)
    initialBelief = np.asarray(initialBelief, dtype=np.float64)
    
    num_checks, num_vars = H.shape
    H_sparse = csr_matrix(H)
    H_T_sparse = H_sparse.T.tocsr()
    
    # Precompute adjacency lists as arrays for faster access
    check_to_var = [H_sparse.indices[H_sparse.indptr[c]:H_sparse.indptr[c+1]] for c in range(num_checks)]
    var_to_check = [H_T_sparse.indices[H_T_sparse.indptr[v]:H_T_sparse.indptr[v+1]] for v in range(num_vars)]
    
    # Precompute syndrome signs: +1 for syndrome=0, -1 for syndrome=1
    syndrome_sign = 1 - 2 * syndrome.astype(np.float64)
    
    if verbose:
        print(f"Initial syndrome: {syndrome}")
    
    if plotPath is not None:
        plotGraph(H, path=plotPath)
    
    # Initialize messages using sparse structure
    # R[c,v] and Q[c,v] only exist where H[c,v] = 1
    # Store as dense arrays but only update non-zero positions
    R = np.zeros((num_checks, num_vars), dtype=np.float64)
    Q = np.zeros((num_checks, num_vars), dtype=np.float64)
    
    # Initialize Q with initial beliefs where H is non-zero
    for v in range(num_vars):
        for c in var_to_check[v]:
            Q[c, v] = initialBelief[v]
    
    # Precompute tanh lookup for speed (used in horizontal step)
    CLIP_VAL = 0.9999999
    
    candidateError = np.zeros(num_vars, dtype=np.int8)
    
    for currentIter in range(maxIter):
        # Horizontal step (check-to-variable messages)
        for c in range(num_checks):
            vars_c = check_to_var[c]
            if len(vars_c) == 0:
                continue
            
            # Get tanh(Q/2) for all connected variables
            tanh_Q = np.tanh(Q[c, vars_c] * 0.5)
            
            # Product of all tanh values
            prod_all = np.prod(tanh_Q)
            
            # For each variable, divide out its contribution
            # R[c,v] = 2 * arctanh(prod_{v' != v} tanh(Q[c,v']/2) * sign)
            with np.errstate(divide='ignore', invalid='ignore'):
                # Avoid division by zero
                tanh_Q_safe = np.where(np.abs(tanh_Q) < 1e-15, 1e-15, tanh_Q)
                prod_others = prod_all / tanh_Q_safe
            
            prod_others_clipped = np.clip(prod_others * syndrome_sign[c], -CLIP_VAL, CLIP_VAL)
            R[c, vars_c] = 2.0 * np.arctanh(prod_others_clipped)
        
        # Vertical step (variable-to-check messages) + compute beliefs
        values = np.zeros(num_vars, dtype=np.float64)
        
        for v in range(num_vars):
            checks_v = var_to_check[v]
            R_sum = np.sum(R[checks_v, v])
            values[v] = R_sum + initialBelief[v]
            
            # Update Q messages
            for c in checks_v:
                Q[c, v] = values[v] - R[c, v]
        
        # Hard decision
        candidateError = (values < 0).astype(np.int8)
        
        # Check syndrome
        calculateSyndrome = H_sparse.dot(candidateError) % 2
        
        if np.array_equal(calculateSyndrome, syndrome):
            if verbose:
                print(f"Error found at iteration {currentIter}: {candidateError}")
            return candidateError, True, values
    
    return candidateError, False, values


def performBeliefPropagationFast(H, syndrome, initialBelief, verbose=True, maxIter=50):
    """
    Even faster BP using fully vectorized operations with sparse matrices.
    Best for larger codes.
    """
    H = np.asarray(H, dtype=np.float64)
    syndrome = np.asarray(syndrome, dtype=np.int8)
    initialBelief = np.asarray(initialBelief, dtype=np.float64)
    
    num_checks, num_vars = H.shape
    H_sparse = csr_matrix(H)
    
    # Mask for non-zero entries
    mask = H != 0
    
    # Syndrome sign matrix
    syndrome_sign = (1 - 2 * syndrome).reshape(-1, 1)
    
    # Initialize messages
    Q = np.where(mask, initialBelief, 0)
    R = np.zeros_like(Q)
    
    CLIP_VAL = 0.9999999
    
    for currentIter in range(maxIter):
        # Horizontal step - vectorized
        tanh_Q = np.tanh(Q * 0.5)
        tanh_Q = np.where(mask, tanh_Q, 1.0)  # Set non-edges to 1 (neutral for product)
        
        # Product along each row
        row_prod = np.prod(tanh_Q, axis=1, keepdims=True)
        
        # Divide out each variable's contribution
        with np.errstate(divide='ignore', invalid='ignore'):
            tanh_Q_safe = np.where(np.abs(tanh_Q) < 1e-15, 1e-15, tanh_Q)
            prod_others = row_prod / tanh_Q_safe
        
        prod_clipped = np.clip(prod_others * syndrome_sign, -CLIP_VAL, CLIP_VAL)
        R = np.where(mask, 2.0 * np.arctanh(prod_clipped), 0)
        
        # Vertical step - vectorized
        R_sum = np.sum(R, axis=0)
        values = R_sum + initialBelief
        
        # Update Q
        Q = np.where(mask, values - R, 0)
        
        # Hard decision and syndrome check
        candidateError = (values < 0).astype(np.int8)
        calculateSyndrome = H_sparse.dot(candidateError) % 2
        
        if np.array_equal(calculateSyndrome, syndrome):
            if verbose:
                print(f"Error found at iteration {currentIter}: {candidateError}")
            return candidateError, True, values
    
    return candidateError, False, values