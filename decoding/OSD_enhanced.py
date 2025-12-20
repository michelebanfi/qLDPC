import numpy as np
from itertools import combinations


def performOSD_enhanced(H, syndrome, llr, hard, order=0, max_combinations=None):
    """
    Enhanced Ordered Statistics Decoding with support for higher orders.
    
    Parameters:
    -----------
    H : np.ndarray
        Parity check matrix (m x n)
    syndrome : np.ndarray
        Target syndrome vector
    llr : np.ndarray
        Log-likelihood ratios for each bit
    hard : np.ndarray
        Hard decision (initial guess)
    order : int
        OSD order (0, 1, 2, ...). Higher orders test more bit flip combinations.
    max_combinations : int, optional
        Maximum number of combinations to test. If None, tests all combinations.
        Useful for higher orders to limit computational cost.
    
    Returns:
    --------
    np.ndarray
        Corrected error vector
    """
    
    m, n = H.shape
    
    # Calculate current syndrome
    current_syndrome = (hard @ H.T) % 2
    residual_syndrome = (syndrome + current_syndrome) % 2
    
    # Sort by reliability (absolute LLR)
    llr_abs = np.abs(llr)
    ordering = np.argsort(llr_abs)
    H_permuted = H[:, ordering]
    
    # Perform Gaussian elimination
    _, s_reduced, pivots = gf2_elimination(H_permuted, residual_syndrome)
    
    # Get OSD-0 solution
    e_permuted = np.zeros(n, dtype=int)
    pivot_row, pivot_cols = pivots
    
    for r, c in zip(pivot_row, pivot_cols):
        e_permuted[c] = s_reduced[r]
    
    # Unpermute to get OSD-0 solution
    e_correction = np.zeros(n, dtype=int)
    e_correction[ordering] = e_permuted
    osd0_solution = (hard + e_correction) % 2
    
    # Check if OSD-0 gives valid codeword
    osd0_syndrome = (osd0_solution @ H.T) % 2
    if np.all(osd0_syndrome == syndrome):
        return osd0_solution
    
    # If order is 0, return OSD-0 solution
    if order == 0:
        return osd0_solution
    
    # For higher orders, test bit flip combinations
    # Identify least reliable positions (those not in pivot columns)
    pivot_set = set(pivot_cols)
    non_pivot_positions = [i for i in range(n) if i not in pivot_set]
    
    if len(non_pivot_positions) == 0:
        return osd0_solution
    
    # Sort non-pivot positions by reliability (least reliable first)
    non_pivot_llr = llr_abs[ordering[non_pivot_positions]]
    non_pivot_sorted_idx = np.argsort(non_pivot_llr)
    non_pivot_positions = [non_pivot_positions[i] for i in non_pivot_sorted_idx]
    
    # Limit the number of positions to consider based on order
    num_test_positions = min(len(non_pivot_positions), order + 10)  # Use some buffer
    test_positions = non_pivot_positions[:num_test_positions]
    
    best_solution = osd0_solution.copy()
    best_metric = compute_metric(osd0_solution, llr, H, syndrome)
    found_valid = np.all(osd0_syndrome == syndrome)
    
    # Test combinations of bit flips up to order w
    combinations_tested = 0
    for w in range(1, min(order + 1, len(test_positions) + 1)):
        
        if max_combinations and combinations_tested >= max_combinations:
            break
        
        for flip_positions in combinations(test_positions, w):
            
            if max_combinations and combinations_tested >= max_combinations:
                break
            
            # Create test pattern by flipping bits in e_permuted
            e_test = e_permuted.copy()
            for pos in flip_positions:
                e_test[pos] ^= 1
            
            # Recompute with flipped bits
            # Need to propagate changes through the parity check equations
            e_test_full = recompute_solution(H_permuted, s_reduced, e_test, pivots)
            
            # Unpermute
            e_test_correction = np.zeros(n, dtype=int)
            e_test_correction[ordering] = e_test_full
            test_solution = (hard + e_test_correction) % 2
            
            # Check syndrome
            test_syndrome = (test_solution @ H.T) % 2
            is_valid = np.all(test_syndrome == syndrome)
            
            if is_valid and not found_valid:
                # First valid solution found
                best_solution = test_solution.copy()
                best_metric = compute_metric(test_solution, llr, H, syndrome)
                found_valid = True
            elif is_valid or not found_valid:
                # Compare metrics
                test_metric = compute_metric(test_solution, llr, H, syndrome)
                if test_metric < best_metric:
                    best_solution = test_solution.copy()
                    best_metric = test_metric
            
            combinations_tested += 1
    
    return best_solution


def recompute_solution(H_permuted, s_reduced, e_permuted, pivots):
    """
    Recompute the full solution after flipping bits in non-pivot positions.
    Updates pivot positions to maintain consistency with parity checks.
    """
    m, n = H_permuted.shape
    e_full = e_permuted.copy()
    pivot_rows, pivot_cols = pivots
    
    # For each pivot, recompute its value based on the syndrome and non-pivot bits
    for r, c in zip(pivot_rows, pivot_cols):
        # The pivot bit must satisfy: e @ H^T = s
        # For row r: sum of (e_i * H[r,i]) mod 2 = s[r]
        row_contribution = 0
        for col in range(n):
            if col != c and H_permuted[r, col] == 1:
                row_contribution ^= e_full[col]
        
        # Set pivot bit to satisfy the equation
        e_full[c] = s_reduced[r] ^ row_contribution
    
    return e_full


def compute_metric(solution, llr, H, target_syndrome):
    """
    Compute a metric for solution quality.
    Lower is better. Prioritizes valid codewords, then LLR-based cost.
    """
    syndrome = (solution @ H.T) % 2
    syndrome_weight = np.sum(syndrome != target_syndrome)
    
    # Penalize invalid syndromes heavily
    if syndrome_weight > 0:
        metric = 1e10 + syndrome_weight * 1e8
    else:
        metric = 0.0
    
    # Add LLR cost (prefer solutions consistent with channel observations)
    # Higher LLR magnitude means more confident, so flipping costs more
    llr_cost = np.sum(solution * np.abs(llr))
    metric += llr_cost
    
    return metric


def gf2_elimination(H, s):
    """
    Gaussian elimination over GF(2) to solve H @ e = s.
    Returns the reduced matrix, syndrome, and pivot information.
    """
    m, n = H.shape
    A = H.copy()
    b = s.copy()
    
    pivot_rows = []
    pivot_cols = []
    
    row = 0
    for col in range(n):
        
        if row >= m:
            break
        
        # Find pivot
        pivot_row = -1
        for r in range(row, m):
            if A[r, col] == 1:
                pivot_row = r
                break
        
        if pivot_row == -1:
            continue
        
        # Swap pivot row to current position
        if pivot_row != row:
            A[[row, pivot_row]] = A[[pivot_row, row]]
            b[[row, pivot_row]] = b[[pivot_row, row]]
        
        pivot_rows.append(row)
        pivot_cols.append(col)
        
        # Eliminate all other rows
        for r in range(m):
            if r != row and A[r, col] == 1:
                A[r, :] ^= A[row, :]
                b[r] ^= b[row]
        
        row += 1
    
    return A, b, (pivot_rows, pivot_cols)
