import numpy as np

def performOSD(H, syndrome, llr, hard):
    
    m, n = H.shape
    
    current_syndrome = (hard @ H.T) % 2
    residual_syndrome = (syndrome + current_syndrome) % 2
    
    llr = np.abs(llr)
    ordering = np.argsort(llr)
    H_permuted = H[:, ordering]
    
    H_reduced, s_reduced, pivots = gf2_elimination(H_permuted, residual_syndrome)
    
    e_permuted = np.zeros(n, dtype=int)
    
    pivot_row, pivot_cols = pivots
    
    for r, c in zip(pivot_row, pivot_cols):
        e_permuted[c] = s_reduced[r]
    
    e_correction = np.zeros(n, dtype=int)
    e_correction[ordering] = e_permuted
    
    solution = (hard + e_correction) % 2
    
    return solution
    

def gf2_elimination(H, s):
    
    m, n = H.shape
    A = H.copy()
    b = s.copy()
    
    pivot_rows = []
    pivot_cols = []
    
    row = 0
    for col in range(n):
        
        if row >= m:
            break
        
        pivot_row = -1
        for r in range(row, m):
            if A[r, col] == 1:
                pivot_row = r
                break
            
        if pivot_row == -1:
            continue
        
        # swap pivot row up
        if pivot_row != row:
            A[[row, pivot_row]] = A[[pivot_row, row]]
            b[[row, pivot_row]] = b[[pivot_row, row]]
        
        pivot_rows.append(row)
        pivot_cols.append(col)
        
        # eliminate below
        for r in range(m):
            if r != row and A[r, col] == 1:
                
                A[r, :] ^= A[row, :]
                b[r] ^= b[row]
                
        row += 1

    return A, b, (pivot_rows, pivot_cols)