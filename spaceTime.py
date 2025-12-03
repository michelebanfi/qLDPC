import numpy as np


def spaceTimeMatrix(H, n_cycles):
    
    m, n = H.shape
    
    # spatial component, we simply repeat the code by n_cycle times
    H_spatial = np.kron(np.eye(n_cycles), H)
    
    # temporal component => m rows (checks) * the number of cycles
    I_Current = np.eye(m * n_cycles)
    I_next = np.eye(m * n_cycles, k=-m) # this is the identity but shifted by the number of checks, this should be the t-1
    
    H_temporal = (I_Current + I_next) % 2 # measurement at t affect t and t+1
    
    spacetime = np.hstack([H_spatial, H_temporal]) # concatenate them horizontally
    return spacetime

def spacetimeSyndrome(code, error_rate, n_cycles):
    
    error = (np.random.random(len(code[0])) < error_rate).astype(int)
    
    syndrome = (error @ code.T) % 2
    
    syndrome_history = []
    
    for i in range(n_cycles):
        s_error = (np.random.random(len(code)) < error_rate).astype(int)
        syndrome = (syndrome + s_error) % 2
        
        syndrome_history.append(syndrome)
        
    syndrome_history_diff = []
    syndrome_history_diff.append(syndrome)
    
    for i in range(1, n_cycles):
        diff = (syndrome_history[i] + syndrome_history[i - 1]) % 2
        syndrome_history_diff.append(diff)
    
    flatten = np.concatenate(syndrome_history_diff) # make a single vector
    
    return error, flatten