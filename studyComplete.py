import numpy as np
from qldpc import codes, circuits
from qldpc.objects import Pauli
from ldpc.ckt_noise.dem_matrices import detector_error_model_to_check_matrices
from sympy.abc import x, y
import matplotlib.pyplot as plt

from beliefPropagation import performBeliefPropagation

code = codes.BBCode(
        {x: 6, y: 6},
        x**3 + y + y**2,
        y**3 + x + x**2,
        )

trials = 10
num_rounds = 2

physical_error_rate = 0.05

noise_model = circuits.DepolarizingNoiseModel(0.1)
circuit = circuits.get_memory_experiment(
    code = code,
    basis = Pauli.Z,
    num_rounds = num_rounds,
    noise_model = noise_model,
)

# For BP, we don't need to decompose errors into graphlike components
# BP can handle hyperedges (errors affecting >2 detectors)
dem = circuit.detector_error_model(decompose_errors=False) 

# Convert DEM to Matrices using the helper function
# allow_undecomposed_hyperedges=True lets BP handle errors affecting >2 detectors
matrices = detector_error_model_to_check_matrices(dem, allow_undecomposed_hyperedges=True)

# Convert sparse matrices to dense arrays for BP
H_spacetime = np.array(matrices.check_matrix.todense())
L_matrix = np.array(matrices.observables_matrix.todense())
priors = matrices.priors

# Calculate LLRs for BP initialization
# Avoid log(0) by clipping probabilities slightly
clipped_priors = np.clip(priors, 1e-15, 1 - 1e-15)
initial_beliefs = np.log((1 - clipped_priors) / clipped_priors)

# Compile sampler
sampler = circuit.compile_detector_sampler()

# 2. SIMULATION LOOP
logical_errors = 0
# Sample batch of shots (syndromes and actual logical flips)
detection_events, actual_observables = sampler.sample(shots=trials, separate_observables=True)

for i in range(trials):
    syndrome = detection_events[i]
    
    # Run your BP decoder
    # 'prediction' is the vector of estimated errors (which error mechanisms happened)
    prediction, converged, _ = performBeliefPropagation(
        H_spacetime, 
        syndrome, 
        initial_beliefs, 
        verbose=False
    )
    
    # 3. LOGICAL CHECK
    # Calculate the logical effect of the decoder's predicted error
    # prediction is a binary vector (1 if decoder thinks that error happened)
    predicted_logical_flip = (L_matrix @ prediction) % 2
    
    # Compare with the actual logical flip that occurred in the simulation
    # If they don't match, we failed to correct the error
    if not np.array_equal(predicted_logical_flip, actual_observables[i]):
        logical_errors += 1

print(f"Logical Error Rate: {logical_errors / trials}")