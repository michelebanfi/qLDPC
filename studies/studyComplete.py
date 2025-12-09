import numpy as np
from qldpc import codes, circuits
from qldpc.objects import Pauli
from ldpc.ckt_noise.dem_matrices import detector_error_model_to_check_matrices
from sympy.abc import x, y
import matplotlib.pyplot as plt
import tqdm

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from decoding.beliefPropagation import performBeliefPropagationFast

codes = [
    {"code": codes.BBCode(
        {x: 6, y: 6},
        x**3 + y + y**2,
        y**3 + x + x**2,
        ),
     "name": '[[72, 12, 6]]',
     "distance": 6
    },
    {"code": codes.BBCode(
        {x: 15, y: 3},
        x**9 + y + y**2,
        1 + x**2 + x**7,
    ),
     "name": '[[90, 8, 10]]',
     "distance": 10
    },
    {"code": codes.BBCode(
        {x: 9, y: 6},
        x**3 + y + y**2,
        y**3 + x + x**2,
    ),
     "name": '[[108, 8, 10]]',
        "distance": 10
    },
    {"code": codes.BBCode(
        {x: 12, y: 6},
        x**3 + y + y**2,
        y**3 + x + x**2,
    ),
     "name": '[[144, 12, 12]]',
     "distance": 12
    },
    {"code": codes.BBCode(
        {x: 12, y: 12},
        x**3 + y**2 + y**7,
        y**3 + x + x**2,
    ),
     "name": '[[288, 12, 18]]',
     "distance": 18
    },
]

trials = 2
results = {}

physicalErrorRates = np.logspace(-3.2, -1.3, 8)

for code in tqdm.tqdm(codes):
    logicalErrorRates = []
    
    for p in physicalErrorRates:
        num_rounds = code['distance']
        results[code["name"]] = {}
        
        oc = code["code"]

        noise_model = circuits.DepolarizingNoiseModel(p)
        circuit = circuits.get_memory_experiment(
            code = oc,
            basis = Pauli.Z,
            num_rounds = num_rounds,
            noise_model = noise_model,
        )

        dem = circuit.detector_error_model(decompose_errors=False) # can handle hyperedges
        matrices = detector_error_model_to_check_matrices(dem, allow_undecomposed_hyperedges=True) # transform the DEM to a matrix

        H_spacetime = np.array(matrices.check_matrix.todense())
        L_matrix = np.array(matrices.observables_matrix.todense())
        priors = matrices.priors

        clipped_priors = np.clip(priors, 1e-15, 1 - 1e-15)
        initial_beliefs = np.log((1 - clipped_priors) / clipped_priors)

        sampler = circuit.compile_detector_sampler()

        logical_errors = 0
        detection_events, actual_observables = sampler.sample(shots=trials, separate_observables=True)

        for i in range(trials):
            syndrome = detection_events[i]
            
            prediction, converged, _ = performBeliefPropagationFast(
                H_spacetime, 
                syndrome, 
                initial_beliefs, 
                verbose=False
            )
            
            predicted_logical_flip = (L_matrix @ prediction) % 2
            
            if not np.array_equal(predicted_logical_flip, actual_observables[i]):
                logical_errors += 1

        logicalErrorRates.append(logical_errors / trials)
        
    results[code["name"]]["ler"] = logicalErrorRates
    
np.savez("data/COMPLETE-LERS.npz", physicalErrorRates=physicalErrorRates, results=results)

plt.figure(figsize=(10, 6))
for name in results:
    plt.plot(physicalErrorRates, results[name]['ler'], label=name, marker='o')
    plt.grid(True)
    plt.xscale('log')
    plt.xlabel('Physical error rate')
    plt.ylabel('Logical error rate')
    plt.legend()
plt.savefig("media/COMPLETE-LERS")