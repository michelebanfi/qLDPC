import numpy as np
from qldpc import codes, circuits
from qldpc.objects import Pauli
from sympy.abc import x, y
import matplotlib.pyplot as plt

code = codes.BBCode(
        {x: 6, y: 6},
        x**3 + y + y**2,
        y**3 + x + x**2,
        )

num_rounds = 3
noise_model = circuits.DepolarizingNoiseModel(0.1)
noisy_circuit = circuits.get_memory_experiment(
    code = code,
    basis = Pauli.Z,
    num_rounds = num_rounds,
    noise_model = noise_model,
)

svg = noisy_circuit.diagram("timeline-svg")
with open('media/timeline.svg', 'w') as f:
    print(svg, file=f)
