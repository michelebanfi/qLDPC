import numpy as np
from qldpc import codes
from sympy.abc import x, y

codes = [
    {"code": codes.BBCode(
        {x: 6, y: 6},
        x**3 + y + y**2,
        y**3 + x + x**2,
        ),
     "name": '[[72, 12, 6]]'
    },
    {"code": codes.BBCode(
        {x: 15, y: 3},
        x**9 + y + y**2,
        1 + x**2 + x**7,
    ),
     "name": '[[90, 8, 10]]'
    },
    {"code": codes.BBCode(
        {x: 9, y: 6},
        x**3 + y + y**2,
        y**3 + x + x**2,
    ),
     "name": '[[108, 8, 10]]'
    },
    {"code": codes.BBCode(
        {x: 12, y: 6},
        x**3 + y + y**2,
        y**3 + x + x**2,
    ),
     "name": '[[144, 12, 12]]'
    },
    {"code": codes.BBCode(
        {x: 12, y: 12},
        x**3 + y**2 + y**7,
        y**3 + x + x**2,
    ),
     "name": '[[288, 12, 18]]'
    },
]

for code in codes:
    Hx = code["code"].matrix_x
    Hz = code["code"].matrix_z
    np.savez(f'codes/{code["name"]}.npz', Hx=np.array(Hx), Hz=np.array(Hz))

H = [
    [1, 0, 1, 0, 1, 0, 1],
    [0, 1, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 1],
]
np.array(H)
np.savez(f'codes/steane.npz', Hx=np.array(H), Hz=np.array(H))
