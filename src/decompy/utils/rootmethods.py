import numpy as np

def roots_cubic(a2, a1, a0):
    Q = (3 * a1 - a2 ** 2) / 9
    R = (9 * a2 * a1 - 27 * a0 - 2 * a2 ** 3) / 54
    D = Q ** 3 + R ** 2
    S = (R + np.sqrt(D, dtype='complex')) ** (1/3)
    T = (R - np.sqrt(D, dtype='complex')) ** (1/3)

    root1 = -a2 / 3 + (S + T)
    root2 = -a2 / 3 - 0.5 * (S + T) + 1j * 0.5 * np.sqrt(3, dtype = 'complex') * (S - T)
    root3 = -a2 / 3 - 0.5 * (S + T) - 1j * 0.5 * np.sqrt(3, dtype = 'complex') * (S - T)

    return np.array([root1, root2, root3], dtype = 'complex')

def roots_quartic(a3, a2, a1, a0):
    z3 = roots_cubic(-a2, a1 * a3 - 4 * a0, 4 * a2 * a0 - a1**2 - a3**2 * a0)
    y = np.real(z3[np.isreal(z3)])[0]

    R = np.sqrt(0.25 * a3**2 - a2 + y, dtype = 'complex')
    D = np.sqrt(0.25 * 3 * a3**2 - R**2 - 2 * a2 + 0.25 * (4 * a3 * a2 - 8 * a1 - a3**3) / R, dtype = 'complex')
    E = np.sqrt(0.25 * 3 * a3**2 - R**2 - 2 * a2 - 0.25 * (4 * a3 * a2 - 8 * a1 - a3**3) / R, dtype = 'complex')

    root1 = -a3 / 4 + R / 2 + D / 2
    root2 = -a3 / 4 + R / 2 - D / 2
    root3 = -a3 / 4 - R / 2 + E / 2
    root4 = -a3 / 4 - R / 2 - E / 2

    return np.array([root1, root2, root3, root4], dtype = 'complex')

