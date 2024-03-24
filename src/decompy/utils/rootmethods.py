import numpy as np


def roots_cubic(a2, a1, a0):
    """The `roots_cubic` function calculates the roots of a cubic equation given the coefficients.

    Parameters
    ----------
    a2
        The coefficient of the cubic term in the equation.
    a1
        The parameter `a1` represents the coefficient of the linear term in the cubic equation.
    a0
        The coefficient of the constant term in the cubic equation.

    Returns
    -------
        The function `roots_cubic` returns an array containing the three roots of a cubic equation.

    """
    Q = (3 * a1 - a2**2) / 9
    R = (9 * a2 * a1 - 27 * a0 - 2 * a2**3) / 54
    D = Q**3 + R**2
    S = (R + np.sqrt(D, dtype="complex")) ** (1 / 3)
    T = (R - np.sqrt(D, dtype="complex")) ** (1 / 3)

    root1 = -a2 / 3 + (S + T)
    root2 = -a2 / 3 - 0.5 * (S + T) + 1j * 0.5 * np.sqrt(3, dtype="complex") * (S - T)
    root3 = -a2 / 3 - 0.5 * (S + T) - 1j * 0.5 * np.sqrt(3, dtype="complex") * (S - T)

    return np.array([root1, root2, root3], dtype="complex")


def roots_quartic(a3, a2, a1, a0):
    """The `roots_quartic` function calculates the roots of a quartic equation given its coefficients.

    Parameters
    ----------
    a3
        The parameter `a3` represents the coefficient of the quartic term in the quartic equation.
    a2
        The parameter `a2` represents the coefficient of the quadratic term in the quartic equation.
    a1
        The parameter `a1` represents the coefficient of the linear term in the quartic equation.
    a0
        The parameter `a0` represents the coefficient of the constant term in the quartic equation.

    Returns
    -------
        The function `roots_quartic` returns an array containing the four roots of the quartic equation.

    """
    z3 = roots_cubic(-a2, a1 * a3 - 4 * a0, 4 * a2 * a0 - a1**2 - a3**2 * a0)
    y = np.real(z3[np.isreal(z3)])[0]

    R = np.sqrt(0.25 * a3**2 - a2 + y, dtype="complex")
    D = np.sqrt(
        0.25 * 3 * a3**2 - R**2 - 2 * a2 + 0.25 * (4 * a3 * a2 - 8 * a1 - a3**3) / R,
        dtype="complex",
    )
    E = np.sqrt(
        0.25 * 3 * a3**2 - R**2 - 2 * a2 - 0.25 * (4 * a3 * a2 - 8 * a1 - a3**3) / R,
        dtype="complex",
    )

    root1 = -a3 / 4 + R / 2 + D / 2
    root2 = -a3 / 4 + R / 2 - D / 2
    root3 = -a3 / 4 - R / 2 + E / 2
    root4 = -a3 / 4 - R / 2 - E / 2

    return np.array([root1, root2, root3, root4], dtype="complex")
