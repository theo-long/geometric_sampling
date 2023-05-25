import math
import numba

# Note: these functions were all originally generated using sympy.
@numba.njit
def theta_to_phi(theta, x, y):
    return 2 * math.atan(
        (x + y * math.tan((1 / 2) * theta) - 1)
        / (x * math.tan((1 / 2) * theta) - y + math.tan((1 / 2) * theta))
    )

@numba.njit
def sqrt_detJ_func(h, phi, theta):
    return math.sqrt(
        ((h - math.sin(phi)) ** 2 + (h - math.sin(theta)) ** 2)
        * (math.cos(phi - theta) - 1) ** 2
        / (math.sin(phi) - math.sin(theta)) ** 4
    )

@numba.njit
def dphi_dtheta_func(theta, x, y):
    return (
        2
        * (
            y
            * ((1 / 2) * math.tan((1 / 2) * theta) ** 2 + 1 / 2)
            / (x * math.tan((1 / 2) * theta) - y + math.tan((1 / 2) * theta))
            + (x + y * math.tan((1 / 2) * theta) - 1)
            * (
                -x * ((1 / 2) * math.tan((1 / 2) * theta) ** 2 + 1 / 2)
                - 1 / 2 * math.tan((1 / 2) * theta) ** 2
                - 1 / 2
            )
            / (x * math.tan((1 / 2) * theta) - y + math.tan((1 / 2) * theta)) ** 2
        )
        / (
            (x + y * math.tan((1 / 2) * theta) - 1) ** 2
            / (x * math.tan((1 / 2) * theta) - y + math.tan((1 / 2) * theta)) ** 2
            + 1
        )
    )
