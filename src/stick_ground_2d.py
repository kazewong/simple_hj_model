import sympy as sp
import numpy as np

# Define variables
M, m, ux, uy, omega_init, L, theta, mu = sp.symbols("M m ux uy omega_init L theta mu")
Vx, Vy, vx, vy, omega_after = sp.symbols("Vx Vy vx vy omega_after")

# Intermediate calculations
I = m * L**2 / 3
phi = sp.atan(1 / mu)
L_prime = L * sp.sin(theta) - L * sp.cos(theta) * sp.tan(phi)
I_hat = I + m * L**2
I_tilde = I + m * L_prime**2

# Equations
f1 = m * ux - (M * Vx + m * vx)
f2 = m * uy - (M * Vy + m * vy)
f3 = (
    (1 / 2) * (m * ux**2 + m * uy**2)
    + (1 / 2) * I * omega_init**2
    - (
        (1 / 2) * M * Vx**2
        + (1 / 2) * M * Vy**2
        + (1 / 2) * m * vx**2
        + (1 / 2) * m * vy**2
        + (1 / 2) * I * omega_after**2
    )
)
f4 = (
    I_hat * omega_init
    + m * uy * L * sp.cos(theta)
    - m * ux * L * sp.sin(theta)
    - (I_hat * omega_after + m * L * sp.cos(theta) * vy - m * L * sp.sin(theta) * vx)
)
f5 = (
    I_tilde * omega_init + m * ux * L_prime - (I_tilde * omega_after + m * L_prime * vx)
)

equation_sets = [f1, f2, f3, f4, f5]


def solve_system(
    ux_value,
    uy_value,
    omega_init_value,
    theta_value,
    mu_value,
    M_value = 10000000,
    m_value = 1,
    L_value = 1,
):
    # Substitute concrete values
    subs_dict = {
        M: M_value,
        m: m_value,
        ux: ux_value,
        uy: uy_value,
        omega_init: omega_init_value,
        L: L_value,
        theta: theta_value,
        mu: mu_value,
    }
    concrete_eqs = [eq.subs(subs_dict) for eq in equation_sets]

    # Solve the system
    # result = sp.nsolve(
    #     concrete_eqs, (Vx, Vy, vx, vy, omega_after), (0, 0, 100.0, 100.0, 100.0), dict=True
    # )
    result = sp.solve(concrete_eqs, (Vx, Vy, vx, vy, omega_after), dict=True)
    print("Result: ", result)

    test_result = [eq.subs(result[-1]).evalf() for eq in concrete_eqs]
    for i, eq in enumerate(test_result):
        if abs(eq) > 1e-5:
            print(f"Equation {i+1} not satisfied: {eq}")

    return {
        "vx": result[0][vx],
        "vy": result[0][vy],
        "omega_after": result[0][omega_after],
    }


# Substitute concrete values
subs_dict = {
    "M_value": 10000000,
    "m_value": 1,
    "ux_value": -1.0,
    "uy_value": 0.0,
    "omega_init_value": 0.1,
    "L_value": 1,
    "theta_value": np.deg2rad(70),
    "mu_value": 1,
}

result = solve_system(**subs_dict)
print("Result: ", result)
