import sympy as sp

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
    I_tilde * omega_init + m * uy * L_prime - (I_tilde * omega_after + m * L_prime * vy)
)

equation_sets = [f1, f2, f3, f4, f5]

# Substitute concrete values
subs_dict = {M: 1000000, m: 1, ux: 1, uy: 0.1, omega_init: 0.1, L: 1, theta: 80, mu: 1}
concrete_eqs = [eq.subs(subs_dict) for eq in equation_sets]

print("Solving the following equations:")
# Solve the system
result = sp.solve(concrete_eqs, [Vx, Vy, vx, vy, omega_after], dict=True)

print(result)
