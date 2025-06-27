import sympy as sp
import numpy as np
import multiprocessing as mp
import holoviews as hv
from holoviews import opts
hv.extension("bokeh")

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
    I_tilde * omega_init - m * ux * L_prime - (I_tilde * omega_after - m * L_prime * vx)
)

equation_sets = [f1, f2, f3, f4, f5]


def solve_system(
    ux_value,
    uy_value,
    omega_init_value,
    theta_value,
    mu_value = 1.0,
    M_value=1000000000,
    m_value=1,
    L_value=1,
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
    results = sp.solve(concrete_eqs, (Vx, Vy, vx, vy, omega_after), dict=True)
    current_delta = 0
    best_result = results[0]
    for result in results:
        delta = np.sum(
            [
                abs(result[vx] - ux_value),
                abs(result[vy] - uy_value),
                abs(result[omega_after] - omega_init_value),
            ]
        )
        if delta > current_delta:
            current_delta = delta
            best_result = result


    test_result = [eq.subs(best_result).evalf() for eq in concrete_eqs]
    for i, eq in enumerate(test_result):
        if abs(eq) > 1e-5:
            print(f"Equation {i+1} not satisfied: {eq}")

    return {
        "Vx": best_result[Vx],
        "Vy": best_result[Vy],
        "vx": best_result[vx],
        "vy": best_result[vy],
        "omega_after": best_result[omega_after],
    }

N_samples = 5

ux_range = np.array([0.25])
uy_range = np.linspace(-0.5, 0.5, N_samples)
omega_init_range = np.linspace(-0.5, 0.5, N_samples)
theta_range = np.linspace(45, 90, 15)

results = {}

def worker(args):
    ux_value, uy_value, omega_init_value, theta_value = args
    result = solve_system(
        ux_value=ux_value,
        uy_value=uy_value,
        omega_init_value=omega_init_value,
        theta_value=np.deg2rad(theta_value),
        mu_value=0.5,
    )
    return (result["vx"], result["vy"], result["omega_after"])

param_grid = np.array([
    (ux_value, uy_value, omega_init_value, theta_value)
    for ux_value in ux_range
    for uy_value in uy_range
    for omega_init_value in omega_init_range
    for theta_value in theta_range
])

with mp.Pool() as pool:
    results_list = pool.map(worker, param_grid)

param_grid = np.array(param_grid)
results = np.array(results_list)
# results[:, 0] = results[:, 0] - param_grid[:, 0]
# results[:, 1] = results[:, 1] - param_grid[:, 1]
# results[:, 2] = results[:, 2] - param_grid[:, 2]

np.savez(
    "results.npz",
    param_grid=param_grid,
    results=results,
)

# Make a plot
plot_data = np.concatenate((param_grid, results), axis=1).astype(float)
names = ["vx", "vy", "omega_after"]
ds = hv.Dataset(plot_data, ["ux", "uy", "omega_init", "theta"], names)
# ds = hv.Dataset(plot_data, ["ux", "uy", "omega_init", "theta"], ["delta_vx", "delta_vy", "delta_omega_after"])

graph = [ds.to(hv.Curve, "theta", name).overlay("uy").opts(
    opts.Curve(
        width=600,
        line_width=2,
        show_grid=True,
        framewise=True,
    )
) for name in names]
plots = (graph[0] + graph[1] + graph[2]).cols(1)
plots.opts(
    opts.Curve(
        show_grid=True,
    )
)

omega_init_range