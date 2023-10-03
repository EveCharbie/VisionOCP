
"""
The goal of this program is to show that the RAM consumption of bioptim if going crazy
"""

import numpy as np
import pickle
import biorbd_casadi as biorbd
import casadi as cas
import time
import sys

from bioptim import (
    OptimalControlProgram,
    DynamicsList,
    DynamicsFcn,
    ObjectiveList,
    ObjectiveFcn,
    BoundsList,
    InitialGuessList,
    InterpolationType,
    OdeSolver,
    Node,
    Solver,
    BiMappingList,
    CostType,
    ConstraintList,
    ConstraintFcn,
    PenaltyController,
    BiorbdModel,
    Shooting,
    SolutionIntegrator,
)

def prepare_ocp(
    biorbd_model_path: str, n_shooting: tuple, num_twists: int, n_threads: int, ode_solver: OdeSolver = OdeSolver.RK4(), WITH_VISUAL_CRITERIA: bool = False
) -> OptimalControlProgram:
    """
    Prepare the ocp
    Parameters
    ----------
    biorbd_model_path: str
        The path to the bioMod file
    n_shooting: tuple
        The number of shooting points
    ode_solver: OdeSolver
        The ode solver to use
    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    final_time = 1.47
    biorbd_model = BiorbdModel(biorbd_model_path)

    nb_q = biorbd_model.nb_q
    nb_tau = nb_q - biorbd_model.nb_root

    X = 0
    Y = 1
    Z = 2
    Xrot = 3
    Yrot = 4
    Zrot = 5

    # Add objective functions
    objective_functions = ObjectiveList()

    # Min controls
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", node=Node.ALL_SHOOTING, weight=1, phase=0
    )

    # # Min control derivative
    # objective_functions.add(
    #     ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", node=Node.ALL_SHOOTING, weight=1, phase=0, derivative=True,
    # )
    #
    # objective_functions.add(
    #     ObjectiveFcn.Mayer.MINIMIZE_TIME, min_bound=0.05, max_bound=final_time, weight=0.00001, phase=0
    # )

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)

    tau_min, tau_max, tau_init = -1000, 1000, 0
    u_bounds = BoundsList()
    u_bounds.add("tau", min_bound=[tau_min] * nb_tau, max_bound=[tau_max] * nb_tau, phase=0)

    u_init = InitialGuessList()
    u_init.add("tau", initial_guess=[tau_init] * nb_tau, phase=0)

    # Path constraint
    x_bounds = BoundsList()
    q_bounds_min_0 = biorbd_model.bounds_from_ranges("q").min
    q_bounds_max_0 = biorbd_model.bounds_from_ranges("q").max
    qdot_bounds_min_0 = biorbd_model.bounds_from_ranges("qdot").min
    qdot_bounds_max_0 = biorbd_model.bounds_from_ranges("qdot").max
    
    # For lisibility
    START, MIDDLE, END = 0, 1, 2
    zmax = 9.81 / 8 * final_time**2 + 1
    vzinit = 9.81 / 2 * final_time

    # Pelvis translations
    q_bounds_min_0[X, :] = -0.25
    q_bounds_max_0[X, :] = 0.25
    q_bounds_min_0[Y, :] = -0.5
    q_bounds_max_0[Y, :] = 0.5
    q_bounds_min_0[: Z + 1, START] = 0
    q_bounds_max_0[: Z + 1, START] = 0
    q_bounds_min_0[Z, MIDDLE:] = 0
    q_bounds_max_0[Z, MIDDLE:] = zmax

    # Somersault
    q_bounds_min_0[Xrot, START] = 0
    q_bounds_max_0[Xrot, START] = 0
    q_bounds_min_0[Xrot, MIDDLE:] = -3/2 * np.pi
    q_bounds_max_0[Xrot, MIDDLE:] = 0.5
    # Tilt
    q_bounds_min_0[Yrot, START] = 0
    q_bounds_max_0[Yrot, START] = 0
    q_bounds_min_0[Yrot, MIDDLE:] = -np.pi / 4  # avoid gimbal lock
    q_bounds_max_0[Yrot, MIDDLE:] = np.pi / 4
    # Twist
    q_bounds_min_0[Zrot, START] = 0
    q_bounds_max_0[Zrot, START] = 0
    q_bounds_min_0[Zrot, MIDDLE] = -0.5
    q_bounds_max_0[Zrot, MIDDLE] = 2 * np.pi * num_twists
    q_bounds_min_0[Zrot, END] = 2 * np.pi * num_twists - 0.5
    q_bounds_max_0[Zrot, END] = 2 * np.pi * num_twists + 0.5

    # Pelis translation velocities
    qdot_bounds_min_0[X : Y + 1, :] = -10
    qdot_bounds_max_0[X : Y + 1, :] = 10
    qdot_bounds_min_0[X : Y + 1, START] = -0.5
    qdot_bounds_max_0[X : Y + 1, START] = 0.5
    qdot_bounds_min_0[Z, :] = -100
    qdot_bounds_max_0[Z, :] = 100
    qdot_bounds_min_0[Z, START] = vzinit - 0.5
    qdot_bounds_max_0[Z, START] = vzinit + 0.5

    # Somersault
    qdot_bounds_min_0[Xrot, :] = -10
    qdot_bounds_max_0[Xrot, :] = -0.5
    # Tilt
    qdot_bounds_min_0[Yrot, :] = -100
    qdot_bounds_max_0[Yrot, :] = 100
    qdot_bounds_min_0[Yrot, START] = 0
    qdot_bounds_max_0[Yrot, START] = 0
    # Twist
    qdot_bounds_min_0[Zrot, :] = -100
    qdot_bounds_max_0[Zrot, :] = 100
    qdot_bounds_min_0[Zrot, START] = 0
    qdot_bounds_max_0[Zrot, START] = 0

    x_bounds.add("q", min_bound=q_bounds_min_0, max_bound=q_bounds_max_0, phase=0)
    x_bounds.add("qdot", min_bound=qdot_bounds_min_0, max_bound=qdot_bounds_max_0, phase=0)

    # ------------------------------- Initial guesses ------------------------------- #

    x0 = np.vstack((np.zeros((nb_q, 2)), np.zeros((nb_q, 2))))

    x0[Xrot] = np.array([0, -3/2 * np.pi])
    x0[Zrot] = np.array([0, 2 * np.pi * num_twists])

    x_init = InitialGuessList()
    x_init.add("q", initial_guess=x0[:nb_q], interpolation=InterpolationType.LINEAR, phase=0)
    x_init.add("qdot", initial_guess=x0[nb_q:], interpolation=InterpolationType.LINEAR, phase=0)

    variable_mappings = BiMappingList()
    variable_mappings.add("tau", to_second=[None, None, None, None, None, None, 0], to_first=[6])
        
    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting,
        final_time,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        ode_solver=ode_solver,
        n_threads=1,
        variable_mappings=variable_mappings,
        assume_phase_dynamics=True,
    )


def main():

    biorbd_model_path = "models/modele_for_minmal_example.bioMod"

    n_shooting = 100
    num_twists = 1
    ocp = prepare_ocp(biorbd_model_path, n_shooting=n_shooting, num_twists=num_twists, n_threads=7)

    # solver = Solver.IPOPT(show_online_optim=True, show_options=dict(show_bounds=True))
    solver = Solver.IPOPT(show_online_optim=False)
    # solver.set_linear_solver("ma57")
    solver.set_maximum_iterations(10)
    solver.set_convergence_tolerance(1e-4)

    tic = time.time()
    sol = ocp.solve(solver)
    toc = time.time() - tic
    print(toc)

if __name__ == "__main__":
    main()
