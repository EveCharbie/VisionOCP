
"""
The goal of this program is to optimize the movement to achieve a 42/.
Phase 0 : Twist
Phase 1 : preparation for landing
"""

import numpy as np
import pickle
import biorbd_casadi as biorbd
import casadi as cas
import IPython
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

def custom_trampoline_bed_in_peripheral_vision(controller: PenaltyController) -> cas.MX:
    """
    This function aims to encourage the avatar to keep the trampoline bed in his peripheral vision.
    It is done by discretizing the vision cone into vectors and determining if the vector projection of the gaze are inside the trampoline bed.
    """

    q = controller
    a = 1.07  # Trampoline with/2
    b = 2.14  # Trampoline length/2
    n = 6  # order of the polynomial for the trampoline bed rectangle equation

    # Get the gaze vector
    eyes_vect_start_marker_idx = controller.model.marker_index(f'eyes_vect_start')
    eyes_vect_end_marker_idx = controller.model.marker_index(f'eyes_vect_end')
    gaze_vector = controller.model.markers(controller.states["q"].mx)[eyes_vect_end_marker_idx] - controller.model.markers(controller.states["q"].mx)[eyes_vect_start_marker_idx]

    point_in_the_plane = np.array([1, 2, -0.83])
    vector_normal_to_the_plane = np.array([0, 0, 1])
    obj = 0
    for i_r in range(11):
        for i_th in range(10):

            # Get this vector from the vision cone
            marker_idx = controller.model.marker_index(f'cone_approx_{i_r}_{i_th}')
            vector_origin = controller.model.markers(controller.states["q"].mx)[eyes_vect_start_marker_idx]
            vector_end = controller.model.markers(controller.states["q"].mx)[marker_idx]
            vector = vector_end - vector_origin

            # Get the intersection between the vector and the trampoline plane
            t = (cas.dot(point_in_the_plane, vector_normal_to_the_plane) - cas.dot(vector_normal_to_the_plane, vector_origin)) / cas.dot(
                vector, vector_normal_to_the_plane
            )
            point_projection = vector_origin + vector * cas.fabs(t)

            # Determine if the point is inside the trampoline bed
            # Rectangle equation : (x/a)**n + (y/b)**n = 1
            # The function is convoluted with tanh to make it:
            # 1. Continuous
            # 2. Not encourage to look to the middle of the trampoline bed
            # 3. Largely penalized when outside the trampoline bed
            # 4. Equaly penalized when looking upward
            obj += cas.tanh(((point_projection[0]/a)**n + (point_projection[1]/b)**n) - 1) + 1

    val = cas.if_else(gaze_vector[2] > -0.01, 2*10*11,
                cas.if_else(cas.fabs(gaze_vector[0]/gaze_vector[2]) > np.tan(3*np.pi/8), 2*10*11,
                            cas.if_else(cas.fabs(gaze_vector[1]/gaze_vector[2]) > np.tan(3*np.pi/8), 2*10*11, obj)))
    out = controller.mx_to_cx("peripheral_vision", val, controller.states["q"])

    return out


def prepare_ocp(
    biorbd_model_path: str, n_shooting: tuple, num_twists: int, n_threads: int, ode_solver: OdeSolver = OdeSolver.RK4(), WITH_VISUAL_CRITERIA: bool = False
) -> OptimalControlProgram:
    """
    Prepare the ocp
    Parameters
    ----------
    biorbd_model_path: str
        The path to the bioMod file
    n_shooting: int
        The number of shooting points
    ode_solver: OdeSolver
        The ode solver to use
    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    final_time = 1.47
    biorbd_model = (
        BiorbdModel(biorbd_model_path),
        BiorbdModel(biorbd_model_path),
    )

    nb_q = biorbd_model[0].nb_q
    nb_qdot = biorbd_model[0].nb_qdot
    nb_tau = nb_q - biorbd_model[0].nb_root

    # for lisibility
    if not WITH_VISUAL_CRITERIA:
        X = 0
        Y = 1
        Z = 2
        Xrot = 3
        Yrot = 4
        Zrot = 5
        ZrotRightUpperArm = 6
        YrotRightUpperArm = 7
        ZrotLeftUpperArm = 8
        YrotLeftUpperArm = 9
        vX = 0
        vY = 1
        vZ = 2
        vXrot = 3
        vYrot = 4
        vZrot = 5
        vZrotRightUpperArm = 6
        vYrotRightUpperArm = 7
        vZrotLeftUpperArm = 8
        vYrotLeftUpperArm = 9
    else:
        X = 0
        Y = 1
        Z = 2
        Xrot = 3
        Yrot = 4
        Zrot = 5
        ZrotHead = 6
        XrotHead = 7
        ZrotEyes = 8
        XrotEyes = 9
        ZrotRightUpperArm = 10
        YrotRightUpperArm = 11
        ZrotLeftUpperArm = 12
        YrotLeftUpperArm = 13
        vX = 0
        vY = 1
        vZ = 2
        vXrot = 3
        vYrot = 4
        vZrot = 5
        vZrotHead = 6
        vXrotHead = 7
        vZrotEyes = 8
        vXrotEyes = 9
        vZrotRightUpperArm = 10
        vYrotRightUpperArm = 11
        vZrotLeftUpperArm = 12
        vYrotLeftUpperArm = 13

    # Add objective functions
    objective_functions = ObjectiveList()

    # Min controls
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", node=Node.ALL_SHOOTING, weight=1, phase=0
    )
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", node=Node.ALL_SHOOTING, weight=1, phase=1
    )

    # Min control derivative
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", node=Node.ALL_SHOOTING, weight=1, phase=0, derivative=True,
    )
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", node=Node.ALL_SHOOTING, weight=1, phase=1, derivative=True,
    )

    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_TIME, min_bound=0.05, max_bound=final_time, weight=0.00001, phase=0
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_TIME, min_bound=0.05, max_bound=final_time / 2, weight=0.00001, phase=1
    )

    # aligning with the FIG regulations
    objective_functions.add(
         ObjectiveFcn.Lagrange.MINIMIZE_STATE,
         key="q",
         node=Node.ALL_SHOOTING,
         index=[YrotRightUpperArm, YrotLeftUpperArm],
         weight=100,
         phase=0,
    )


    if WITH_VISUAL_CRITERIA:

        # Spotting
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_SEGMENT_VELOCITY, segment="Head", weight=10000, phase=1)

        # Self-motion detection
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key='qdot', index=[ZrotEyes, XrotEyes], weight=100, phase=0)

        # Keeping the trampoline bed in the peripheral vision
        # objective_functions.add(custom_trampoline_bed_in_peripheral_vision, custom_type=ObjectiveFcn.Lagrange, weight=100, phase=0)

        # Quiet eye
        objective_functions.add(ObjectiveFcn.Lagrange.TRACK_VECTOR_ORIENTATIONS_FROM_MARKERS,
                                vector_0_marker_0="eyes_vect_start",
                                vector_0_marker_1="eyes_vect_end",
                                vector_1_marker_0="eyes_vect_start",
                                vector_1_marker_1="fixation_front",
                                weight=1, phase=0)
        objective_functions.add(ObjectiveFcn.Lagrange.TRACK_VECTOR_ORIENTATIONS_FROM_MARKERS,
                                vector_0_marker_0="eyes_vect_start",
                                vector_0_marker_1="eyes_vect_end",
                                vector_1_marker_0="eyes_vect_start",
                                vector_1_marker_1="fixation_front",
                                weight=10000, phase=1)

        # Avoid extreme eye and neck angles
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", index=[ZrotHead, XrotHead, ZrotEyes, XrotEyes], weight=10, phase=0)
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", index=[ZrotHead, XrotHead, ZrotEyes, XrotEyes], weight=10, phase=1)


    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)

    tau_min, tau_max, tau_init = -1000, 1000, 0
    u_bounds = BoundsList()
    u_bounds.add("tau", min_bound=[tau_min] * nb_tau, max_bound=[tau_max] * nb_tau, phase=0)
    u_bounds.add("tau", min_bound=[tau_min] * nb_tau, max_bound=[tau_max] * nb_tau, phase=1)

    u_init = InitialGuessList()
    u_init.add("tau", initial_guess=[tau_init] * nb_tau, phase=0)
    u_init.add("tau", initial_guess=[tau_init] * nb_tau, phase=1)

    # Path constraint
    x_bounds = BoundsList()
    q_bounds_min_0 = biorbd_model[0].bounds_from_ranges("q").min
    q_bounds_max_0 = biorbd_model[0].bounds_from_ranges("q").max
    qdot_bounds_min_0 = biorbd_model[0].bounds_from_ranges("qdot").min
    qdot_bounds_max_0 = biorbd_model[0].bounds_from_ranges("qdot").max
    q_bounds_min_1 = biorbd_model[1].bounds_from_ranges("q").min
    q_bounds_max_1 = biorbd_model[1].bounds_from_ranges("q").max
    qdot_bounds_min_1 = biorbd_model[1].bounds_from_ranges("qdot").min
    qdot_bounds_max_1 = biorbd_model[1].bounds_from_ranges("qdot").max
    
    # For lisibility
    START, MIDDLE, END = 0, 1, 2

    # ------------------------------- Phase 0 : twist ------------------------------- #
    zmax = 9.81 / 8 * final_time**2 + 1

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

    # Right arm
    q_bounds_min_0[YrotRightUpperArm, START] = 2.9
    q_bounds_max_0[YrotRightUpperArm, START] = 2.9
    q_bounds_min_0[ZrotRightUpperArm, START] = 0
    q_bounds_max_0[ZrotRightUpperArm, START] = 0
    # Left arm
    q_bounds_min_0[YrotLeftUpperArm, START] = -2.9
    q_bounds_max_0[YrotLeftUpperArm, START] = -2.9
    q_bounds_min_0[ZrotLeftUpperArm, START] = 0
    q_bounds_max_0[ZrotLeftUpperArm, START] = 0

    # Head and eyes
    if WITH_VISUAL_CRITERIA:
        q_bounds_min_0[ZrotHead, START] = -0.1
        q_bounds_max_0[ZrotHead, START] = 0.1
        q_bounds_min_0[XrotHead, START] = -0.1
        q_bounds_max_0[XrotHead, START] = 0.1
        q_bounds_min_0[ZrotEyes, START] = -0.1
        q_bounds_max_0[ZrotEyes, START] = 0.1
        q_bounds_min_0[XrotEyes, START] = np.pi/4 - 0.1
        q_bounds_max_0[XrotEyes, START] = np.pi/4 + 0.1

    vzinit = 9.81 / 2 * final_time

    # Shift the initial vertical speed at the CoM
    CoM_Q_sym = cas.MX.sym("CoM", nb_q)
    CoM_Q_init = q_bounds_min_0[:nb_q, START]
    CoM_Q_func = cas.Function("CoM_Q_func", [CoM_Q_sym], [biorbd_model[0].center_of_mass(CoM_Q_sym)])
    bassin_Q_func = cas.Function(
        "bassin_Q_func", [CoM_Q_sym], [biorbd_model[0].homogeneous_matrices_in_global(CoM_Q_sym, 0).to_mx()]
    )

    r = np.array(CoM_Q_func(CoM_Q_init)).reshape(1, 3) - np.array(bassin_Q_func(CoM_Q_init))[-1, :3]

    # Pelis translation velocities
    qdot_bounds_min_0[vX : vY + 1, :] = -10
    qdot_bounds_max_0[vX : vY + 1, :] = 10
    qdot_bounds_min_0[vX : vY + 1, START] = -0.5
    qdot_bounds_max_0[vX : vY + 1, START] = 0.5
    qdot_bounds_min_0[vZ, :] = -100
    qdot_bounds_max_0[vZ, :] = 100
    qdot_bounds_min_0[vZ, START] = vzinit - 0.5
    qdot_bounds_max_0[vZ, START] = vzinit + 0.5

    # Somersault
    qdot_bounds_min_0[vXrot, :] = -10
    qdot_bounds_max_0[vXrot, :] = -0.5
    # Tilt
    qdot_bounds_min_0[vYrot, :] = -100
    qdot_bounds_max_0[vYrot, :] = 100
    qdot_bounds_min_0[vYrot, START] = 0
    qdot_bounds_max_0[vYrot, START] = 0
    # Twist
    qdot_bounds_min_0[vZrot, :] = -100
    qdot_bounds_max_0[vZrot, :] = 100
    qdot_bounds_min_0[vZrot, START] = 0
    qdot_bounds_max_0[vZrot, START] = 0

    min_bound_trans_velocity = (
        qdot_bounds_min_0[vX : vZ + 1, START] + np.cross(r, qdot_bounds_min_0[vXrot : vZrot + 1, START])
    )[0]
    max_bound_trans_velocity = (
        qdot_bounds_max_0[vX : vZ + 1, START] + np.cross(r, qdot_bounds_max_0[vXrot : vZrot + 1, START])
    )[0]
    qdot_bounds_min_0[vX : vZ + 1, START] = (
        min(max_bound_trans_velocity[0], min_bound_trans_velocity[0]),
        min(max_bound_trans_velocity[1], min_bound_trans_velocity[1]),
        min(max_bound_trans_velocity[2], min_bound_trans_velocity[2]),
    )
    qdot_bounds_max_0[vX : vZ + 1, START] = (
        max(max_bound_trans_velocity[0], min_bound_trans_velocity[0]),
        max(max_bound_trans_velocity[1], min_bound_trans_velocity[1]),
        max(max_bound_trans_velocity[2], min_bound_trans_velocity[2]),
    )

    # Right arm
    qdot_bounds_min_0[vZrotRightUpperArm : vYrotRightUpperArm + 1, :] = -100
    qdot_bounds_max_0[vZrotRightUpperArm : vYrotRightUpperArm + 1, :] = 100
    qdot_bounds_min_0[vZrotRightUpperArm : vYrotRightUpperArm + 1, START] = 0
    qdot_bounds_max_0[vZrotRightUpperArm : vYrotRightUpperArm + 1, START] = 0
    # Left arm
    qdot_bounds_min_0[vZrotLeftUpperArm : vYrotLeftUpperArm + 1, :] = -100
    qdot_bounds_max_0[vZrotLeftUpperArm : vYrotLeftUpperArm + 1, :] = 100
    qdot_bounds_min_0[vZrotLeftUpperArm : vYrotLeftUpperArm + 1, START] = 0
    qdot_bounds_max_0[vZrotLeftUpperArm : vYrotLeftUpperArm + 1, START] = 0

    # ------------------------------- Phase 1 : landing ------------------------------- #

    # Pelvis translations
    q_bounds_min_1[X, :] = -0.25
    q_bounds_max_1[X, :] = 0.25
    q_bounds_min_1[Y, :] = -0.5
    q_bounds_max_1[Y, :] = 0.5
    q_bounds_min_1[Z, :] = 0
    q_bounds_max_1[Z, :] = zmax
    q_bounds_min_1[Z, END] = 0
    q_bounds_max_1[Z, END] = 0.1

    # Somersault
    q_bounds_min_1[Xrot, :] = -0.5 - 2 * np.pi - 0.1
    q_bounds_max_1[Xrot, :] = -3/2 * np.pi + 0.2 + 0.2
    q_bounds_min_1[Xrot, END] = 0.5 - 2 * np.pi - 0.1
    q_bounds_max_1[Xrot, END] = 0.5 - 2 * np.pi + 0.1
    # Tilt
    q_bounds_min_1[Yrot, :] = -np.pi / 16
    q_bounds_max_1[Yrot, :] = np.pi / 16
    # Twist
    q_bounds_min_1[Zrot, :] = 2 * np.pi * num_twists - np.pi / 8
    q_bounds_max_1[Zrot, :] = 2 * np.pi * num_twists + np.pi / 8

    # Right arm
    q_bounds_min_1[YrotRightUpperArm, START] = - 0.1
    q_bounds_max_1[YrotRightUpperArm, START] = + 0.1
    q_bounds_min_1[YrotRightUpperArm, END] = 2.9 - 0.1
    q_bounds_max_1[YrotRightUpperArm, END] = 2.9 + 0.1
    q_bounds_min_1[ZrotRightUpperArm, END] = -0.1
    q_bounds_max_1[ZrotRightUpperArm, END] = 0.1
    # Left arm
    q_bounds_min_1[YrotLeftUpperArm, START] = - 0.1
    q_bounds_max_1[YrotLeftUpperArm, START] = + 0.1
    q_bounds_min_1[YrotLeftUpperArm, END] = -2.9 - 0.1
    q_bounds_max_1[YrotLeftUpperArm, END] = -2.9 + 0.1
    q_bounds_min_1[ZrotLeftUpperArm, END] = -0.1
    q_bounds_max_1[ZrotLeftUpperArm, END] = 0.1

    # Translations velocities
    q_bounds_min_1[vX : vY + 1, :] = -10
    q_bounds_max_1[vX : vY + 1, :] = 10
    q_bounds_min_1[vZ, :] = -100
    q_bounds_max_1[vZ, :] = 100

    # Somersault
    qdot_bounds_min_1[vXrot, :] = -100
    qdot_bounds_max_1[vXrot, :] = 100
    # Tilt
    qdot_bounds_min_1[vYrot, :] = -100
    qdot_bounds_max_1[vYrot, :] = 100
    # Twist
    qdot_bounds_min_1[vZrot, :] = -100
    qdot_bounds_max_1[vZrot, :] = 100

    # Right arm
    qdot_bounds_min_1[vZrotRightUpperArm : vYrotRightUpperArm + 1, :] = -100
    qdot_bounds_max_1[vZrotRightUpperArm : vYrotRightUpperArm + 1, :] = 100
    # Left arm
    qdot_bounds_min_1[vZrotLeftUpperArm : vYrotLeftUpperArm + 1, :] = -100
    qdot_bounds_max_1[vZrotLeftUpperArm : vYrotLeftUpperArm + 1, :] = 100

    x_bounds.add("q", min_bound=q_bounds_min_0, max_bound=q_bounds_max_0, phase=0)
    x_bounds.add("qdot", min_bound=qdot_bounds_min_0, max_bound=qdot_bounds_max_0, phase=0)
    x_bounds.add("q", min_bound=q_bounds_min_1, max_bound=q_bounds_max_1, phase=1)
    x_bounds.add("qdot", min_bound=qdot_bounds_min_1, max_bound=qdot_bounds_max_1, phase=1)

    # ------------------------------- Initial guesses ------------------------------- #

    x0 = np.vstack((np.zeros((nb_q, 2)), np.zeros((nb_qdot, 2))))
    x1 = np.vstack((np.zeros((nb_q, 2)), np.zeros((nb_qdot, 2))))

    x0[Xrot] = np.array([0, -3/2 * np.pi])
    x0[Zrot] = np.array([0, 2 * np.pi * num_twists])
    x0[ZrotLeftUpperArm] = -0.75
    x0[ZrotRightUpperArm] = 0.75
    x0[YrotLeftUpperArm, 0] = -2.9
    x0[YrotRightUpperArm, 0] = 2.9
    x0[vXrot] = - 2 * np.pi

    x1[Xrot] = np.array([-3/2 * np.pi, -2 * np.pi])
    x1[Zrot] = np.array([2 * np.pi * num_twists, 2 * np.pi * num_twists])

    x_init = InitialGuessList()
    x_init.add("q", initial_guess=x0[:nb_q], interpolation=InterpolationType.LINEAR, phase=0)
    x_init.add("qdot", initial_guess=x0[nb_q:], interpolation=InterpolationType.LINEAR, phase=0)
    x_init.add("q", initial_guess=x1[:nb_q], interpolation=InterpolationType.LINEAR, phase=1)
    x_init.add("qdot", initial_guess=x1[nb_q:], interpolation=InterpolationType.LINEAR, phase=1)

    if WITH_VISUAL_CRITERIA: 
        variable_mappings = BiMappingList()
        variable_mappings.add("tau", to_second=[None, None, None, None, None, None, 0, 1, 2, 3, 4, 5, 6, 7], to_first=[6, 7, 8, 9, 10, 11, 12, 13])
    else:
        variable_mappings = BiMappingList()
        variable_mappings.add("tau", to_second=[None, None, None, None, None, None, 0, 1, 2, 3], to_first=[6, 7, 8, 9])
        
    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting,
        [final_time / len(biorbd_model)] * len(biorbd_model),
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        ode_solver=ode_solver,
        n_threads=n_threads,
        variable_mappings=variable_mappings,
        assume_phase_dynamics=True,
    )


def main():
    """
    Prepares and solves an ocp for a 42/ with and without visual criteria.
    """

    WITH_VISUAL_CRITERIA = True

    if WITH_VISUAL_CRITERIA:
        biorbd_model_path = "models/SoMe_42_with_visual_criteria.bioMod"
    else:
        biorbd_model_path = "models/SoMe_42.bioMod"

    n_shooting = (100, 40)
    num_twists = 1
    ocp = prepare_ocp(biorbd_model_path, n_shooting=n_shooting, num_twists=num_twists, n_threads=7, WITH_VISUAL_CRITERIA=WITH_VISUAL_CRITERIA)
    # ocp.add_plot_penalty(CostType.ALL)

    # solver = Solver.IPOPT(show_online_optim=True, show_options=dict(show_bounds=True))
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_linear_solver("ma57")
    solver.set_maximum_iterations(10)
    solver.set_convergence_tolerance(1e-4)

    tic = time.time()
    sol = ocp.solve(solver)
    toc = time.time() - tic
    print(toc)

    timestamp = time.strftime("%Y-%m-%d-%H%M")
    name = biorbd_model_path.split("/")[-1].removesuffix(".bioMod")
    qs = sol.states[0]["q"]
    qdots = sol.states[0]["qdot"]
    qddots = sol.controls[0]["tau"]
    for i in range(1, len(sol.states)):
        qs = np.hstack((qs, sol.states[i]["q"]))
        qdots = np.hstack((qdots, sol.states[i]["qdot"]))
        qddots = np.hstack((qddots, sol.controls[i]["tau"]))
    time_parameters = sol.parameters["time"]


    integrated_sol = sol.integrate(shooting_type=Shooting.SINGLE, integrator=SolutionIntegrator.SCIPY_DOP853)

    time_vector = integrated_sol.time[0]
    q_reintegrated = integrated_sol.states[0]["q"]
    qdot_reintegrated = integrated_sol.states[0]["qdot"]
    for i in range(1, len(sol.states)):
        time_vector = np.hstack((time_vector, integrated_sol.time[i]))
        q_reintegrated = np.hstack((q_reintegrated, integrated_sol.states[i]["q"]))
        qdot_reintegrated = np.hstack((qdot_reintegrated, integrated_sol.states[i]["qdot"]))

    del sol.ocp
    with open(f"Solutions/{name}-{str(n_shooting).replace(', ', '_')}-{timestamp}.pkl", "wb") as f:
        pickle.dump((sol, qs, qdots, qddots, time_parameters, q_reintegrated, qdot_reintegrated, time_vector), f)

    # sol.animate(n_frames=-1, show_floor=False)

if __name__ == "__main__":
    main()
