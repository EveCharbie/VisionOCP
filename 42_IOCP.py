"""
This code uses a genetic algorithm to find the optimal weights for the OCP (42/).
"""

import pygmo as pg
import numpy as np
import matplotlib.pyplot as plt
import pickle
import casadi as cas
import biorbd
from IPython import embed

import sys
sys.path.append("/home/mickaelbegon/Documents/Eve/BiorbdOptim")
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
    BiorbdModel,
    Shooting,
    SolutionIntegrator,
)
from get_kinematics_and_gaze import reorder_markers_xsens
from TechOpt42 import custom_trampoline_bed_in_peripheral_vision


def set_bounds(biorbd_model, final_time, num_twists, nb_q, nb_qdot, nb_qddot_joints, fancy_names_index):

    # Path constraint
    bounds_q_min_0 = biorbd_model[0].bounds_from_ranges("q").min
    bounds_q_max_0 = biorbd_model[0].bounds_from_ranges("q").max
    bounds_qdot_min_0 = biorbd_model[0].bounds_from_ranges("qdot").min
    bounds_qdot_max_0 = biorbd_model[0].bounds_from_ranges("qdot").max
    bounds_q_min_1 = biorbd_model[1].bounds_from_ranges("q").min
    bounds_q_max_1 = biorbd_model[1].bounds_from_ranges("q").max
    bounds_qdot_min_1 = biorbd_model[1].bounds_from_ranges("qdot").min
    bounds_qdot_max_1 = biorbd_model[1].bounds_from_ranges("qdot").max

    # For lisibility
    START, MIDDLE, END = 0, 1, 2

    # ------------------------------- Phase 0 : twist ------------------------------- #
    zmax = 9.81 / 8 * final_time**2 + 1

    # Pelvis translations
    bounds_q_min_0[fancy_names_index["X"], :] = -0.25
    bounds_q_max_0[fancy_names_index["X"], :] = 0.25
    bounds_q_min_0[fancy_names_index["Y"], :] = -0.5
    bounds_q_max_0[fancy_names_index["Y"], :] = 0.5
    bounds_q_min_0[: fancy_names_index["Z"] + 1, START] = 0
    bounds_q_max_0[: fancy_names_index["Z"] + 1, START] = 0
    bounds_q_min_0[fancy_names_index["Z"], MIDDLE:] = 0
    bounds_q_max_0[fancy_names_index["Z"], MIDDLE:] = zmax

    # Somersault
    bounds_q_min_0[fancy_names_index["Xrot"], START] = 0
    bounds_q_max_0[fancy_names_index["Xrot"], START] = 0
    bounds_q_min_0[fancy_names_index["Xrot"], MIDDLE:] = -3/2 * np.pi
    bounds_q_max_0[fancy_names_index["Xrot"], MIDDLE:] = 0.5
    # Tilt
    bounds_q_min_0[fancy_names_index["Yrot"], START] = 0
    bounds_q_max_0[fancy_names_index["Yrot"], START] = 0
    bounds_q_min_0[fancy_names_index["Yrot"], MIDDLE:] = -np.pi / 4  # avoid gimbal lock
    bounds_q_max_0[fancy_names_index["Yrot"], MIDDLE:] = np.pi / 4
    # Twist
    bounds_q_min_0[fancy_names_index["Zrot"], START] = 0
    bounds_q_max_0[fancy_names_index["Zrot"], START] = 0
    bounds_q_min_0[fancy_names_index["Zrot"], MIDDLE] = -0.5
    bounds_q_max_0[fancy_names_index["Zrot"], MIDDLE] = 2 * np.pi * num_twists
    bounds_q_min_0[fancy_names_index["Zrot"], END] = 2 * np.pi * num_twists - 0.5
    bounds_q_max_0[fancy_names_index["Zrot"], END] = 2 * np.pi * num_twists + 0.5

    # Right arm
    bounds_q_min_0[fancy_names_index["YrotRightUpperArm"], START] = 2.9
    bounds_q_max_0[fancy_names_index["YrotRightUpperArm"], START] = 2.9
    bounds_q_min_0[fancy_names_index["ZrotRightUpperArm"], START] = 0
    bounds_q_max_0[fancy_names_index["ZrotRightUpperArm"], START] = 0
    # Left arm
    bounds_q_min_0[fancy_names_index["YrotLeftUpperArm"], START] = -2.9
    bounds_q_max_0[fancy_names_index["YrotLeftUpperArm"], START] = -2.9
    bounds_q_min_0[fancy_names_index["ZrotLeftUpperArm"], START] = 0
    bounds_q_max_0[fancy_names_index["ZrotLeftUpperArm"], START] = 0

    bounds_q_min_0[fancy_names_index["ZrotHead"], START] = -0.1
    bounds_q_max_0[fancy_names_index["ZrotHead"], START] = 0.1
    bounds_q_min_0[fancy_names_index["XrotHead"], START] = -0.1
    bounds_q_max_0[fancy_names_index["XrotHead"], START] = 0.1
    bounds_q_min_0[fancy_names_index["ZrotEyes"], START] = -0.1
    bounds_q_max_0[fancy_names_index["ZrotEyes"], START] = 0.1
    bounds_q_min_0[fancy_names_index["XrotEyes"], START] = np.pi/4 - 0.1
    bounds_q_max_0[fancy_names_index["XrotEyes"], START] = np.pi/4 + 0.1

    vzinit = 9.81 / 2 * final_time

    # Shift the initial vertical speed at the CoM
    CoM_Q_sym = cas.MX.sym("CoM", nb_q)
    CoM_Q_init = bounds_q_min_0[:nb_q, START]
    CoM_Q_func = cas.Function("CoM_Q_func", [CoM_Q_sym], [biorbd_model[0].center_of_mass(CoM_Q_sym)])
    bassin_Q_func = cas.Function(
        "bassin_Q_func", [CoM_Q_sym], [biorbd_model[0].homogeneous_matrices_in_global(CoM_Q_sym, 0).to_mx()]
    )

    r = np.array(CoM_Q_func(CoM_Q_init)).reshape(1, 3) - np.array(bassin_Q_func(CoM_Q_init))[-1, :3]

    # Pelis translation velocities
    bounds_qdot_min_0[fancy_names_index["vX"] : fancy_names_index["vY"] + 1, :] = -10
    bounds_qdot_max_0[fancy_names_index["vX"] : fancy_names_index["vY"] + 1, :] = 10
    bounds_qdot_min_0[fancy_names_index["vX"] : fancy_names_index["vY"] + 1, START] = -0.5
    bounds_qdot_max_0[fancy_names_index["vX"] : fancy_names_index["vY"] + 1, START] = 0.5
    bounds_qdot_min_0[fancy_names_index["vZ"], :] = -100
    bounds_qdot_max_0[fancy_names_index["vZ"], :] = 100
    bounds_qdot_min_0[fancy_names_index["vZ"], START] = vzinit - 0.5
    bounds_qdot_max_0[fancy_names_index["vZ"], START] = vzinit + 0.5

    # Somersault
    bounds_qdot_min_0[fancy_names_index["vXrot"], :] = -10
    bounds_qdot_max_0[fancy_names_index["vXrot"], :] = -0.5
    # Tile
    bounds_qdot_min_0[fancy_names_index["vYrot"], :] = -100
    bounds_qdot_max_0[fancy_names_index["vYrot"], :] = 100
    bounds_qdot_min_0[fancy_names_index["vYrot"], START] = 0
    bounds_qdot_max_0[fancy_names_index["vYrot"], START] = 0
    # Twist
    bounds_qdot_min_0[fancy_names_index["vZrot"], :] = -100
    bounds_qdot_max_0[fancy_names_index["vZrot"], :] = 100
    bounds_qdot_min_0[fancy_names_index["vZrot"], START] = 0
    bounds_qdot_max_0[fancy_names_index["vZrot"], START] = 0

    min_bound_trans_velocity = (
        bounds_qdot_min_0[fancy_names_index["vX"] : fancy_names_index["vZ"] + 1, START] + np.cross(r, bounds_qdot_min_0[fancy_names_index["vXrot"] : fancy_names_index["vZrot"] + 1, START])
    )[0]
    max_bound_trans_velocity = (
        bounds_qdot_max_0[fancy_names_index["vX"] : fancy_names_index["vZ"] + 1, START] + np.cross(r, bounds_qdot_min_0[fancy_names_index["vXrot"] : fancy_names_index["vZrot"] + 1, START])
    )[0]
    bounds_qdot_min_0[fancy_names_index["vX"] : fancy_names_index["vZ"] + 1, START] = (
        min(max_bound_trans_velocity[0], min_bound_trans_velocity[0]),
        min(max_bound_trans_velocity[1], min_bound_trans_velocity[1]),
        min(max_bound_trans_velocity[2], min_bound_trans_velocity[2]),
    )
    bounds_qdot_max_0[fancy_names_index["vX"] : fancy_names_index["vZ"] + 1, START] = (
        max(max_bound_trans_velocity[0], min_bound_trans_velocity[0]),
        max(max_bound_trans_velocity[1], min_bound_trans_velocity[1]),
        max(max_bound_trans_velocity[2], min_bound_trans_velocity[2]),
    )

    # Head and eyes
    bounds_qdot_min_0[fancy_names_index["vZrotHead"] : fancy_names_index["vXrotEyes"] + 1, :] = -100
    bounds_qdot_max_0[fancy_names_index["vZrotHead"] : fancy_names_index["vXrotEyes"] + 1, :] = 100
    bounds_qdot_min_0[fancy_names_index["vZrotHead"] : fancy_names_index["vXrotEyes"] + 1, START] = 0
    bounds_qdot_max_0[fancy_names_index["vZrotHead"]: fancy_names_index["vXrotEyes"] + 1, START] = 0

    # Right arm
    bounds_qdot_min_0[fancy_names_index["vZrotRightUpperArm"] : fancy_names_index["vYrotRightUpperArm"] + 1, :] = -100
    bounds_qdot_max_0[fancy_names_index["vZrotRightUpperArm"] : fancy_names_index["vYrotRightUpperArm"] + 1, :] = 100
    bounds_qdot_min_0[fancy_names_index["vZrotRightUpperArm"] : fancy_names_index["vYrotRightUpperArm"] + 1, START] = 0
    bounds_qdot_max_0[fancy_names_index["vZrotRightUpperArm"]: fancy_names_index["vYrotRightUpperArm"] + 1, START] = 0
    # Left arm
    bounds_qdot_min_0[fancy_names_index["vZrotLeftUpperArm"] : fancy_names_index["vYrotLeftUpperArm"] + 1, :] = -100
    bounds_qdot_max_0[fancy_names_index["vZrotLeftUpperArm"] : fancy_names_index["vYrotLeftUpperArm"] + 1, :] = 100
    bounds_qdot_min_0[fancy_names_index["vZrotLeftUpperArm"] : fancy_names_index["vYrotLeftUpperArm"] + 1, START] = 0
    bounds_qdot_max_0[fancy_names_index["vZrotLeftUpperArm"]: fancy_names_index["vYrotLeftUpperArm"] + 1, START] = 0

    # ------------------------------- Phase 1 : landing ------------------------------- #

    # Pelvis translations
    bounds_q_min_1[fancy_names_index["X"], :] = -0.25
    bounds_q_max_1[fancy_names_index["X"], :] = 0.25
    bounds_q_min_1[fancy_names_index["Y"], :] = -0.5
    bounds_q_max_1[fancy_names_index["Y"], :] = 0.5
    bounds_q_min_1[fancy_names_index["Z"], :] = 0
    bounds_q_max_1[fancy_names_index["Z"], :] = zmax
    bounds_q_min_1[fancy_names_index["Z"], END] = 0
    bounds_q_max_1[fancy_names_index["Z"], END] = 0.1

    # Somersault
    bounds_q_min_1[fancy_names_index["Xrot"], :] = -0.5 - 2 * np.pi - 0.1
    bounds_q_max_1[fancy_names_index["Xrot"], :] = -3/2 * np.pi + 0.2 + 0.2
    bounds_q_min_1[fancy_names_index["Xrot"], END] = 0.5 - 2 * np.pi - 0.1
    bounds_q_max_1[fancy_names_index["Xrot"], END] = 0.5 - 2 * np.pi + 0.1
    # Tilt
    bounds_q_min_1[fancy_names_index["Yrot"], :] = -np.pi / 16
    bounds_q_max_1[fancy_names_index["Yrot"], :] = np.pi / 16
    # Twist
    bounds_q_min_1[fancy_names_index["Zrot"], :] = 2 * np.pi * num_twists - np.pi / 8
    bounds_q_max_1[fancy_names_index["Zrot"], :] = 2 * np.pi * num_twists + np.pi / 8

    # Right arm
    bounds_q_min_1[fancy_names_index["YrotRightUpperArm"], START] = - 0.1
    bounds_q_max_1[fancy_names_index["YrotRightUpperArm"], START] = + 0.1
    bounds_q_min_1[fancy_names_index["YrotRightUpperArm"], END] = 2.9 - 0.1
    bounds_q_max_1[fancy_names_index["YrotRightUpperArm"], END] = 2.9 + 0.1
    bounds_q_min_1[fancy_names_index["ZrotRightUpperArm"], END] = -0.1
    bounds_q_max_1[fancy_names_index["ZrotRightUpperArm"], END] = 0.1
    # Left arm
    bounds_q_min_1[fancy_names_index["YrotLeftUpperArm"], START] = - 0.1
    bounds_q_max_1[fancy_names_index["YrotLeftUpperArm"], START] = + 0.1
    bounds_q_min_1[fancy_names_index["YrotLeftUpperArm"], END] = -2.9 - 0.1
    bounds_q_max_1[fancy_names_index["YrotLeftUpperArm"], END] = -2.9 + 0.1
    bounds_q_min_1[fancy_names_index["ZrotLeftUpperArm"], END] = -0.1
    bounds_q_max_1[fancy_names_index["ZrotLeftUpperArm"], END] = 0.1

    # Translations velocities
    bounds_qdot_min_1[fancy_names_index["vX"] : fancy_names_index["vY"] + 1, :] = -10
    bounds_qdot_max_1[fancy_names_index["vX"] : fancy_names_index["vY"] + 1, :] = 10
    bounds_qdot_min_1[fancy_names_index["vZ"], :] = -100
    bounds_qdot_max_1[fancy_names_index["vZ"], :] = 100

    # Somersault
    bounds_qdot_min_1[fancy_names_index["vXrot"], :] = -100
    bounds_qdot_max_1[fancy_names_index["vXrot"], :] = 100
    # Tilt
    bounds_qdot_min_1[fancy_names_index["vYrot"], :] = -100
    bounds_qdot_max_1[fancy_names_index["vYrot"], :] = 100
    # Twist
    bounds_qdot_min_1[fancy_names_index["vZrot"], :] = -100
    bounds_qdot_max_1[fancy_names_index["vZrot"], :] = 100

    # Head and eyes
    bounds_qdot_min_1[fancy_names_index["vZrotHead"] : fancy_names_index["vXrotEyes"] + 1, :] = -100
    bounds_qdot_max_1[fancy_names_index["vZrotHead"] : fancy_names_index["vXrotEyes"] + 1, :] = 100
    bounds_qdot_min_1[fancy_names_index["vZrotHead"] : fancy_names_index["vXrotEyes"] + 1, START] = 0
    bounds_qdot_max_1[fancy_names_index["vZrotHead"]: fancy_names_index["vXrotEyes"] + 1, START] = 0

    # Right arm
    bounds_qdot_min_1[fancy_names_index["vZrotRightUpperArm"] : fancy_names_index["vYrotRightUpperArm"] + 1, :] = -100
    bounds_qdot_max_1[fancy_names_index["vZrotRightUpperArm"] : fancy_names_index["vYrotRightUpperArm"] + 1, :] = 100
    # Left arm
    bounds_qdot_min_1[fancy_names_index["vZrotLeftUpperArm"] : fancy_names_index["vYrotLeftUpperArm"] + 1, :] = -100
    bounds_qdot_max_1[fancy_names_index["vZrotLeftUpperArm"] : fancy_names_index["vYrotLeftUpperArm"] + 1, :] = 100

    x_bounds = BoundsList()
    x_bounds.add("q", min_bound=bounds_q_min_0, max_bound=bounds_q_max_0, phase=0)
    x_bounds.add("qdot", min_bound=bounds_qdot_min_0, max_bound=bounds_qdot_max_0, phase=0)
    x_bounds.add("q", min_bound=bounds_q_min_1, max_bound=bounds_q_max_1, phase=1)
    x_bounds.add("qdot", min_bound=bounds_qdot_min_1, max_bound=bounds_qdot_max_1, phase=1)

    qddot_joints_min, qddot_joints_max, qddot_joints_init = -500, 500, 0
    u_bounds = BoundsList()
    u_bounds.add("qddot_joints", min_bound=[qddot_joints_min] * nb_qddot_joints, max_bound=[qddot_joints_max] * nb_qddot_joints, phase=0)
    u_bounds.add("qddot_joints", min_bound=[qddot_joints_min] * nb_qddot_joints, max_bound=[qddot_joints_max] * nb_qddot_joints, phase=1)

    return x_bounds, u_bounds


def set_initial_guesses(nb_q, nb_qdot, num_twists, nb_qddot_joints, fancy_names_index):

    q_0 = np.zeros((nb_q, 2))
    qdot_0 = np.zeros((nb_qdot, 2))
    q_1 = np.zeros((nb_q, 2))
    qdot_1 = np.zeros((nb_qdot, 2))

    q_0[fancy_names_index["Xrot"]] = np.array([0, -3/2 * np.pi])
    q_0[fancy_names_index["Zrot"]] = np.array([0, 2 * np.pi * num_twists])
    q_0[fancy_names_index["ZrotLeftUpperArm"]] = -0.75
    q_0[fancy_names_index["ZrotRightUpperArm"]] = 0.75
    q_0[fancy_names_index["YrotLeftUpperArm"], 0] = -2.9
    q_0[fancy_names_index["YrotRightUpperArm"], 0] = 2.9
    qdot_0[fancy_names_index["vXrot"]] = - 2 * np.pi

    q_1[fancy_names_index["Xrot"]] = np.array([-3/2 * np.pi, -2 * np.pi])
    q_1[fancy_names_index["Zrot"]] = np.array([2 * np.pi * num_twists, 2 * np.pi * num_twists])

    x_init = InitialGuessList()
    x_init.add("q", initial_guess=q_0, interpolation=InterpolationType.LINEAR, phase=0)
    x_init.add("qdot", initial_guess=qdot_0, interpolation=InterpolationType.LINEAR, phase=0)
    x_init.add("q", initial_guess=q_1, interpolation=InterpolationType.LINEAR, phase=1)
    x_init.add("qdot", initial_guess=qdot_1, interpolation=InterpolationType.LINEAR, phase=1)

    u_init = InitialGuessList()
    u_init.add("qddot_joints", initial_guess=[0] * nb_qddot_joints, phase=0)
    u_init.add("qddot_joints", initial_guess=[0] * nb_qddot_joints, phase=1)

    return x_init, u_init


def set_fancy_names_index(nb_q):
    """
    For readability
    """
    fancy_names_index = {}
    fancy_names_index["X"] = 0
    fancy_names_index["Y"] = 1
    fancy_names_index["Z"] = 2
    fancy_names_index["Xrot"] = 3
    fancy_names_index["Yrot"] = 4
    fancy_names_index["Zrot"] = 5
    fancy_names_index["ZrotHead"] = 6
    fancy_names_index["XrotHead"] = 7
    fancy_names_index["ZrotEyes"] = 8
    fancy_names_index["XrotEyes"] = 9
    fancy_names_index["ZrotRightUpperArm"] = 10
    fancy_names_index["YrotRightUpperArm"] = 11
    fancy_names_index["ZrotLeftUpperArm"] = 12
    fancy_names_index["YrotLeftUpperArm"] = 13
    fancy_names_index["vX"] = 0
    fancy_names_index["vY"] = 1
    fancy_names_index["vZ"] = 2
    fancy_names_index["vXrot"] = 3
    fancy_names_index["vYrot"] = 4
    fancy_names_index["vZrot"] = 5
    fancy_names_index["vZrotHead"] = 6
    fancy_names_index["vXrotHead"] = 7
    fancy_names_index["vZrotEyes"] = 8
    fancy_names_index["vXrotEyes"] = 9
    fancy_names_index["vZrotRightUpperArm"] = 10
    fancy_names_index["vYrotRightUpperArm"] = 11
    fancy_names_index["vZrotLeftUpperArm"] = 12
    fancy_names_index["vYrotLeftUpperArm"] = 13
    return fancy_names_index

def prepare_ocp(weights, coefficients, biorbd_model_path) -> OptimalControlProgram:

    # for lisibility
    weight_qddot_joint = weights[0]
    coefficient_qddot_joint = coefficients[0]
    weight_qddot_joint_derivative = weights[1]
    coefficient_qddot_joint_derivative = coefficients[1]
    weight_time = weights[2]
    coefficient_time = coefficients[2]
    weight_arms = weights[3]
    coefficient_arms = coefficients[3]
    weight_spotting = weights[4]
    coefficient_spotting = coefficients[4]
    weight_self_motion_detection = weights[5]
    coefficient_self_motion_detection = coefficients[5]
    weight_peripheral_vision = weights[6]
    coefficient_peripheral_vision = coefficients[6]
    weight_quiet_eye_phase_0 = weights[7]
    coefficient_quiet_eye_phase_0 = coefficients[7]
    weight_quiet_eye_phase_1 = weights[8]
    coefficient_quiet_eye_phase_1 = coefficients[8]
    weight_extreme_angles = weights[9]
    coefficient_extreme_angles = coefficients[9]


    num_twists = 1
    n_threads = 32
    n_shooting = (120, 40)
    final_time = 1.47

    biorbd_model = (
        BiorbdModel(biorbd_model_path),
        BiorbdModel(biorbd_model_path),
    )

    nb_q = biorbd_model[0].nb_q
    nb_qdot = biorbd_model[0].nb_qdot
    nb_qddot_joints = nb_q - biorbd_model[0].nb_root

    fancy_names_index = set_fancy_names_index(nb_q)

    # Add objective functions
    objective_functions = ObjectiveList()

    # Min controls
    if weight_qddot_joint*coefficient_qddot_joint != 0:
        objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="qddot_joints", node=Node.ALL_SHOOTING, weight=weight_qddot_joint*coefficient_qddot_joint, phase=0
        )
        objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="qddot_joints", node=Node.ALL_SHOOTING, weight=weight_qddot_joint*coefficient_qddot_joint, phase=1
        )

    # Min control derivative
    if weight_qddot_joint_derivative*coefficient_qddot_joint_derivative != 0:
        objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="qddot_joints", node=Node.ALL_SHOOTING, weight=weight_qddot_joint_derivative*coefficient_qddot_joint_derivative, phase=0, derivative=True,
        )
        objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="qddot_joints", node=Node.ALL_SHOOTING, weight=weight_qddot_joint_derivative*coefficient_qddot_joint_derivative, phase=1, derivative=True,
        )

    if weight_time*coefficient_time != 0:
        objective_functions.add(
            ObjectiveFcn.Mayer.MINIMIZE_TIME, min_bound=0.05, max_bound=final_time, weight=weight_time*coefficient_time, phase=0
        )
        objective_functions.add(
            ObjectiveFcn.Mayer.MINIMIZE_TIME, min_bound=0.05, max_bound=final_time / 2, weight=weight_time*coefficient_time, phase=1
        )
    else:
        objective_functions.add(
            ObjectiveFcn.Mayer.MINIMIZE_TIME, min_bound=0.05, max_bound=final_time, weight=1e-10, phase=0
        )
        objective_functions.add(
            ObjectiveFcn.Mayer.MINIMIZE_TIME, min_bound=0.05, max_bound=final_time / 2, weight=1e-10, phase=1
        )

    # aligning with the FIG regulations
    if weight_arms*coefficient_arms != 0:
        objective_functions.add(
             ObjectiveFcn.Lagrange.MINIMIZE_STATE,
             key="q",
             node=Node.ALL_SHOOTING,
             index=[fancy_names_index["YrotRightUpperArm"], fancy_names_index["YrotLeftUpperArm"]],
             weight=weight_arms*coefficient_arms,
             phase=0,
        )

    # --- Visual criteria ---
    # Spotting
    if weight_spotting*coefficient_spotting != 0:
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_SEGMENT_VELOCITY, segment="Head", weight=weight_spotting*coefficient_spotting, phase=1)

    # Self-motion detection
    if weight_self_motion_detection*coefficient_self_motion_detection != 0:
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key='qdot', index=[fancy_names_index["ZrotEyes"], fancy_names_index["XrotEyes"]],
                                weight=weight_self_motion_detection*coefficient_self_motion_detection, phase=0)

    # Keeping the trampoline bed in the peripheral vision
    if weight_peripheral_vision*coefficient_peripheral_vision != 0:
        objective_functions.add(custom_trampoline_bed_in_peripheral_vision, custom_type=ObjectiveFcn.Lagrange, weight=weight_peripheral_vision*coefficient_peripheral_vision, phase=0)

    # Quiet eye
    if weight_quiet_eye_phase_0*coefficient_quiet_eye_phase_0 != 0:
        objective_functions.add(ObjectiveFcn.Lagrange.TRACK_VECTOR_ORIENTATIONS_FROM_MARKERS,
                                vector_0_marker_0="eyes_vect_start",
                                vector_0_marker_1="eyes_vect_end",
                                vector_1_marker_0="eyes_vect_start",
                                vector_1_marker_1="fixation_front",
                                weight=weight_quiet_eye_phase_0*coefficient_quiet_eye_phase_0, phase=0)
    if weight_quiet_eye_phase_1*coefficient_quiet_eye_phase_1 != 0:
        objective_functions.add(ObjectiveFcn.Lagrange.TRACK_VECTOR_ORIENTATIONS_FROM_MARKERS,
                                vector_0_marker_0="eyes_vect_start",
                                vector_0_marker_1="eyes_vect_end",
                                vector_1_marker_0="eyes_vect_start",
                                vector_1_marker_1="fixation_front",
                                weight=weight_quiet_eye_phase_1*coefficient_quiet_eye_phase_1, phase=1)

    # Avoid extreme eye and neck angles
    if weight_extreme_angles*coefficient_extreme_angles != 0:
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q",
                                index=[fancy_names_index["ZrotHead"], fancy_names_index["XrotHead"], fancy_names_index["ZrotEyes"], fancy_names_index["XrotEyes"]],
                                weight=weight_extreme_angles*coefficient_extreme_angles, phase=0)
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q",
                                index=[fancy_names_index["ZrotHead"], fancy_names_index["XrotHead"], fancy_names_index["ZrotEyes"], fancy_names_index["XrotEyes"]],
                                weight=weight_extreme_angles*coefficient_extreme_angles, phase=1)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.JOINTS_ACCELERATION_DRIVEN)
    dynamics.add(DynamicsFcn.JOINTS_ACCELERATION_DRIVEN)

    x_bounds, u_bounds = set_bounds(biorbd_model, final_time, num_twists, nb_q, nb_qdot, nb_qddot_joints, fancy_names_index)
    x_init, u_init = set_initial_guesses(nb_q, nb_qdot, num_twists, nb_qddot_joints, fancy_names_index)

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
        n_threads=n_threads,
        assume_phase_dynamics=True,
    )


def get_final_markers(biorbd_model_path, q_final):
    num_markers = 17
    m = biorbd.Model(biorbd_model_path)
    markers_final = np.zeros((3, num_markers, np.shape(q_final)[1]))
    for i in range(np.shape(q_final)[1]):
        for j in range(num_markers):
            markers_final[:, j, i] = m.markers(q_final[:, i])[j].to_array()
    return markers_final


class prepare_iocp:

    def __init__(self, coefficients, biorbd_model_path, solver, markers_xsens):
        self.coefficients = coefficients
        self.solver = solver
        self.markers_xsens = markers_xsens
        self.biorbd_model_path = biorbd_model_path

    def fitness(self, weights):
        global i_inverse
        i_inverse += 1

        ocp = prepare_ocp(weights, self.coefficients, self.biorbd_model_path)

        solver = Solver.IPOPT(show_online_optim=False, show_options=dict(show_bounds=True))
        solver.set_linear_solver("ma57")
        solver.set_maximum_iterations(10000)
        solver.set_convergence_tolerance(1e-6)
        sol = ocp.solve(solver)

        print(
            f"+++++++++++++++++++++++++++ Optimized the {i_inverse}th ocp in the inverse algo +++++++++++++++++++++++++++"
        )
        if sol.status == 0:

            name = biorbd_model_path.split("/")[-1].removesuffix(".bioMod")
            qs, qdots, qddots, time_parameters = get_parameters_from_optim(sol)

            markers_final = get_final_markers(self.biorbd_model_path, qs)
            out_score = np.sum((self.markers_xsens - markers_final) ** 2)

            del sol.ocp
            with open(f"Solutions/{name}-{i_inverse}.pkl", "wb") as f:
                data = {"qs": qs,
                        "qdots": qdots,
                        "qddots": qddots,
                        "time_parameters": time_parameters,
                        "weights": weights,
                        "coefficients": self.coefficients,
                        "out_score": out_score}
                pickle.dump(data, f)

        else:
            out_score = 1000000

        return [out_score]

    def get_nobj(self):
        return 1

    def get_bounds(self):
        return ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1])


def get_parameters_from_optim(sol):
    qs = np.hstack((sol.states[0]["q"][:, :-1], sol.states[1]["q"]))
    qdots = np.hstack((sol.states[0]["qdot"][:, :-1], sol.states[1]["qdot"]))
    qddots = np.hstack((sol.controls[0]["qddot_joints"][:, :-1], sol.controls[1]["qddot_joints"]))
    time_parameters = sol.parameters["time"]
    return qs, qdots, qddots, time_parameters


def main():

    SOLVE_PARETO_FLAG = True # False
    global i_inverse

    move_filename = "a62d4691_0_0-45_796__42__0__eyetracking_metrics.pkl"
    biorbd_model_path = "models/SoMe_42_with_visual_criteria.bioMod"

    # Load the data to track
    with open(move_filename, "rb") as f:
        data = pickle.load(f)
        eye_position = data["eye_position"]
        gaze_orientation = data["gaze_orientation"]
        Xsens_position_no_level_CoM_corrected_rotated_per_move = data["Xsens_position_no_level_CoM_corrected_rotated_per_move"]

    # get joint positions from the xsens model
    num_markers = 17
    num_joints = int(Xsens_position_no_level_CoM_corrected_rotated_per_move.shape[1]/3)
    num_frames = Xsens_position_no_level_CoM_corrected_rotated_per_move.shape[0]
    JCS_xsens = np.zeros((3, num_joints, num_frames))
    for j in range(num_frames):
        for i in range(num_joints):
            JCS_xsens[:, i, j] = Xsens_position_no_level_CoM_corrected_rotated_per_move[j, i*3:(i+1)*3]

    markers_xsens = reorder_markers_xsens(num_markers, num_frames, JCS_xsens, eye_position, gaze_orientation)
    markers_xsens = markers_xsens[:, :, ::2]

    # Define solver options
    solver = Solver.IPOPT(show_online_optim=False, show_options=dict(show_bounds=True))
    solver.set_linear_solver("ma57")
    solver.set_maximum_iterations(10000)
    solver.set_convergence_tolerance(1e-6)

    # Find coefficients of the objective using Pareto
    if SOLVE_PARETO_FLAG:
        coefficients = []
        num_weights = 10
        for i in range(num_weights):
            i_inverse = f"pareto_{i}"
            weights_pareto = [0 for _ in range(num_weights)]
            weights_pareto[i] = 1
            ocp_pareto = prepare_ocp(weights=weights_pareto, coefficients=[1 for _ in range(num_weights)], biorbd_model_path=biorbd_model_path)
            sol_pareto = ocp_pareto.solve(solver)
            coefficients.append(sol_pareto.cost)

            if sol_pareto.status != 0:
                raise RuntimeError("Pareto failed to converge")
            else:
                name = biorbd_model_path.split("/")[-1].removesuffix(".bioMod")
                qs, qdots, qddots, time_parameters = get_parameters_from_optim(sol_pareto)

                markers_final = get_final_markers(biorbd_model_path, qs)
                out_score = np.sum((markers_xsens - markers_final) ** 2)

                del sol_pareto.ocp
                with open(f"Solutions/{name}-{i_inverse}.pkl", "wb") as f:
                    data = {"qs": qs,
                            "qdots": qdots,
                            "qddots": qddots,
                            "time_parameters": time_parameters,
                            "weights": weights_pareto,
                            "coefficients": [1 for _ in range(num_weights)],
                            "out_score": out_score}
                    pickle.dump(data, f)

        print("+++++++++++++++++++++++++++ coefficients generated +++++++++++++++++++++++++++")
        with open("coefficients_pareto.pkl", "wb") as f:
            pickle.dump(coefficients, f)
    else:
        with open("coefficients_pareto.pkl", "rb") as f:
            coefficients = pickle.load(f)

    # Running IOCP
    i_inverse = 0
    iocp = pg.problem(prepare_iocp(coefficients, biorbd_model_path, solver, markers_xsens))
    algo = pg.algorithm(pg.simulated_annealing())
    pop = pg.population(iocp, size=10)

    epsilon = 1e-4
    diff = 10000
    while i_inverse < 1000 and diff > epsilon:
        olf_pop_f = np.min(pop.get_f())
        pop = algo.evolve(pop)
        diff = olf_pop_f - np.min(pop.get_f())
        pop_weights = pop.get_x()[np.argmin(pop.get_f())]

    print("+++++++++++++++++++++++++++ optimal weights found +++++++++++++++++++++++++++")
    print(
        "The optimizaed weight are : \n",
        "weight_qddot_joint =", pop_weights[0] * coefficients[0], "\n",
        "weight_qddot_joint_derivative =", pop_weights[1] * coefficients[1], "\n",
        "weight_time =", pop_weights[2] * coefficients[2], "\n",
        "weight_arms =", pop_weights[3] * coefficients[3], "\n",
        "weight_spotting =", pop_weights[4] * coefficients[4], "\n",
        "weight_self_motion_detection =", pop_weights[5] * coefficients[5], "\n",
        "weight_peripheral_vision =", pop_weights[6] * coefficients[6], "\n",
        "weight_quiet_eye_phase_0 =", pop_weights[7] * coefficients[7], "\n",
        "weight_quiet_eye_phase_1 =", pop_weights[8] * coefficients[8], "\n",
        "weight_extreme_angles =", pop_weights[9] * coefficients[9], "\n",
    )
    with open("optim_weights.pkl", "wb") as f:
        pickle.dump(pop_weights, f)

    # Compare the kinematics
    import biorbd
    ocp_final = prepare_ocp(weights=pop_weights, coefficients=coefficients)
    sol_final = ocp_final.solve(solver)
    q_final, qdot_final, tau_final = sol_final.states["q"], sol_final.states["qdot"], sol_final.controls["tau"]

    markers_final = get_final_markers(biorbd_model_path, q_final)

    ax = fig.add_subplot(111, projection='3d')
    cmap = matplotlib.cm.get_cmap('viridis')
    for j in range(num_markers):
        ax.plot(markers_final[j, :, 0], markers_final[j, :, 1], markers_final[j, :, 2], "-", color=cmap(1/(num_markers-1)), label="Markers")
        ax.plot(markers_xsens[j, :, 0], markers_xsens[j, :, 1], markers_xsens[j, :, 2], ".", color=cmap(1/(num_markers-1)), label="Tracked reference from xsens")
    ax.legend()
    ax.set_title(
        "Marker trajectory of the reference problem and the final solution generated with the optimal solutions."
    )
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    plt.show()

    # Animate the optimal solution
    b = bioviz.Viz(biorbd_model_path)
    b.load_movement(num_markers)
    b.load_experimental_markers(markers_xsens)
    b.exec()


if __name__ == "__main__":
    main()
