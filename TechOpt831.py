
"""
The goal of this program is to optimize the movement to achieve a 831< or a 851<.
Phase 0 : Twist
Phase 1: Pike
Phase 2 : Rotate in somersault in pike position
Phase 3 : Kick out
Phase 4: Half twist
Phase 5 : preparation for landing
"""

import numpy as np
import pickle
import biorbd_casadi as biorbd
import casadi as cas
import IPython
import time
import sys
sys.path.append("/home/charbie/Documents/Programmation/BiorbdOptim")
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
    CostType,
    BiorbdModel,
    Shooting,
    SolutionIntegrator,
    ConstraintList,
    ConstraintFcn,
    MultinodeConstraintList,
    MultinodeConstraintFcn,
)

from TechOpt42 import custom_trampoline_bed_in_peripheral_vision

def prepare_ocp(
        biorbd_model_path: str,
        n_shooting: tuple,
        num_twists: int,
        n_threads: int,
        ode_solver: OdeSolver = OdeSolver.RK4(),
        WITH_VISUAL_CRITERIA: bool = False
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

    final_time = 1.87
    biorbd_model = (
        BiorbdModel(biorbd_model_path),
        BiorbdModel(biorbd_model_path),
        BiorbdModel(biorbd_model_path),
        BiorbdModel(biorbd_model_path),
        BiorbdModel(biorbd_model_path),
        BiorbdModel(biorbd_model_path),
    )

    nb_q = biorbd_model[0].nb_q
    nb_qdot = biorbd_model[0].nb_qdot
    nb_qddot_joints = nb_q - biorbd_model[0].nb_root

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
        ZrotRightLowerArm = 8
        XrotRightLowerArm = 9
        ZrotLeftUpperArm = 10
        YrotLeftUpperArm = 11
        ZrotLeftLowerArm = 12
        XrotLeftLowerArm = 13
        XrotLegs = 14
        YrotLegs = 15
        vX = 0
        vY = 1
        vZ = 2
        vXrot = 3
        vYrot = 4
        vZrot = 5
        vZrotRightUpperArm = 6
        vYrotRightUpperArm = 7
        vZrotRightLowerArm = 8
        vYrotRightLowerArm = 9
        vZrotLeftUpperArm = 10
        vYrotLeftUpperArm = 11
        vZrotLeftLowerArm = 12
        vYrotLeftLowerArm = 13
        vXrotLegs = 14
        vYrotLegs = 15
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
        ZrotRightLowerArm = 12
        XrotRightLowerArm = 13
        ZrotLeftUpperArm = 14
        YrotLeftUpperArm = 15
        ZrotLeftLowerArm = 16
        XrotLeftLowerArm = 17
        XrotLegs = 18
        YrotLegs = 19
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
        vZrotRightLowerArm = 12
        vYrotRightLowerArm = 13
        vZrotLeftUpperArm = 14
        vYrotLeftUpperArm = 15
        vZrotLeftLowerArm = 16
        vYrotLeftLowerArm = 17
        vXrotLegs = 18
        vYrotLegs = 19

    # Add objective functions
    objective_functions = ObjectiveList()

    for i in range(6):
        # Min controls
        objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="qddot_joints", node=Node.ALL_SHOOTING, weight=1, quadratic=True, phase=i
        )
        # Min control derivative
        objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="qddot_joints", node=Node.ALL_SHOOTING, weight=1, quadratic=True, phase=i, derivative=True,
        )

    # Min/Max time
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_TIME, min_bound=0.05, max_bound=final_time / 2, weight=1, quadratic=True, phase=0
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_TIME, min_bound=0.05, max_bound=final_time / 2, weight=100, quadratic=True, phase=1
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_TIME, min_bound=0.05, max_bound=final_time / 2, weight=-0.01, quadratic=True, phase=2
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_TIME, min_bound=0.05, max_bound=final_time / 2, weight=100, quadratic=True, phase=3
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_TIME, min_bound=0.05, max_bound=final_time / 2, weight=-0.01, quadratic=True, phase=4
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_TIME, min_bound=0.05, max_bound=final_time / 2, weight=-0.01, quadratic=True, phase=5
    )


    # Aim to put the hands on the lower legs to grab the pike position
    objective_functions.add(
        ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS,
        node=Node.END,
        first_marker="MidMainG",
        second_marker="CibleMainG",
        weight=1,
        quadratic=True,
        phase=1,
    )
    objective_functions.add(
         ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS,
         node=Node.END,
         first_marker="MidMainD",
         second_marker="CibleMainD",
         weight=1,
         quadratic=True,
         phase=1,
     )

    # Aligning with the FIG regulations
    arm_dofs = [ZrotRightUpperArm, YrotRightUpperArm, ZrotRightLowerArm, XrotRightLowerArm, ZrotLeftUpperArm, YrotLeftUpperArm, ZrotLeftLowerArm, XrotLeftLowerArm]
    shoulder_dofs = [ZrotRightUpperArm, YrotRightUpperArm, ZrotLeftUpperArm, YrotLeftUpperArm]
    elbow_dofs = [ZrotRightLowerArm, XrotRightLowerArm, ZrotLeftLowerArm, XrotLeftLowerArm]

    objective_functions.add(
         ObjectiveFcn.Lagrange.MINIMIZE_STATE,
         key="q",
         node=Node.ALL_SHOOTING,
         index=elbow_dofs,
         weight=50000,
         quadratic=True,
         phase=0,
    )
    objective_functions.add(
         ObjectiveFcn.Lagrange.MINIMIZE_STATE,
         key="q",
         node=Node.ALL_SHOOTING,
         index=shoulder_dofs,
         weight=50000,
         quadratic=True,
         phase=2,
    )
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_STATE,
        key="q",
        node=Node.ALL_SHOOTING,
        index=arm_dofs,
        weight=50000,
        quadratic=True,
        phase=4,
    )
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_STATE,
        key="q",
        node=Node.ALL_SHOOTING,
        index=elbow_dofs,
        weight=50000,
        quadratic=True,
        phase=5,
    )

    # Minimize wobbling
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, key="q", node=Node.ALL, index=[Yrot], weight=100, quadratic=True, phase=2
    )

    # Land safely (without tilt)
    objective_functions.add(
         ObjectiveFcn.Mayer.MINIMIZE_STATE,
         key="q",
         node=Node.END,
         index=[Yrot],
         weight=1000,
         quadratic=True,
         phase=5,
    )

    if WITH_VISUAL_CRITERIA:

        # Spotting
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_SEGMENT_VELOCITY, segment="Head", weight=10, quadratic=True, phase=0)
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_SEGMENT_VELOCITY, segment="Head", weight=10, quadratic=True, phase=5)

        # Self-motion detection
        for i in range(6):
            objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key='qdot', index=[ZrotEyes, XrotEyes], weight=1, quadratic=True, phase=i)

        # Keeping the trampoline bed in the peripheral vision
        objective_functions.add(custom_trampoline_bed_in_peripheral_vision, custom_type=ObjectiveFcn.Lagrange, weight=100, quadratic=True, phase=0)
        objective_functions.add(custom_trampoline_bed_in_peripheral_vision, custom_type=ObjectiveFcn.Lagrange, weight=100, quadratic=True, phase=3)
        objective_functions.add(custom_trampoline_bed_in_peripheral_vision, custom_type=ObjectiveFcn.Lagrange, weight=100, quadratic=True, phase=4)
        objective_functions.add(custom_trampoline_bed_in_peripheral_vision, custom_type=ObjectiveFcn.Lagrange, weight=100, quadratic=True, phase=5)

        # Quiet eye
        objective_functions.add(ObjectiveFcn.Lagrange.TRACK_VECTOR_ORIENTATIONS_FROM_MARKERS,
                                vector_0_marker_0="eyes_vect_start",
                                vector_0_marker_1="eyes_vect_end",
                                vector_1_marker_0="eyes_vect_start",
                                vector_1_marker_1="fixation_front",
                                weight=1, quadratic=True, phase=0)
        objective_functions.add(ObjectiveFcn.Lagrange.TRACK_VECTOR_ORIENTATIONS_FROM_MARKERS,
                                vector_0_marker_0="eyes_vect_start",
                                vector_0_marker_1="eyes_vect_end",
                                vector_1_marker_0="eyes_vect_start",
                                vector_1_marker_1="fixation_front",
                                weight=1000, quadratic=True, phase=5)

        # Avoid extreme eye and neck angles
        for i in range(6):
            objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", index=[ZrotHead, XrotHead], weight=100, quadratic=True, phase=i)
            objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", index=[ZrotEyes, XrotEyes], weight=10, quadratic=True, phase=i)

    constraints = ConstraintList()
    constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS,
        node=Node.ALL_SHOOTING,
        min_bound=-0.05,
        max_bound=0.05,
        first_marker="MidMainG",
        second_marker="CibleMainG",
        phase=2,
    )
    constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS,
        node=Node.ALL_SHOOTING,
        min_bound=-0.05,
        max_bound=0.05,
        first_marker="MidMainD",
        second_marker="CibleMainD",
        phase=2,
    )

    multinode_constraints = MultinodeConstraintList()
    multinode_constraints.add(
        MultinodeConstraintFcn.TRACK_TOTAL_TIME,
        nodes_phase=(0, 1, 2, 3, 4, 5),
        nodes=(Node.END, Node.END, Node.END, Node.END, Node.END, Node.END),
        min_bound=final_time - 0.01,
        max_bound=final_time + 0.01,
    )

    # Dynamics
    dynamics = DynamicsList()
    for i in range(6):
        dynamics.add(DynamicsFcn.JOINTS_ACCELERATION_DRIVEN)

    qddot_joints_min, qddot_joints_max, qddot_joints_init = -1000, 1000, 0
    u_bounds = BoundsList()
    u_init = InitialGuessList()
    for i in range(6):
        u_bounds.add(
            "qddot_joints",
            min_bound=[qddot_joints_min] * nb_qddot_joints,
            max_bound=[qddot_joints_max] * nb_qddot_joints,
            phase=i,
        )
        u_init.add("qddot_joints", initial_guess=[qddot_joints_init] * nb_qddot_joints, phase=i)

    # Path constraint
    x_bounds = BoundsList()
    q_bounds_0_min = np.array(biorbd_model[0].bounds_from_ranges("q").min)
    q_bounds_0_max = np.array(biorbd_model[0].bounds_from_ranges("q").max)
    q_bounds_1_min = np.array(biorbd_model[1].bounds_from_ranges("q").min)
    q_bounds_1_max = np.array(biorbd_model[1].bounds_from_ranges("q").max)
    q_bounds_2_min = np.array(biorbd_model[2].bounds_from_ranges("q").min)
    q_bounds_2_max = np.array(biorbd_model[2].bounds_from_ranges("q").max)
    q_bounds_3_min = np.array(biorbd_model[3].bounds_from_ranges("q").min)
    q_bounds_3_max = np.array(biorbd_model[3].bounds_from_ranges("q").max)
    q_bounds_4_min = np.array(biorbd_model[4].bounds_from_ranges("q").min)
    q_bounds_4_max = np.array(biorbd_model[4].bounds_from_ranges("q").max)
    q_bounds_5_min = np.array(biorbd_model[5].bounds_from_ranges("q").min)
    q_bounds_5_max = np.array(biorbd_model[5].bounds_from_ranges("q").max)
    qdot_bounds_0_min = np.array(biorbd_model[0].bounds_from_ranges("qdot").min)
    qdot_bounds_0_max = np.array(biorbd_model[0].bounds_from_ranges("qdot").max)
    qdot_bounds_1_min = np.array(biorbd_model[1].bounds_from_ranges("qdot").min)
    qdot_bounds_1_max = np.array(biorbd_model[1].bounds_from_ranges("qdot").max)
    qdot_bounds_2_min = np.array(biorbd_model[2].bounds_from_ranges("qdot").min)
    qdot_bounds_2_max = np.array(biorbd_model[2].bounds_from_ranges("qdot").max)
    qdot_bounds_3_min = np.array(biorbd_model[3].bounds_from_ranges("qdot").min)
    qdot_bounds_3_max = np.array(biorbd_model[3].bounds_from_ranges("qdot").max)
    qdot_bounds_4_min = np.array(biorbd_model[4].bounds_from_ranges("qdot").min)
    qdot_bounds_4_max = np.array(biorbd_model[4].bounds_from_ranges("qdot").max)
    qdot_bounds_5_min = np.array(biorbd_model[5].bounds_from_ranges("qdot").min)
    qdot_bounds_5_max = np.array(biorbd_model[5].bounds_from_ranges("qdot").max)

    # For lisibility
    START, MIDDLE, END = 0, 1, 2

    # ------------------------------- Phase 0 : twist ------------------------------- #
    zmax = 9.81 / 8 * final_time**2 + 1

    # Pelvis translations
    q_bounds_0_min[X, :] = -0.25
    q_bounds_0_max[X, :] = 0.25
    q_bounds_0_min[Y, :] = -0.5
    q_bounds_0_max[Y, :] = 0.5
    q_bounds_0_min[: Z + 1, START] = 0
    q_bounds_0_max[: Z + 1, START] = 0
    q_bounds_0_min[Z, MIDDLE:] = 0
    q_bounds_0_max[Z, MIDDLE:] = zmax

    # Somersault
    q_bounds_0_min[Xrot, START] = 0
    q_bounds_0_max[Xrot, START] = 0
    q_bounds_0_min[Xrot, MIDDLE:] = -2 * np.pi
    q_bounds_0_max[Xrot, MIDDLE:] = 0.5
    # Tilt
    q_bounds_0_min[Yrot, START] = 0
    q_bounds_0_max[Yrot, START] = 0
    q_bounds_0_min[Yrot, MIDDLE:] = -np.pi / 4  # avoid gimbal lock
    q_bounds_0_max[Yrot, MIDDLE:] = np.pi / 4
    # Twist
    q_bounds_0_min[Zrot, START] = 0
    q_bounds_0_max[Zrot, START] = 0
    q_bounds_0_min[Zrot, MIDDLE] = -0.5
    q_bounds_0_max[Zrot, MIDDLE] = 2 * np.pi * num_twists + np.pi
    q_bounds_0_min[Zrot, END] = 2 * np.pi * num_twists
    q_bounds_0_max[Zrot, END] = 2 * np.pi * num_twists + np.pi

    # Right arm
    q_bounds_0_min[YrotRightUpperArm, START] = 2.9
    q_bounds_0_max[YrotRightUpperArm, START] = 2.9
    q_bounds_0_min[ZrotRightUpperArm, START] = 0
    q_bounds_0_max[ZrotRightUpperArm, START] = 0
    # Left arm
    q_bounds_0_min[YrotLeftUpperArm, START] = -2.9
    q_bounds_0_max[YrotLeftUpperArm, START] = -2.9
    q_bounds_0_min[ZrotLeftUpperArm, START] = 0
    q_bounds_0_max[ZrotLeftUpperArm, START] = 0

    # Right elbow
    q_bounds_0_min[ZrotRightLowerArm : XrotRightLowerArm + 1, START] = 0
    q_bounds_0_max[ZrotRightLowerArm : XrotRightLowerArm + 1, START] = 0
    # Left elbow
    q_bounds_0_min[ZrotLeftLowerArm : XrotLeftLowerArm + 1, START] = 0
    q_bounds_0_max[ZrotLeftLowerArm : XrotLeftLowerArm + 1, START] = 0

    # Hip flexion
    q_bounds_0_min[XrotLegs, START] = 0
    q_bounds_0_max[XrotLegs, START] = 0
    q_bounds_0_min[XrotLegs, MIDDLE:] = -0.2
    q_bounds_0_max[XrotLegs, MIDDLE:] = 0.2
    # Hip sides
    q_bounds_0_min[YrotLegs, START] = 0
    q_bounds_0_max[YrotLegs, START] = 0

    # Head and eyes
    if WITH_VISUAL_CRITERIA:
        q_bounds_0_min[ZrotHead, START] = -0.1
        q_bounds_0_max[ZrotHead, START] = 0.1
        q_bounds_0_min[XrotHead, START] = -0.1
        q_bounds_0_max[XrotHead, START] = 0.1
        q_bounds_0_min[ZrotEyes, START] = -0.1
        q_bounds_0_max[ZrotEyes, START] = 0.1
        q_bounds_0_min[XrotEyes, START] = np.pi / 8 - 0.1
        q_bounds_0_max[XrotEyes, START] = np.pi / 8 + 0.1

    vzinit = 9.81 / 2 * final_time

    # Shift the initial vertical speed at the CoM
    CoM_Q_sym = cas.MX.sym("CoM", nb_q)
    CoM_Q_init = q_bounds_0_min[:nb_q, START]
    CoM_Q_func = cas.Function("CoM_Q_func", [CoM_Q_sym], [biorbd_model[0].center_of_mass(CoM_Q_sym)])
    bassin_Q_func = cas.Function(
        "bassin_Q_func", [CoM_Q_sym], [biorbd_model[0].homogeneous_matrices_in_global(CoM_Q_sym, 0).to_mx()]
    )

    r = np.array(CoM_Q_func(CoM_Q_init)).reshape(1, 3) - np.array(bassin_Q_func(CoM_Q_init))[-1, :3]

    # Pelis translation velocities
    qdot_bounds_0_min[vX : vY + 1, :] = -10
    qdot_bounds_0_max[vX : vY + 1, :] = 10
    qdot_bounds_0_min[vX : vY + 1, START] = -0.5
    qdot_bounds_0_max[vX : vY + 1, START] = 0.5
    qdot_bounds_0_min[vZ, :] = -100
    qdot_bounds_0_max[vZ, :] = 100
    qdot_bounds_0_min[vZ, START] = vzinit - 0.5
    qdot_bounds_0_max[vZ, START] = vzinit + 0.5

    # Somersault
    qdot_bounds_0_min[vXrot, :] = -20
    qdot_bounds_0_max[vXrot, :] = -0.5
    # Tile
    qdot_bounds_0_min[vYrot, :] = -100
    qdot_bounds_0_max[vYrot, :] = 100
    qdot_bounds_0_min[vYrot, START] = 0
    qdot_bounds_0_max[vYrot, START] = 0
    # Twist
    qdot_bounds_0_min[vZrot, :] = -100
    qdot_bounds_0_max[vZrot, :] = 100
    qdot_bounds_0_min[vZrot, START] = 0
    qdot_bounds_0_max[vZrot, START] = 0

    min_bound_trans_velocity = (
        qdot_bounds_0_min[vX : vZ + 1, START] + np.cross(r, qdot_bounds_0_min[vXrot : vZrot + 1, START])
    )[0]
    max_bound_trans_velocity = (
        qdot_bounds_0_max[vX : vZ + 1, START] + np.cross(r, qdot_bounds_0_max[vXrot : vZrot + 1, START])
    )[0]
    qdot_bounds_0_min[vX : vZ + 1, START] = (
        min(max_bound_trans_velocity[0], min_bound_trans_velocity[0]),
        min(max_bound_trans_velocity[1], min_bound_trans_velocity[1]),
        min(max_bound_trans_velocity[2], min_bound_trans_velocity[2]),
    )
    qdot_bounds_0_max[vX : vZ + 1, START] = (
        max(max_bound_trans_velocity[0], min_bound_trans_velocity[0]),
        max(max_bound_trans_velocity[1], min_bound_trans_velocity[1]),
        max(max_bound_trans_velocity[2], min_bound_trans_velocity[2]),
    )

    # Right arm
    qdot_bounds_0_min[vZrotRightUpperArm : vYrotRightUpperArm + 1, :] = -100
    qdot_bounds_0_max[vZrotRightUpperArm : vYrotRightUpperArm + 1, :] = 100
    qdot_bounds_0_min[vZrotRightUpperArm : vYrotRightUpperArm + 1, START] = 0
    qdot_bounds_0_max[vZrotRightUpperArm : vYrotRightUpperArm + 1, START] = 0
    # Left arm
    qdot_bounds_0_min[vZrotLeftUpperArm : vYrotLeftUpperArm + 1, :] = -100
    qdot_bounds_0_max[vZrotLeftUpperArm : vYrotLeftUpperArm + 1, :] = 100
    qdot_bounds_0_min[vZrotLeftUpperArm : vYrotLeftUpperArm + 1, START] = 0
    qdot_bounds_0_max[vZrotLeftUpperArm : vYrotLeftUpperArm + 1, START] = 0

    # Right elbow
    qdot_bounds_0_min[vZrotRightLowerArm : vYrotRightLowerArm + 1, :] = -100
    qdot_bounds_0_max[vZrotRightLowerArm : vYrotRightLowerArm + 1, :] = 100
    qdot_bounds_0_min[vZrotRightLowerArm : vYrotRightLowerArm + 1, START] = 0
    qdot_bounds_0_max[vZrotRightLowerArm : vYrotRightLowerArm + 1, START] = 0
    # Left elbow
    qdot_bounds_0_min[vZrotRightLowerArm : vYrotLeftLowerArm + 1, :] = -100
    qdot_bounds_0_max[vZrotRightLowerArm : vYrotLeftLowerArm + 1, :] = 100
    qdot_bounds_0_min[vZrotLeftLowerArm : vYrotLeftLowerArm + 1, START] = 0
    qdot_bounds_0_max[vZrotLeftLowerArm : vYrotLeftLowerArm + 1, START] = 0

    # Hip flexion
    qdot_bounds_0_min[vXrotLegs, :] = -100
    qdot_bounds_0_max[vXrotLegs, :] = 100
    qdot_bounds_0_min[vXrotLegs, START] = 0
    qdot_bounds_0_max[vXrotLegs, START] = 0
    # Hip sides
    qdot_bounds_0_min[vYrotLegs, :] = -100
    qdot_bounds_0_max[vYrotLegs, :] = 100
    qdot_bounds_0_min[vYrotLegs, START] = 0
    qdot_bounds_0_max[vYrotLegs, START] = 0

    # ------------------------------- Phase 1 : piking ------------------------------- #
    # Pelvis translations
    q_bounds_1_min[X, :] = -0.25
    q_bounds_1_max[X, :] = 0.25
    q_bounds_1_min[Y, :] = -0.5
    q_bounds_1_max[Y, :] = 0.5
    q_bounds_1_min[Z, :] = 0
    q_bounds_1_max[Z, :] = zmax

    # Somersault
    q_bounds_1_min[Xrot, :] = -5 / 4 * np.pi
    q_bounds_1_max[Xrot, :] = 0
    q_bounds_1_min[Xrot, END] = -5 / 4 * np.pi
    q_bounds_1_max[Xrot, END] = -np.pi / 2
    # Tilt
    q_bounds_1_min[Yrot, :] = -np.pi / 4
    q_bounds_1_max[Yrot, :] = np.pi / 4
    # Twist
    q_bounds_1_min[Zrot, :] = 2 * np.pi * num_twists
    q_bounds_1_max[Zrot, :] = 2 * np.pi * num_twists + np.pi + 0.5
    q_bounds_1_min[Zrot, END] = 2 * np.pi * num_twists + np.pi - 0.1
    q_bounds_1_max[Zrot, END] = 2 * np.pi * num_twists + np.pi + 0.1

    # Hips flexion
    q_bounds_1_min[XrotLegs, START] = -0.2
    q_bounds_1_max[XrotLegs, START] = 0.2
    q_bounds_1_min[XrotLegs, MIDDLE] = -2.4 - 0.2
    q_bounds_1_max[XrotLegs, MIDDLE] = 0.2
    q_bounds_1_min[XrotLegs, END] = -2.4 - 0.2
    q_bounds_1_max[XrotLegs, END] = -2.4 + 0.2
    # Hips sides
    q_bounds_1_min[YrotLegs, END] = -0.1
    q_bounds_1_max[YrotLegs, END] = 0.1

    # Translations velocities
    qdot_bounds_1_min[vX : vY + 1, :] = -10
    qdot_bounds_1_max[vX : vY + 1, :] = 10
    qdot_bounds_1_min[vZ, :] = -100
    qdot_bounds_1_max[vZ, :] = 100
    # Somersault
    qdot_bounds_1_min[vXrot, :] = -100
    qdot_bounds_1_max[vXrot, :] = 100
    # Tilt
    qdot_bounds_1_min[vYrot, :] = -100
    qdot_bounds_1_max[vYrot, :] = 100
    # Twist
    qdot_bounds_1_min[vZrot, :] = -100
    qdot_bounds_1_max[vZrot, :] = 100

    # Right arm
    qdot_bounds_1_min[vZrotRightUpperArm : vYrotRightUpperArm + 1, :] = -100
    qdot_bounds_1_max[vZrotRightUpperArm : vYrotRightUpperArm + 1, :] = 100
    # Left elbow
    qdot_bounds_1_min[vZrotLeftUpperArm : vYrotLeftUpperArm + 1, :] = -100
    qdot_bounds_1_max[vZrotLeftUpperArm : vYrotLeftUpperArm + 1, :] = 100

    # Right elbow
    qdot_bounds_1_min[vZrotRightLowerArm : vYrotRightLowerArm + 1, :] = -100
    qdot_bounds_1_max[vZrotRightLowerArm : vYrotRightLowerArm + 1, :] = 100
    # Left elbow
    qdot_bounds_1_min[vZrotRightLowerArm : vYrotLeftLowerArm + 1, :] = -100
    qdot_bounds_1_max[vZrotRightLowerArm : vYrotLeftLowerArm + 1, :] = 100

    # Hip flexion
    qdot_bounds_1_min[vXrotLegs, :] = -100
    qdot_bounds_1_max[vXrotLegs, :] = 100
    # Hip sides
    qdot_bounds_1_min[vYrotLegs, :] = -100
    qdot_bounds_1_max[vYrotLegs, :] = 100

    # ------------------------------- Phase 2 : somersault in pike ------------------------------- #

    # Pelvis translations
    q_bounds_2_min[X, :] = -0.25
    q_bounds_2_max[X, :] = 0.25
    q_bounds_2_min[Y, :] = -0.5
    q_bounds_2_max[Y, :] = 0.5
    q_bounds_2_min[Z, :] = 0
    q_bounds_2_max[Z, :] = zmax  # beaucoup plus que necessaire, juste pour que la parabole fonctionne

    # Somersault
    q_bounds_2_min[Xrot, :] = -3 * np.pi
    q_bounds_2_max[Xrot, :] = - np.pi
    # Tilt
    q_bounds_2_min[Yrot, :] = -np.pi / 8
    q_bounds_2_max[Yrot, :] = np.pi / 8
    # Twist
    q_bounds_2_min[Zrot, :] = 2 * np.pi * num_twists + np.pi - np.pi / 8
    q_bounds_2_max[Zrot, :] = 2 * np.pi * num_twists + np.pi + np.pi / 8

    # Hips flexion
    q_bounds_2_min[XrotLegs, :] = -2.4 - 0.2
    q_bounds_2_max[XrotLegs, :] = -2.4 + 0.2
    # Hips sides
    q_bounds_2_min[YrotLegs, :] = -0.1
    q_bounds_2_max[YrotLegs, :] = 0.1

    # Translations velocities
    qdot_bounds_2_min[vX : vY + 1, :] = -10
    qdot_bounds_2_max[vX : vY + 1, :] = 10
    qdot_bounds_2_min[vZ, :] = -100
    qdot_bounds_2_max[vZ, :] = 100

    # Somersault
    qdot_bounds_2_min[vXrot, :] = -100
    qdot_bounds_2_max[vXrot, :] = 100
    # Tilt
    qdot_bounds_2_min[vYrot, :] = -100
    qdot_bounds_2_max[vYrot, :] = 100
    # Twist
    qdot_bounds_2_min[vZrot, :] = -100
    qdot_bounds_2_max[vZrot, :] = 100

    # Right arm
    qdot_bounds_2_min[vZrotRightUpperArm : vYrotRightUpperArm + 1, :] = -100
    qdot_bounds_2_max[vZrotRightUpperArm : vYrotRightUpperArm + 1, :] = 100
    # Left arm
    qdot_bounds_2_min[vZrotLeftUpperArm : vYrotLeftUpperArm + 1, :] = -100
    qdot_bounds_2_max[vZrotLeftUpperArm : vYrotLeftUpperArm + 1, :] = 100

    # Right elbow
    qdot_bounds_2_min[vZrotRightLowerArm : vYrotRightLowerArm + 1, :] = -100
    qdot_bounds_2_max[vZrotRightLowerArm : vYrotRightLowerArm + 1, :] = 100
    # Left elbow
    qdot_bounds_2_min[vZrotRightLowerArm : vYrotLeftLowerArm + 1, :] = -100
    qdot_bounds_2_max[vZrotRightLowerArm : vYrotLeftLowerArm + 1, :] = 100

    # Hip flexion
    qdot_bounds_2_min[vXrotLegs, :] = -100
    qdot_bounds_2_max[vXrotLegs, :] = 100
    # Hip sides
    qdot_bounds_2_min[vYrotLegs, :] = -100
    qdot_bounds_2_max[vYrotLegs, :] = 100

    # ------------------------------- Phase 3 : kick out + 1/2 twist ------------------------------- #

    # Pelvis translations
    q_bounds_3_min[X, :] = -0.25
    q_bounds_3_max[X, :] = 0.25
    q_bounds_3_min[Y, :] = -0.5
    q_bounds_3_max[Y, :] = 0.5
    q_bounds_3_min[Z, :] = 0
    q_bounds_3_max[Z, :] = zmax

    # Somersault
    q_bounds_3_min[Xrot, START] = -3 * np.pi
    q_bounds_3_max[Xrot, START] = -2 * np.pi
    q_bounds_3_min[Xrot, MIDDLE] = -7/2 * np.pi
    q_bounds_3_max[Xrot, MIDDLE] = -2 * np.pi
    q_bounds_3_min[Xrot, END] = -7/2 * np.pi + 0.2 - 0.2
    q_bounds_3_max[Xrot, END] = -7/2 * np.pi + 0.2 + 0.2
    # Tilt
    q_bounds_3_min[Yrot, :] = -np.pi / 4
    q_bounds_3_max[Yrot, :] = np.pi / 4
    # Twist
    q_bounds_3_min[Zrot, START] = 2 * np.pi * num_twists + np.pi - np.pi / 4
    q_bounds_3_max[Zrot, START] = 2 * np.pi * num_twists + np.pi + np.pi / 4
    q_bounds_3_min[Zrot, MIDDLE] = 2 * np.pi * num_twists + np.pi - np.pi / 4
    q_bounds_3_max[Zrot, MIDDLE] = 2 * np.pi * num_twists + np.pi + np.pi / 4
    q_bounds_3_min[Zrot, END] = 2 * np.pi * num_twists + np.pi + np.pi / 8
    q_bounds_3_max[Zrot, END] = 2 * np.pi * num_twists + np.pi + np.pi / 4

    # Hips flexion
    q_bounds_3_min[XrotLegs, START] = -2.4 - 0.2
    q_bounds_3_max[XrotLegs, START] = -2.4 + 0.2
    q_bounds_3_min[XrotLegs, MIDDLE] = -2.4 - 0.2
    q_bounds_3_max[XrotLegs, MIDDLE] = 0.35
    q_bounds_3_min[XrotLegs, END] = -0.35
    q_bounds_3_max[XrotLegs, END] = 0.35

    # Translations velocities
    qdot_bounds_3_min[vX : vY + 1, :] = -10
    qdot_bounds_3_max[vX : vY + 1, :] = 10
    qdot_bounds_3_min[vZ, :] = -100
    qdot_bounds_3_max[vZ, :] = 100

    # Somersault
    qdot_bounds_3_min[vXrot, :] = -100
    qdot_bounds_3_max[vXrot, :] = 100
    # Tilt
    qdot_bounds_3_min[vYrot, :] = -100
    qdot_bounds_3_max[vYrot, :] = 100
    # Twist
    qdot_bounds_3_min[vZrot, :] = -100
    qdot_bounds_3_max[vZrot, :] = 100

    # Right arm
    qdot_bounds_3_min[vZrotRightUpperArm : vYrotRightUpperArm + 1, :] = -100
    qdot_bounds_3_max[vZrotRightUpperArm : vYrotRightUpperArm + 1, :] = 100
    # Left arm
    qdot_bounds_3_min[vZrotLeftUpperArm : vYrotLeftUpperArm + 1, :] = -100
    qdot_bounds_3_max[vZrotLeftUpperArm : vYrotLeftUpperArm + 1, :] = 100

    # Right elbow
    qdot_bounds_3_min[vZrotRightLowerArm : vYrotRightLowerArm + 1, :] = -100
    qdot_bounds_3_max[vZrotRightLowerArm : vYrotRightLowerArm + 1, :] = 100
    # Left elbow
    qdot_bounds_3_min[vZrotRightLowerArm : vYrotLeftLowerArm + 1, :] = -100
    qdot_bounds_3_max[vZrotRightLowerArm : vYrotLeftLowerArm + 1, :] = 100

    # Hip flexion
    qdot_bounds_3_min[vXrotLegs, :] = -100
    qdot_bounds_3_max[vXrotLegs, :] = 100
    # Hip sides
    qdot_bounds_3_min[vYrotLegs, :] = -100
    qdot_bounds_3_max[vYrotLegs, :] = 100


    # ------------------------------- Phase 4 : 1/2 twist ------------------------------- #

    # Pelvis translations
    q_bounds_4_min[X, :] = -0.25
    q_bounds_4_max[X, :] = 0.25
    q_bounds_4_min[Y, :] = -0.5
    q_bounds_4_max[Y, :] = 0.5
    q_bounds_4_min[Z, :] = 0
    q_bounds_4_max[Z, :] = zmax

    # Somersault
    q_bounds_4_min[Xrot, :] = -7/2 * np.pi
    q_bounds_4_max[Xrot, :] = -2 * np.pi
    q_bounds_4_min[Xrot, END] = -7/2 * np.pi + 0.2 - 0.2
    q_bounds_4_max[Xrot, END] = -7/2 * np.pi + 0.2 + 0.2
    # Tilt
    q_bounds_4_min[Yrot, :] = -np.pi / 4
    q_bounds_4_max[Yrot, :] = np.pi / 4
    q_bounds_4_min[Yrot, END] = -np.pi / 8
    q_bounds_4_max[Yrot, END] = np.pi / 8
    # Twist
    q_bounds_4_min[Zrot, START] = 2 * np.pi * num_twists + np.pi + np.pi / 8
    q_bounds_4_max[Zrot, START] = 2 * np.pi * num_twists + np.pi + np.pi / 4
    q_bounds_4_min[Zrot, MIDDLE] = 2 * np.pi * num_twists + np.pi + np.pi / 8
    q_bounds_4_max[Zrot, MIDDLE] = 2 * np.pi * num_twists + 2 * np.pi + 0.01
    q_bounds_4_min[Zrot, END] = 2 * np.pi * num_twists + 2 * np.pi + 0.01
    q_bounds_4_max[Zrot, END] = 2 * np.pi * num_twists + 2 * np.pi + 0.01

    # Right arm
    q_bounds_4_min[YrotRightUpperArm, END] = 0
    q_bounds_4_max[YrotRightUpperArm, END] = np.pi/8
    # Left arm
    q_bounds_4_min[YrotLeftUpperArm, END] = -np.pi/8
    q_bounds_4_max[YrotLeftUpperArm, END] = 0

    # Right elbow
    q_bounds_4_min[ZrotRightLowerArm : XrotRightLowerArm + 1, END] = -0.1
    q_bounds_4_max[ZrotRightLowerArm : XrotRightLowerArm + 1, END] = 0.1
    # Left elbow
    q_bounds_4_min[ZrotLeftLowerArm : XrotLeftLowerArm + 1, END] = -0.1
    q_bounds_4_max[ZrotLeftLowerArm : XrotLeftLowerArm + 1, END] = 0.1

    # Hips flexion
    q_bounds_4_min[XrotLegs, :] = -0.35
    q_bounds_4_max[XrotLegs, :] = 0.35

    # Translations velocities
    qdot_bounds_4_min[vX : vY + 1, :] = -10
    qdot_bounds_4_max[vX : vY + 1, :] = 10
    qdot_bounds_4_min[vZ, :] = -100
    qdot_bounds_4_max[vZ, :] = 100

    # Somersault
    qdot_bounds_4_min[vXrot, :] = -100
    qdot_bounds_4_max[vXrot, :] = 100
    # Tilt
    qdot_bounds_4_min[vYrot, :] = -100
    qdot_bounds_4_max[vYrot, :] = 100
    # Twist
    qdot_bounds_4_min[vZrot, :] = -100
    qdot_bounds_4_max[vZrot, :] = 100

    # Right arm
    qdot_bounds_4_min[vZrotRightUpperArm : vYrotRightUpperArm + 1, :] = -100
    qdot_bounds_4_max[vZrotRightUpperArm : vYrotRightUpperArm + 1, :] = 100
    # Left arm
    qdot_bounds_4_min[vZrotLeftUpperArm : vYrotLeftUpperArm + 1, :] = -100
    qdot_bounds_4_max[vZrotLeftUpperArm : vYrotLeftUpperArm + 1, :] = 100

    # Right elbow
    qdot_bounds_4_min[vZrotRightLowerArm : vYrotRightLowerArm + 1, :] = -100
    qdot_bounds_4_max[vZrotRightLowerArm : vYrotRightLowerArm + 1, :] = 100
    # Left elbow
    qdot_bounds_4_min[vZrotRightLowerArm : vYrotLeftLowerArm + 1, :] = -100
    qdot_bounds_4_max[vZrotRightLowerArm : vYrotLeftLowerArm + 1, :] = 100

    # Hip flexion
    qdot_bounds_4_min[vXrotLegs, :] = -100
    qdot_bounds_4_max[vXrotLegs, :] = 100
    # Hip sides
    qdot_bounds_4_min[vYrotLegs, :] = -100
    qdot_bounds_4_max[vYrotLegs, :] = 100

    # ------------------------------- Phase 5 : landing ------------------------------- #

    # Pelvis translations
    q_bounds_5_min[X, :] = -0.01
    q_bounds_5_max[X, :] = 0.01
    q_bounds_5_min[Y, :] = -0.5
    q_bounds_5_max[Y, :] = 0.5
    q_bounds_5_min[Z, :] = 0
    q_bounds_5_max[Z, :] = zmax
    q_bounds_5_min[Z, END] = 0
    q_bounds_5_max[Z, END] = 0.01

    # Somersault
    q_bounds_5_min[Xrot, :] = -0.5 - 4 * np.pi - 0.1
    q_bounds_5_max[Xrot, :] = -7/2 * np.pi + 0.2 + 0.2
    q_bounds_5_min[Xrot, END] = 0.5 - 4 * np.pi - 0.01
    q_bounds_5_max[Xrot, END] = 0.5 - 4 * np.pi + 0.01
    # Tilt
    q_bounds_5_min[Yrot, :] = -np.pi / 16
    q_bounds_5_max[Yrot, :] = np.pi / 16
    # Twist
    q_bounds_5_min[Zrot, :] = 2 * np.pi * num_twists + 2 * np.pi - 0.01
    q_bounds_5_max[Zrot, :] = 2 * np.pi * num_twists + 2 * np.pi + 0.01

    # Right arm
    q_bounds_5_min[YrotRightUpperArm, START] = 0
    q_bounds_5_max[YrotRightUpperArm, START] = np.pi/8
    q_bounds_5_min[YrotRightUpperArm, END] = 2.9 - 0.1
    q_bounds_5_max[YrotRightUpperArm, END] = 2.9 + 0.1
    q_bounds_5_min[ZrotRightUpperArm, END] = -0.1
    q_bounds_5_max[ZrotRightUpperArm, END] = 0.1
    # Left arm
    q_bounds_5_min[YrotLeftUpperArm, START] = -np.pi/8
    q_bounds_5_max[YrotLeftUpperArm, START] = 0
    q_bounds_5_min[YrotLeftUpperArm, END] = -2.9 - 0.1
    q_bounds_5_max[YrotLeftUpperArm, END] = -2.9 + 0.1
    q_bounds_5_min[ZrotLeftUpperArm, END] = -0.1
    q_bounds_5_max[ZrotLeftUpperArm, END] = 0.1

    # Right elbow
    q_bounds_5_min[ZrotRightLowerArm : XrotRightLowerArm + 1, END] = -0.1
    q_bounds_5_max[ZrotRightLowerArm : XrotRightLowerArm + 1, END] = 0.1
    # Left elbow
    q_bounds_5_min[ZrotLeftLowerArm : XrotLeftLowerArm + 1, END] = -0.1
    q_bounds_5_max[ZrotLeftLowerArm : XrotLeftLowerArm + 1, END] = 0.1

    # Hips flexion
    q_bounds_5_min[XrotLegs, :] = -0.4
    q_bounds_5_min[XrotLegs, END] = -0.60
    q_bounds_5_max[XrotLegs, END] = -0.40
    # Hips sides
    q_bounds_5_min[YrotLegs, END] = -0.1
    q_bounds_5_max[YrotLegs, END] = 0.1

    # Translations velocities
    qdot_bounds_5_min[vX : vY + 1, :] = -10
    qdot_bounds_5_max[vX : vY + 1, :] = 10
    qdot_bounds_5_min[vZ, :] = -100
    qdot_bounds_5_max[vZ, :] = 100

    # Somersault
    qdot_bounds_5_min[vXrot, :] = -100
    qdot_bounds_5_max[vXrot, :] = 100
    # Tilt
    qdot_bounds_5_min[vYrot, :] = -100
    qdot_bounds_5_max[vYrot, :] = 100
    # Twist
    qdot_bounds_5_min[vZrot, :] = -100
    qdot_bounds_5_max[vZrot, :] = 100

    # Right arm
    qdot_bounds_5_min[vZrotRightUpperArm : vYrotRightUpperArm + 1, :] = -100
    qdot_bounds_5_max[vZrotRightUpperArm : vYrotRightUpperArm + 1, :] = 100
    # Left arm
    qdot_bounds_5_min[vZrotLeftUpperArm : vYrotLeftUpperArm + 1, :] = -100
    qdot_bounds_5_max[vZrotLeftUpperArm : vYrotLeftUpperArm + 1, :] = 100

    # Right elbow
    qdot_bounds_5_min[vZrotRightLowerArm : vYrotRightLowerArm + 1, :] = -100
    qdot_bounds_5_max[vZrotRightLowerArm : vYrotRightLowerArm + 1, :] = 100
    # Left elbow
    qdot_bounds_5_min[vZrotRightLowerArm : vYrotLeftLowerArm + 1, :] = -100
    qdot_bounds_5_max[vZrotRightLowerArm : vYrotLeftLowerArm + 1, :] = 100

    # Hip flexion
    qdot_bounds_5_min[vXrotLegs, :] = -100
    qdot_bounds_5_max[vXrotLegs, :] = 100
    # Hip sides
    qdot_bounds_5_min[vYrotLegs, :] = -100
    qdot_bounds_5_max[vYrotLegs, :] = 100

    x_bounds.add("q", min_bound=q_bounds_0_min, max_bound=q_bounds_0_max, phase=0)
    x_bounds.add("q", min_bound=q_bounds_1_min, max_bound=q_bounds_1_max, phase=1)
    x_bounds.add("q", min_bound=q_bounds_2_min, max_bound=q_bounds_2_max, phase=2)
    x_bounds.add("q", min_bound=q_bounds_3_min, max_bound=q_bounds_3_max, phase=3)
    x_bounds.add("q", min_bound=q_bounds_4_min, max_bound=q_bounds_4_max, phase=4)
    x_bounds.add("q", min_bound=q_bounds_5_min, max_bound=q_bounds_5_max, phase=5)
    x_bounds.add("qdot", min_bound=qdot_bounds_0_min, max_bound=qdot_bounds_0_max, phase=0)
    x_bounds.add("qdot", min_bound=qdot_bounds_1_min, max_bound=qdot_bounds_1_max, phase=1)
    x_bounds.add("qdot", min_bound=qdot_bounds_2_min, max_bound=qdot_bounds_2_max, phase=2)
    x_bounds.add("qdot", min_bound=qdot_bounds_3_min, max_bound=qdot_bounds_3_max, phase=3)
    x_bounds.add("qdot", min_bound=qdot_bounds_4_min, max_bound=qdot_bounds_4_max, phase=4)
    x_bounds.add("qdot", min_bound=qdot_bounds_5_min, max_bound=qdot_bounds_5_max, phase=5)

    # ------------------------------- Initial guesses ------------------------------- #

    q_0 = np.zeros((nb_q, 2))
    qdot_0 = np.zeros((nb_qdot, 2))
    q_1 = np.zeros((nb_q, 2))
    qdot_1 = np.zeros((nb_qdot, 2))
    q_2 = np.zeros((nb_q, 2))
    qdot_2 = np.zeros((nb_qdot, 2))
    q_3 = np.zeros((nb_q, 2))
    qdot_3 = np.zeros((nb_qdot, 2))
    q_4 = np.zeros((nb_q, 2))
    qdot_4 = np.zeros((nb_qdot, 2))
    q_5 = np.zeros((nb_q, 2))
    qdot_5 = np.zeros((nb_qdot, 2))

    q_0[Xrot] = np.array([0, -np.pi / 2])
    q_0[Zrot] = np.array([0, 2 * np.pi * num_twists])
    q_0[ZrotLeftUpperArm] = -0.75
    q_0[ZrotRightUpperArm] = 0.75
    q_0[YrotLeftUpperArm, 0] = -2.9
    q_0[YrotRightUpperArm, 0] = 2.9
    qdot_0[vXrot] = - 4 * np.pi

    q_1[Xrot] = np.array([-np.pi / 2, -3 / 4 * np.pi])
    q_1[Zrot] = np.array([2 * np.pi * num_twists, 2 * np.pi * num_twists + np.pi])
    q_1[XrotLegs] = np.array([0, -2.4])

    q_2[Xrot] = np.array([-3 / 4 * np.pi, -3 * np.pi])
    q_2[Zrot] = np.array([2 * np.pi * num_twists + np.pi, 2 * np.pi * num_twists + np.pi])
    q_2[XrotLegs, 0] = -2.4

    q_3[Xrot] = np.array([-3 * np.pi, -2 * np.pi - 5 / 4 * np.pi + 0.1])
    q_3[Zrot] = np.array([2 * np.pi * num_twists + np.pi, 2 * np.pi * num_twists + np.pi + np.pi/8])
    q_3[XrotLegs] = np.array([-2.4, 0])

    q_4[Xrot] = np.array([-2 * np.pi - 5 / 4 * np.pi + 0.1, -2 * np.pi - 5 / 4 * np.pi - 0.1])
    q_4[Zrot] = np.array([2 * np.pi * num_twists + np.pi + np.pi/8, 2 * np.pi * num_twists + 2 * np.pi])

    q_5[Xrot] = np.array([-2 * np.pi - 5 / 4 * np.pi - 0.1, -4 * np.pi + 0.5])
    q_5[Zrot] = np.array([2 * np.pi * num_twists + 2 * np.pi, 2 * np.pi * num_twists + 2 * np.pi])
    q_5[XrotLegs] = np.array([0, -0.5])

    x_init = InitialGuessList()
    x_init.add("q", initial_guess=q_0, interpolation=InterpolationType.LINEAR, phase=0)
    x_init.add("q", initial_guess=q_1, interpolation=InterpolationType.LINEAR, phase=1)
    x_init.add("q", initial_guess=q_2, interpolation=InterpolationType.LINEAR, phase=2)
    x_init.add("q", initial_guess=q_3, interpolation=InterpolationType.LINEAR, phase=3)
    x_init.add("q", initial_guess=q_4, interpolation=InterpolationType.LINEAR, phase=4)
    x_init.add("q", initial_guess=q_5, interpolation=InterpolationType.LINEAR, phase=5)
    x_init.add("qdot", initial_guess=qdot_0, interpolation=InterpolationType.LINEAR, phase=0)
    x_init.add("qdot", initial_guess=qdot_1, interpolation=InterpolationType.LINEAR, phase=1)
    x_init.add("qdot", initial_guess=qdot_2, interpolation=InterpolationType.LINEAR, phase=2)
    x_init.add("qdot", initial_guess=qdot_3, interpolation=InterpolationType.LINEAR, phase=3)
    x_init.add("qdot", initial_guess=qdot_4, interpolation=InterpolationType.LINEAR, phase=4)
    x_init.add("qdot", initial_guess=qdot_5, interpolation=InterpolationType.LINEAR, phase=5)

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
        constraints=constraints,
        ode_solver=ode_solver,
        n_threads=n_threads,
    )


def main():
    """
    Prepares and solves an ocp for a 831< with and without visual criteria.
    """

    WITH_VISUAL_CRITERIA = True

    if WITH_VISUAL_CRITERIA:
        biorbd_model_path = "models/SoMe_with_visual_criteria_without_mesh.bioMod"
    else:
        biorbd_model_path = "models/SoMe_without_mesh.bioMod"

    n_shooting = (40, 40, 40, 40, 40, 40)
    num_twists = 1
    ocp = prepare_ocp(biorbd_model_path, n_shooting=n_shooting, num_twists=num_twists, n_threads=7, WITH_VISUAL_CRITERIA=WITH_VISUAL_CRITERIA)
    # ocp.add_plot_penalty(CostType.ALL)

    solver = Solver.IPOPT(show_online_optim=False, show_options=dict(show_bounds=True))
    solver.set_linear_solver("ma57")
    solver.set_maximum_iterations(10000)
    solver.set_convergence_tolerance(1e-6)

    tic = time.time()
    sol = ocp.solve(solver)
    toc = time.time() - tic
    print(toc)
    # sol.graphs(show_bounds=True)

    timestamp = time.strftime("%Y-%m-%d-%H%M")
    name = biorbd_model_path.split("/")[-1].removesuffix(".bioMod")
    qs = sol.states[0]["q"][:, :-1]
    qdots = sol.states[0]["qdot"][:, :-1]
    qddots = sol.controls[0]["qddot_joints"][:, :-1]
    q_per_phase = [sol.states[0]["q"]]
    for i in range(1, len(sol.states)-1):
        qs = np.hstack((qs, sol.states[i]["q"][:, :-1]))
        qdots = np.hstack((qdots, sol.states[i]["qdot"][:, :-1]))
        qddots = np.hstack((qddots, sol.controls[i]["qddot_joints"][:, :-1]))
        q_per_phase.append(sol.states[i]["q"])
    qs = np.hstack((qs, sol.states[len(sol.states)-1]["q"]))
    qdots = np.hstack((qdots, sol.states[len(sol.states)-1]["qdot"]))
    qddots = np.hstack((qddots, sol.controls[len(sol.states)-1]["qddot_joints"]))
    time_parameters = sol.parameters["time"]
    q_per_phase.append(sol.states[len(sol.states)-1]["q"])


    integrated_sol = sol.integrate(shooting_type=Shooting.SINGLE,
                                   integrator=SolutionIntegrator.SCIPY_DOP853,
                                   keep_intermediate_points=False,
                                   merge_phases=True)

    time_vector = integrated_sol.time
    q_reintegrated = integrated_sol.states["q"]
    qdot_reintegrated = integrated_sol.states["qdot"]


    del sol.ocp
    with open(f"Solutions/{name}_831-{str(n_shooting).replace(', ', '_')}-{timestamp}.pkl", "wb") as f:
        pickle.dump((sol, q_per_phase, qs, qdots, qddots, time_parameters, q_reintegrated, qdot_reintegrated, time_vector), f)

    # sol.animate(n_frames=-1, show_floor=False)

if __name__ == "__main__":
    main()
