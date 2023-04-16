
"""
The goal of this program is to optimize the movement to achieve a 831< or a 851<.
Phase 0 : Twist
Phase 1: Pike
Phase 2 : Rotate in somersault in pike position
Phase 3 : Kick out + half twist
Phase 4 : preparation for landing
"""

import numpy as np
import pickle
import biorbd_casadi as biorbd
from casadi import MX, Function
import IPython
import time
import sys
sys.path.append("/home/charbie/Documents/Programmation/BiorbdOptim")
from bioptim import (
    OptimalControlProgram,
    DynamicsList,
    DynamicsFcn,
    PenaltyNode,
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
    PenaltyNodeList,
    BiorbdModel,
    Shooting,
    SolutionIntegrator,
)


def prepare_ocp(
    biorbd_model_path: str, n_shooting: tuple, num_twists: int, n_threads: int, ode_solver: OdeSolver = OdeSolver.RK4()
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
    )

    nb_q = biorbd_model[0].nb_q
    nb_qdot = biorbd_model[0].nb_qdot
    nb_qddot_joints = nb_q - biorbd_model[0].nb_root

    # for lisibility
    X = 0
    Y = 1
    Z = 2
    Xrot = 3
    Yrot = 4
    Zrot = 5
    ZrotBD = 6
    YrotBD = 7
    ZrotABD = 8
    XrotABD = 9
    ZrotBG = 10
    YrotBG = 11
    ZrotABG = 12
    XrotABG = 13
    XrotC = 14
    YrotC = 15
    vX = 0 + nb_q
    vY = 1 + nb_q
    vZ = 2 + nb_q
    vXrot = 3 + nb_q
    vYrot = 4 + nb_q
    vZrot = 5 + nb_q
    vZrotBD = 6 + nb_q
    vYrotBD = 7 + nb_q
    vZrotABD = 8 + nb_q
    vYrotABD = 9 + nb_q
    vZrotBG = 10 + nb_q
    vYrotBG = 11 + nb_q
    vZrotABG = 12 + nb_q
    vYrotABG = 13 + nb_q
    vXrotC = 14 + nb_q
    vYrotC = 15 + nb_q

    # Add objective functions
    objective_functions = ObjectiveList()

    # Min controls
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="qddot_joints", node=Node.ALL_SHOOTING, weight=1, phase=0
    )
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="qddot_joints", node=Node.ALL_SHOOTING, weight=1, phase=1
    )
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="qddot_joints", node=Node.ALL_SHOOTING, weight=1, phase=2
    )
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="qddot_joints", node=Node.ALL_SHOOTING, weight=1, phase=3
    )
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="qddot_joints", node=Node.ALL_SHOOTING, weight=1, phase=4
    )

    # Min control derivative
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="qddot_joints", node=Node.ALL_SHOOTING, weight=1, phase=0, derivative=True,
    )
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="qddot_joints", node=Node.ALL_SHOOTING, weight=1, phase=1, derivative=True,
    )
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="qddot_joints", node=Node.ALL_SHOOTING, weight=1, phase=2, derivative=True,
    )
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="qddot_joints", node=Node.ALL_SHOOTING, weight=1, phase=3, derivative=True,
    )
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="qddot_joints", node=Node.ALL_SHOOTING, weight=1, phase=4, derivative=True,
    )

    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_TIME, min_bound=0.05, max_bound=final_time / 2, weight=100, phase=0
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_TIME, min_bound=0.05, max_bound=final_time / 2, weight=100, phase=1
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_TIME, min_bound=0.05, max_bound=final_time / 2, weight=-0.01, phase=2
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_TIME, min_bound=0.05, max_bound=final_time / 2, weight=-0.01, phase=3
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_TIME, min_bound=0.05, max_bound=final_time / 2, weight=-0.01, phase=4
    )


    # Aim to put the hands on the lower legs to grab the pike position
    objective_functions.add(
        ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS,
        node=Node.END,
        first_marker="MidMainG",
        second_marker="CibleMainG",
        weight=10,
        phase=1,
    )
    objective_functions.add(
         ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS,
         node=Node.END,
         first_marker="MidMainD",
         second_marker="CibleMainD",
         weight=10,
         phase=1,
     )

    # aligning with the FIG regulations
    arm_dofs = [ZrotBD, YrotBD, ZrotABD, XrotABD, ZrotBG, YrotBG, ZrotABG, XrotABG]
    shoulder_dofs = [ZrotBD, YrotBD, ZrotBG, YrotBG]
    elbow_dofs = [ZrotABD, XrotABD, ZrotABG, XrotABG]

    objective_functions.add(
         ObjectiveFcn.Lagrange.MINIMIZE_STATE,
         key="q",
         node=Node.ALL_SHOOTING,
         index=elbow_dofs,
         target=np.zeros((len(elbow_dofs), n_shooting[0])),
         weight=1000000,
         phase=0,
    )
    objective_functions.add(
         ObjectiveFcn.Lagrange.MINIMIZE_STATE,
         key="q",
         node=Node.ALL_SHOOTING,
         index=shoulder_dofs,
         target=np.zeros((len(shoulder_dofs), n_shooting[2])),
         weight=1000000,
         phase=2,
    )
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_STATE,
        key="q",
        node=Node.ALL_SHOOTING,
        index=arm_dofs,
        target=np.zeros((len(arm_dofs), n_shooting[3])),
        weight=1000000,
        phase=3,
    )
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_STATE,
        key="q",
        node=Node.ALL_SHOOTING,
        index=elbow_dofs,
        target=np.zeros((len(elbow_dofs), n_shooting[4])),
        weight=1000000,
        phase=4,
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, key="q", node=Node.ALL, index=[XrotC], target=[0], weight=10000, phase=3
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, key="qdot", node=list(range(int(n_shooting[3]/4), n_shooting[3])), index=[XrotC], target=[0], weight=100, phase=3,
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, key="qdot", node=Node.START, index=[Xrot], weight=10, phase=0, quadratic=False
    )

    # ajouter une phase avec hanches ouvertes et environ 1/4 du salto

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.JOINTS_ACCELERATION_DRIVEN)
    dynamics.add(DynamicsFcn.JOINTS_ACCELERATION_DRIVEN)
    dynamics.add(DynamicsFcn.JOINTS_ACCELERATION_DRIVEN)
    dynamics.add(DynamicsFcn.JOINTS_ACCELERATION_DRIVEN)
    dynamics.add(DynamicsFcn.JOINTS_ACCELERATION_DRIVEN)

    qddot_joints_min, qddot_joints_max, qddot_joints_init = -1000, 1000, 0
    u_bounds = BoundsList()
    u_bounds.add([qddot_joints_min] * nb_qddot_joints, [qddot_joints_max] * nb_qddot_joints)
    u_bounds.add([qddot_joints_min] * nb_qddot_joints, [qddot_joints_max] * nb_qddot_joints)
    u_bounds.add([qddot_joints_min] * nb_qddot_joints, [qddot_joints_max] * nb_qddot_joints)
    u_bounds.add([qddot_joints_min] * nb_qddot_joints, [qddot_joints_max] * nb_qddot_joints)
    u_bounds.add([qddot_joints_min] * nb_qddot_joints, [qddot_joints_max] * nb_qddot_joints)

    u_init = InitialGuessList()
    u_init.add([qddot_joints_init] * nb_qddot_joints)
    u_init.add([qddot_joints_init] * nb_qddot_joints)
    u_init.add([qddot_joints_init] * nb_qddot_joints)
    u_init.add([qddot_joints_init] * nb_qddot_joints)
    u_init.add([qddot_joints_init] * nb_qddot_joints)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=biorbd_model[0].bounds_from_ranges(["q", "qdot"]))
    x_bounds.add(bounds=biorbd_model[1].bounds_from_ranges(["q", "qdot"]))
    x_bounds.add(bounds=biorbd_model[2].bounds_from_ranges(["q", "qdot"]))
    x_bounds.add(bounds=biorbd_model[3].bounds_from_ranges(["q", "qdot"]))
    x_bounds.add(bounds=biorbd_model[4].bounds_from_ranges(["q", "qdot"]))

    # Pour la lisibilite
    START, MIDDLE, END = 0, 1, 2

    # ------------------------------- Phase 0 : twist ------------------------------- #
    zmax = 9.81 / 8 * final_time**2 + 1

    # Pelvis translations
    x_bounds[0].min[X, :] = -0.25
    x_bounds[0].max[X, :] = 0.25
    x_bounds[0].min[Y, :] = -0.5
    x_bounds[0].max[Y, :] = 0.5
    x_bounds[0][: Z + 1, START] = 0
    x_bounds[0].min[Z, MIDDLE:] = 0
    x_bounds[0].max[Z, MIDDLE:] = zmax

    # Somersault
    x_bounds[0][Xrot, START] = 0
    x_bounds[0].min[Xrot, MIDDLE:] = -2 * np.pi
    x_bounds[0].max[Xrot, MIDDLE:] = 0.5
    # Tilt
    x_bounds[0][Yrot, START] = 0
    x_bounds[0].min[Yrot, MIDDLE:] = -np.pi / 4  # avoid gimbal lock
    x_bounds[0].max[Yrot, MIDDLE:] = np.pi / 4
    # Twist
    x_bounds[0][Zrot, START] = 0
    x_bounds[0].min[Zrot, MIDDLE] = -0.5
    x_bounds[0].max[Zrot, MIDDLE] = 2 * np.pi * num_twists
    x_bounds[0].min[Zrot, END] = 2 * np.pi * num_twists - 0.5
    x_bounds[0].max[Zrot, END] = 2 * np.pi * num_twists + 0.5

    # Right arm
    x_bounds[0][YrotBD, START] = 2.9
    x_bounds[0][ZrotBD, START] = 0
    # Left arm
    x_bounds[0][YrotBG, START] = -2.9
    x_bounds[0][ZrotBG, START] = 0

    # Right elbow
    x_bounds[0][ZrotABD : XrotABD + 1, START] = 0
    # Left elbow
    x_bounds[0][ZrotABG : XrotABG + 1, START] = 0

    # Hip flexion
    x_bounds[0][XrotC, START] = 0
    x_bounds[0].min[XrotC, MIDDLE:] = -0.2
    x_bounds[0].max[XrotC, MIDDLE:] = 0.2
    # Hip sides
    x_bounds[0][YrotC, START] = 0

    vzinit = 9.81 / 2 * final_time

    # Shift the initial vertical speed at the CoM
    CoM_Q_sym = MX.sym("CoM", nb_q)
    CoM_Q_init = x_bounds[0].min[:nb_q, START]
    CoM_Q_func = Function("CoM_Q_func", [CoM_Q_sym], [biorbd_model[0].center_of_mass(CoM_Q_sym)])
    bassin_Q_func = Function(
        "bassin_Q_func", [CoM_Q_sym], [biorbd_model[0].homogeneous_matrices_in_global(CoM_Q_sym, 0).to_mx()]
    )

    r = np.array(CoM_Q_func(CoM_Q_init)).reshape(1, 3) - np.array(bassin_Q_func(CoM_Q_init))[-1, :3]

    # Pelis translation velocities
    x_bounds[0].min[vX : vY + 1, :] = -10
    x_bounds[0].max[vX : vY + 1, :] = 10
    x_bounds[0].min[vX : vY + 1, START] = -0.5
    x_bounds[0].max[vX : vY + 1, START] = 0.5
    x_bounds[0].min[vZ, :] = -100
    x_bounds[0].max[vZ, :] = 100
    x_bounds[0].min[vZ, START] = vzinit - 0.5
    x_bounds[0].max[vZ, START] = vzinit + 0.5

    # Somersault
    x_bounds[0].min[vXrot, :] = -20
    x_bounds[0].max[vXrot, :] = -0.5
    # Tile
    x_bounds[0].min[vYrot, :] = -100
    x_bounds[0].max[vYrot, :] = 100
    x_bounds[0][vYrot, START] = 0
    # Twist
    x_bounds[0].min[vZrot, :] = -100
    x_bounds[0].max[vZrot, :] = 100
    x_bounds[0][vZrot, START] = 0

    min_bound_trans_velocity = (
        x_bounds[0].min[vX : vZ + 1, START] + np.cross(r, x_bounds[0].min[vXrot : vZrot + 1, START])
    )[0]
    max_bound_trans_velocity = (
        x_bounds[0].max[vX : vZ + 1, START] + np.cross(r, x_bounds[0].max[vXrot : vZrot + 1, START])
    )[0]
    x_bounds[0].min[vX : vZ + 1, START] = (
        min(max_bound_trans_velocity[0], min_bound_trans_velocity[0]),
        min(max_bound_trans_velocity[1], min_bound_trans_velocity[1]),
        min(max_bound_trans_velocity[2], min_bound_trans_velocity[2]),
    )
    x_bounds[0].max[vX : vZ + 1, START] = (
        max(max_bound_trans_velocity[0], min_bound_trans_velocity[0]),
        max(max_bound_trans_velocity[1], min_bound_trans_velocity[1]),
        max(max_bound_trans_velocity[2], min_bound_trans_velocity[2]),
    )

    # Right arm
    x_bounds[0].min[vZrotBD : vYrotBD + 1, :] = -100
    x_bounds[0].max[vZrotBD : vYrotBD + 1, :] = 100
    x_bounds[0][vZrotBD : vYrotBD + 1, START] = 0
    # Left arm
    x_bounds[0].min[vZrotBG : vYrotBG + 1, :] = -100
    x_bounds[0].max[vZrotBG : vYrotBG + 1, :] = 100
    x_bounds[0][vZrotBG : vYrotBG + 1, START] = 0

    # Right elbow
    x_bounds[0].min[vZrotABD : vYrotABD + 1, :] = -100
    x_bounds[0].max[vZrotABD : vYrotABD + 1, :] = 100
    x_bounds[0][vZrotABD : vYrotABD + 1, START] = 0
    # Left elbow
    x_bounds[0].min[vZrotABD : vYrotABG + 1, :] = -100
    x_bounds[0].max[vZrotABD : vYrotABG + 1, :] = 100
    x_bounds[0][vZrotABG : vYrotABG + 1, START] = 0

    # Hip flexion
    x_bounds[0].min[vXrotC, :] = -100
    x_bounds[0].max[vXrotC, :] = 100
    x_bounds[0][vXrotC, START] = 0
    # Hip sides
    x_bounds[0].min[vYrotC, :] = -100
    x_bounds[0].max[vYrotC, :] = 100
    x_bounds[0][vYrotC, START] = 0

    # ------------------------------- Phase 1 : piking ------------------------------- #
    # Pelvis translations
    x_bounds[1].min[X, :] = -0.25
    x_bounds[1].max[X, :] = 0.25
    x_bounds[1].min[Y, :] = -0.5
    x_bounds[1].max[Y, :] = 0.5
    x_bounds[1].min[Z, :] = 0
    x_bounds[1].max[Z, :] = zmax

    # Somersault
    x_bounds[1].min[Xrot, :] = -5 / 4 * np.pi
    x_bounds[1].max[Xrot, :] = 0
    x_bounds[1].min[Xrot, END] = -5 / 4 * np.pi
    x_bounds[1].max[Xrot, END] = -np.pi / 2
    # Tilt
    x_bounds[1].min[Yrot, :] = -np.pi / 4
    x_bounds[1].max[Yrot, :] = np.pi / 4
    # Twist
    x_bounds[1].min[Zrot, :] = 2 * np.pi * num_twists - 0.5
    x_bounds[1].max[Zrot, :] = 2 * np.pi * num_twists + np.pi + 0.5
    x_bounds[1].min[Zrot, END] = 2 * np.pi * num_twists + np.pi - 0.5
    x_bounds[1].max[Zrot, END] = 2 * np.pi * num_twists + np.pi + 0.5

    # Hips flexion
    x_bounds[1].min[XrotC, START] = -0.2
    x_bounds[1].max[XrotC, START] = 0.2
    x_bounds[1].min[XrotC, MIDDLE] = -2.5 - 0.2
    x_bounds[1].max[XrotC, MIDDLE] = 0.2
    x_bounds[1].min[XrotC, END] = -2.5 - 0.2
    x_bounds[1].max[XrotC, END] = -2.5 + 0.2
    # Hips sides
    x_bounds[1].min[YrotC, END] = -0.1
    x_bounds[1].max[YrotC, END] = 0.1

    # Translations velocities
    x_bounds[1].min[vX : vY + 1, :] = -10
    x_bounds[1].max[vX : vY + 1, :] = 10
    x_bounds[1].min[vZ, :] = -100
    x_bounds[1].max[vZ, :] = 100
    # Somersault
    x_bounds[1].min[vXrot, :] = -100
    x_bounds[1].max[vXrot, :] = 100
    # Tilt
    x_bounds[1].min[vYrot, :] = -100
    x_bounds[1].max[vYrot, :] = 100
    # Twist
    x_bounds[1].min[vZrot, :] = -100
    x_bounds[1].max[vZrot, :] = 100

    # Right arm
    x_bounds[1].min[vZrotBD : vYrotBD + 1, :] = -100
    x_bounds[1].max[vZrotBD : vYrotBD + 1, :] = 100
    # Left elbow
    x_bounds[1].min[vZrotBG : vYrotBG + 1, :] = -100
    x_bounds[1].max[vZrotBG : vYrotBG + 1, :] = 100

    # Right elbow
    x_bounds[1].min[vZrotABD : vYrotABD + 1, :] = -100
    x_bounds[1].max[vZrotABD : vYrotABD + 1, :] = 100
    # Left elbow
    x_bounds[1].min[vZrotABD : vYrotABG + 1, :] = -100
    x_bounds[1].max[vZrotABD : vYrotABG + 1, :] = 100

    # Hip flexion
    x_bounds[1].min[vXrotC, :] = -100
    x_bounds[1].max[vXrotC, :] = 100
    # Hip sides
    x_bounds[1].min[vYrotC, :] = -100
    x_bounds[1].max[vYrotC, :] = 100

    # ------------------------------- Phase 2 : somersault in pike ------------------------------- #

    # Pelvis translations
    x_bounds[2].min[X, :] = -0.25
    x_bounds[2].max[X, :] = 0.25
    x_bounds[2].min[Y, :] = -0.5
    x_bounds[2].max[Y, :] = 0.5
    x_bounds[2].min[Z, :] = 0
    x_bounds[2].max[Z, :] = zmax  # beaucoup plus que necessaire, juste pour que la parabole fonctionne

    # Somersault
    x_bounds[2].min[Xrot, :] = -3 * np.pi
    x_bounds[2].max[Xrot, :] = - np.pi
    # Tilt
    x_bounds[2].min[Yrot, :] = -np.pi / 8
    x_bounds[2].max[Yrot, :] = np.pi / 8
    # Twist
    x_bounds[2].min[Zrot, :] = 2 * np.pi * num_twists + np.pi - np.pi / 8
    x_bounds[2].max[Zrot, :] = 2 * np.pi * num_twists + np.pi + np.pi / 8

    # Hips flexion
    x_bounds[2].min[XrotC, :] = -2.5 - 0.2
    x_bounds[2].max[XrotC, :] = -2.5 + 0.2
    # Hips sides
    x_bounds[2].min[YrotC, :] = -0.1
    x_bounds[2].max[YrotC, :] = 0.1

    # Translations velocities
    x_bounds[2].min[vX : vY + 1, :] = -10
    x_bounds[2].max[vX : vY + 1, :] = 10
    x_bounds[2].min[vZ, :] = -100
    x_bounds[2].max[vZ, :] = 100

    # Somersault
    x_bounds[2].min[vXrot, :] = -100
    x_bounds[2].max[vXrot, :] = 100
    # Tilt
    x_bounds[2].min[vYrot, :] = -100
    x_bounds[2].max[vYrot, :] = 100
    # Twist
    x_bounds[2].min[vZrot, :] = -100
    x_bounds[2].max[vZrot, :] = 100

    # Right arm
    x_bounds[2].min[vZrotBD : vYrotBD + 1, :] = -100
    x_bounds[2].max[vZrotBD : vYrotBD + 1, :] = 100
    # Left arm
    x_bounds[2].min[vZrotBG : vYrotBG + 1, :] = -100
    x_bounds[2].max[vZrotBG : vYrotBG + 1, :] = 100

    # Right elbow
    x_bounds[2].min[vZrotABD : vYrotABD + 1, :] = -100
    x_bounds[2].max[vZrotABD : vYrotABD + 1, :] = 100
    # Left elbow
    x_bounds[2].min[vZrotABD : vYrotABG + 1, :] = -100
    x_bounds[2].max[vZrotABD : vYrotABG + 1, :] = 100

    # Hip flexion
    x_bounds[2].min[vXrotC, :] = -100
    x_bounds[2].max[vXrotC, :] = 100
    # Hip sides
    x_bounds[2].min[vYrotC, :] = -100
    x_bounds[2].max[vYrotC, :] = 100

    # ------------------------------- Phase 3 : kick out + 1/2 twist ------------------------------- #

    # Pelvis translations
    x_bounds[3].min[X, :] = -0.25
    x_bounds[3].max[X, :] = 0.25
    x_bounds[3].min[Y, :] = -0.5
    x_bounds[3].max[Y, :] = 0.5
    x_bounds[3].min[Z, :] = 0
    x_bounds[3].max[Z, :] = zmax

    # Somersault
    x_bounds[3].min[Xrot, START] = -3 * np.pi
    x_bounds[3].max[Xrot, START] = -2 * np.pi
    x_bounds[3].min[Xrot, MIDDLE] = -7/2 * np.pi
    x_bounds[3].max[Xrot, MIDDLE] = -2 * np.pi
    x_bounds[3].min[Xrot, END] = -7/2 * np.pi + 0.2 - 0.2
    x_bounds[3].max[Xrot, END] = -7/2 * np.pi + 0.2 + 0.2
    # Tilt
    x_bounds[3].min[Yrot, :] = -np.pi / 4
    x_bounds[3].max[Yrot, :] = np.pi / 4
    x_bounds[3].min[Yrot, END] = -np.pi / 8
    x_bounds[3].max[Yrot, END] = np.pi / 8
    # Twist
    x_bounds[3].min[Zrot, START] = 2 * np.pi * num_twists + np.pi - np.pi / 4
    x_bounds[3].max[Zrot, START] = 2 * np.pi * num_twists + np.pi + np.pi / 4
    x_bounds[3].min[Zrot, MIDDLE] = 2 * np.pi * num_twists + np.pi - np.pi / 4
    x_bounds[3].max[Zrot, MIDDLE] = 2 * np.pi * num_twists + 2 * np.pi + np.pi / 8
    x_bounds[3].min[Zrot, END] = 2 * np.pi * num_twists + 2 * np.pi - np.pi / 8
    x_bounds[3].max[Zrot, END] = 2 * np.pi * num_twists + 2 * np.pi + np.pi / 8

    # Hips flexion
    x_bounds[3].min[XrotC, START] = -2.5 - 0.2
    x_bounds[3].max[XrotC, START] = -2.5 + 0.2
    x_bounds[3].min[XrotC, MIDDLE] = -2.5 - 0.2
    x_bounds[3].max[XrotC, MIDDLE] = 0.2
    x_bounds[3].min[XrotC, END] = -0.2
    x_bounds[3].max[XrotC, END] = 0.2

    # Translations velocities
    x_bounds[3].min[vX : vY + 1, :] = -10
    x_bounds[3].max[vX : vY + 1, :] = 10
    x_bounds[3].min[vZ, :] = -100
    x_bounds[3].max[vZ, :] = 100

    # Somersault
    x_bounds[3].min[vXrot, :] = -100
    x_bounds[3].max[vXrot, :] = 100
    # Tilt
    x_bounds[3].min[vYrot, :] = -100
    x_bounds[3].max[vYrot, :] = 100
    # Twist
    x_bounds[3].min[vZrot, :] = -100
    x_bounds[3].max[vZrot, :] = 100

    # Right arm
    x_bounds[3].min[vZrotBD : vYrotBD + 1, :] = -100
    x_bounds[3].max[vZrotBD : vYrotBD + 1, :] = 100
    # Left arm
    x_bounds[3].min[vZrotBG : vYrotBG + 1, :] = -100
    x_bounds[3].max[vZrotBG : vYrotBG + 1, :] = 100

    # Right elbow
    x_bounds[3].min[vZrotABD : vYrotABD + 1, :] = -100
    x_bounds[3].max[vZrotABD : vYrotABD + 1, :] = 100
    # Left elbow
    x_bounds[3].min[vZrotABD : vYrotABG + 1, :] = -100
    x_bounds[3].max[vZrotABD : vYrotABG + 1, :] = 100

    # Hip flexion
    x_bounds[3].min[vXrotC, :] = -100
    x_bounds[3].max[vXrotC, :] = 100
    # Hip sides
    x_bounds[3].min[vYrotC, :] = -100
    x_bounds[3].max[vYrotC, :] = 100

    # ------------------------------- Phase 4 : landing ------------------------------- #

    # Pelvis translations
    x_bounds[4].min[X, :] = -0.25
    x_bounds[4].max[X, :] = 0.25
    x_bounds[4].min[Y, :] = -0.5
    x_bounds[4].max[Y, :] = 0.5
    x_bounds[4].min[Z, :] = 0
    x_bounds[4].max[Z, :] = zmax
    x_bounds[4].min[Z, END] = 0
    x_bounds[4].max[Z, END] = 0.1

    # Somersault
    x_bounds[4].min[Xrot, :] = -0.5 - 4 * np.pi - 0.1
    x_bounds[4].max[Xrot, :] = -7/2 * np.pi + 0.2 + 0.2
    x_bounds[4].min[Xrot, END] = 0.5 - 4 * np.pi - 0.1
    x_bounds[4].max[Xrot, END] = 0.5 - 4 * np.pi + 0.1
    # Tilt
    x_bounds[4].min[Yrot, :] = -np.pi / 16
    x_bounds[4].max[Yrot, :] = np.pi / 16
    # Twist
    x_bounds[4].min[Zrot, :] = 2 * np.pi * num_twists + 2 * np.pi - np.pi / 8
    x_bounds[4].max[Zrot, :] = 2 * np.pi * num_twists + 2 * np.pi + np.pi / 8

    # Right arm
    x_bounds[4].min[YrotBD, END] = 2.9 - 0.1
    x_bounds[4].max[YrotBD, END] = 2.9 + 0.1
    x_bounds[4].min[ZrotBD, END] = -0.1
    x_bounds[4].max[ZrotBD, END] = 0.1
    # Left arm
    x_bounds[4].min[YrotBG, END] = -2.9 - 0.1
    x_bounds[4].max[YrotBG, END] = -2.9 + 0.1
    x_bounds[4].min[ZrotBG, END] = -0.1
    x_bounds[4].max[ZrotBG, END] = 0.1

    # Right elbow
    x_bounds[4].min[ZrotABD : XrotABD + 1, END] = -0.1
    x_bounds[4].max[ZrotABD : XrotABD + 1, END] = 0.1
    # Left elbow
    x_bounds[4].min[ZrotABG : XrotABG + 1, END] = -0.1
    x_bounds[4].max[ZrotABG : XrotABG + 1, END] = 0.1

    # Hips flexion
    x_bounds[4].min[XrotC, :] = -0.4
    x_bounds[4].min[XrotC, END] = -0.60
    x_bounds[4].max[XrotC, END] = -0.40
    # Hips sides
    x_bounds[4].min[YrotC, END] = -0.1
    x_bounds[4].max[YrotC, END] = 0.1

    # Translations velocities
    x_bounds[4].min[vX : vY + 1, :] = -10
    x_bounds[4].max[vX : vY + 1, :] = 10
    x_bounds[4].min[vZ, :] = -100
    x_bounds[4].max[vZ, :] = 100

    # Somersault
    x_bounds[4].min[vXrot, :] = -100
    x_bounds[4].max[vXrot, :] = 100
    # Tilt
    x_bounds[4].min[vYrot, :] = -100
    x_bounds[4].max[vYrot, :] = 100
    # Twist
    x_bounds[4].min[vZrot, :] = -100
    x_bounds[4].max[vZrot, :] = 100

    # Right arm
    x_bounds[4].min[vZrotBD : vYrotBD + 1, :] = -100
    x_bounds[4].max[vZrotBD : vYrotBD + 1, :] = 100
    # Left arm
    x_bounds[4].min[vZrotBG : vYrotBG + 1, :] = -100
    x_bounds[4].max[vZrotBG : vYrotBG + 1, :] = 100

    # Right elbow
    x_bounds[4].min[vZrotABD : vYrotABD + 1, :] = -100
    x_bounds[4].max[vZrotABD : vYrotABD + 1, :] = 100
    # Left elbow
    x_bounds[4].min[vZrotABD : vYrotABG + 1, :] = -100
    x_bounds[4].max[vZrotABD : vYrotABG + 1, :] = 100

    # Hip flexion
    x_bounds[4].min[vXrotC, :] = -100
    x_bounds[4].max[vXrotC, :] = 100
    # Hip sides
    x_bounds[4].min[vYrotC, :] = -100
    x_bounds[4].max[vYrotC, :] = 100

    # ------------------------------- Initial guesses ------------------------------- #

    x0 = np.vstack((np.zeros((nb_q, 2)), np.zeros((nb_qdot, 2))))
    x1 = np.vstack((np.zeros((nb_q, 2)), np.zeros((nb_qdot, 2))))
    x2 = np.vstack((np.zeros((nb_q, 2)), np.zeros((nb_qdot, 2))))
    x3 = np.vstack((np.zeros((nb_q, 2)), np.zeros((nb_qdot, 2))))
    x4 = np.vstack((np.zeros((nb_q, 2)), np.zeros((nb_qdot, 2))))

    x0[Xrot] = np.array([0, -np.pi / 2])
    x0[Zrot] = np.array([0, 2 * np.pi * num_twists])
    x0[ZrotBG] = -0.75
    x0[ZrotBD] = 0.75
    x0[YrotBG, 0] = -2.9
    x0[YrotBD, 0] = 2.9
    # x0[YrotBG, 1] = -1.35
    # x0[YrotBD, 1] = 1.35
    # x0[XrotC, 0] = -.5
    x0[vXrot] = - 4 * np.pi

    x1[Xrot] = np.array([-np.pi / 2, -3 / 4 * np.pi])
    x1[Zrot] = np.array([2 * np.pi * num_twists, 2 * np.pi * num_twists + np.pi])
    # x1[ZrotBG] = -.75
    # x1[ZrotBD] = .75
    # x1[YrotBG] = -1.35
    # x1[YrotBD] = 1.35
    x1[XrotC] = np.array([0, -2.5])

    x2[Xrot] = np.array([-3 / 4 * np.pi, -3 * np.pi])
    x2[Zrot] = np.array([2 * np.pi * num_twists + np.pi, 2 * np.pi * num_twists + np.pi])
    # x2[ZrotBG, 0] = -.75
    # x2[ZrotBD, 0] = .75
    # x2[YrotBG, 0] = -1.35
    # x2[YrotBD, 0] = 1.35
    x2[XrotC, 0] = -2.5

    x3[Xrot] = np.array([-3 * np.pi, -2 * np.pi - 5 / 4 * np.pi])
    x3[Zrot] = np.array([2 * np.pi * num_twists + np.pi, 2 * np.pi * num_twists + 2 * np.pi])
    x3[XrotC] = np.array([-2.5, 0])

    x4[Xrot] = np.array([-2 * np.pi - 5 / 4 * np.pi, -4 * np.pi + 0.5])
    x4[Zrot] = np.array([2 * np.pi * num_twists + 2 * np.pi, 2 * np.pi * num_twists + 2 * np.pi])
    x4[XrotC] = np.array([0, -0.5])

    x_init = InitialGuessList()
    x_init.add(x0, interpolation=InterpolationType.LINEAR)
    x_init.add(x1, interpolation=InterpolationType.LINEAR)
    x_init.add(x2, interpolation=InterpolationType.LINEAR)
    x_init.add(x3, interpolation=InterpolationType.LINEAR)
    x_init.add(x4, interpolation=InterpolationType.LINEAR)

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
    # constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=1e-4, max_bound=final_time, phase=0)
    # constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=1e-4, max_bound=final_time, phase=1)
    # constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=1e-4, max_bound=final_time, phase=2)
    # constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=1e-4, max_bound=final_time, phase=3)
    # constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=1e-4, max_bound=final_time, phase=4)

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting,
        [final_time / len(biorbd_model)] * len(biorbd_model),
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        constraints,
        ode_solver=ode_solver,
        n_threads=n_threads,
    )


def main():
    """
    Prepares and solves an ocp for a 803<. Animates the results
    """

    biorbd_model_path = "/home/charbie/Documents/Programmation/VisionOCP/models/SoMe.bioMod"
    n_shooting = (40, 100, 100, 100, 40)
    num_twists = 1
    ocp = prepare_ocp(biorbd_model_path, n_shooting=n_shooting, num_twists=num_twists, n_threads=7)
   # ocp.add_plot_penalty(CostType.ALL)

    solver = Solver.IPOPT(show_online_optim=True, show_options=dict(show_bounds=True))
    solver.set_linear_solver("ma57")
    solver.set_maximum_iterations(10000)
    solver.set_convergence_tolerance(1e-4)
    sol = ocp.solve(solver)

    timestamp = time.strftime("%Y-%m-%d-%H%M")
    name = biorbd_model_path.split("/")[-1].removesuffix(".bioMod")
    qs = sol.states[0]["q"]
    qdots = sol.states[0]["qdot"]
    qddots = sol.controls[0]["qddot_joints"]
    for i in range(1, len(sol.states)):
        qs = np.hstack((qs, sol.states[i]["q"]))
        qdots = np.hstack((qdots, sol.states[i]["qdot"]))
        qddots = np.hstack((qddots, sol.controls[i]["qddot_joints"]))
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
    with open(f"Solutions/{name}-{num_twists}-{str(n_shooting).replace(', ', '_')}-{timestamp}.pkl", "wb") as f:
        pickle.dump((sol, qs, qdots, qddots, time_parameters, q_reintegrated, qdot_reintegrated, time_vector), f)

    # sol.animate(n_frames=-1, show_floor=False)

if __name__ == "__main__":
    main()
