
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
import casadi as cas
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

def custom_trampoline_bed_in_peripheral_vision(all_pn: PenaltyNodeList) -> cas.MX:
    """
    This function aims to encourage the avatar to keep the trampoline bed in his peripheral vision.
    It is done by discretizing the vision cone into vectors and determining if the vector projection of the gaze are inside the trampoline bed.
    """

    a = 1.07  # Trampoline with/2
    b = 2.14  # Trampoline length/2
    n = 6  # order of the polynomial for the trampoline bed rectangle equation

    # Get the gaze vector
    eyes_vect_start_marker_idx = all_pn.nlp.model.marker_index(f'eyes_vect_start')
    eyes_vect_end_marker_idx = all_pn.nlp.model.marker_index(f'eyes_vect_end')
    gaze_vector = all_pn.nlp.model.markers(all_pn.nlp.states["q"].mx).to_mx()[eyes_vect_start_marker_idx] - all_pn.nlp.model.markers(all_pn.nlp.states["q"].mx).to_mx()[eyes_vect_end_marker_idx]

    obj = 0
    for i_r in range(11):
        for i_th in range(10):

            # Get this vector from the vision cone
            marker_idx = all_pn.nlp.model.marker_index(f'cone_approx_{i_r}_{i_th}')
            vector_origin = all_pn.nlp.model.markers(all_pn.nlp.states["q"].mx).to_mx()[eyes_vect_start_marker_idx]
            vector_end = all_pn.nlp.model.markers(all_pn.nlp.states["q"].mx).to_mx()[marker_idx]
            vector = vector_end - vector_origin

            # Get the intersection between the vector and the trampoline plane
            point_in_the_plane = np.array([1, 2, -0.83])
            vector_normal_to_the_plane = np.array([0, 0, 1])
            t = (np.dot(point_in_the_plane, vector_normal_to_the_plane) - np.dot(vector_normal_to_the_plane, vector_origin)) / np.dot(
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

    val = all_pn.nlp.mx_to_cx("peripheral_vision", obj, all_pn.nlp.states["q"])

    return cas.if_else(gaze_vector > 0, 2*10*11, val)


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
        vX = 0 + nb_q
        vY = 1 + nb_q
        vZ = 2 + nb_q
        vXrot = 3 + nb_q
        vYrot = 4 + nb_q
        vZrot = 5 + nb_q
        vZrotRightUpperArm = 6 + nb_q
        vYrotRightUpperArm = 7 + nb_q
        vZrotRightLowerArm = 8 + nb_q
        vYrotRightLowerArm = 9 + nb_q
        vZrotLeftUpperArm = 10 + nb_q
        vYrotLeftUpperArm = 11 + nb_q
        vZrotLeftLowerArm = 12 + nb_q
        vYrotLeftLowerArm = 13 + nb_q
        vXrotLegs = 14 + nb_q
        vYrotLegs = 15 + nb_q
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
        vX = 0 + nb_q
        vY = 1 + nb_q
        vZ = 2 + nb_q
        vXrot = 3 + nb_q
        vYrot = 4 + nb_q
        vZrot = 5 + nb_q
        vZrotHead = 6 + nb_q
        vXrotHead = 7 + nb_q
        vZrotEyes = 8 + nb_q
        vXrotEyes = 9 + nb_q
        vZrotRightUpperArm = 10 + nb_q
        vYrotRightUpperArm = 11 + nb_q
        vZrotRightLowerArm = 12 + nb_q
        vYrotRightLowerArm = 13 + nb_q
        vZrotLeftUpperArm = 14 + nb_q
        vYrotLeftUpperArm = 15 + nb_q
        vZrotLeftLowerArm = 16 + nb_q
        vYrotLeftLowerArm = 17 + nb_q
        vXrotLegs = 18 + nb_q
        vYrotLegs = 19 + nb_q

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
    arm_dofs = [ZrotRightUpperArm, YrotRightUpperArm, ZrotRightLowerArm, XrotRightLowerArm, ZrotLeftUpperArm, YrotLeftUpperArm, ZrotLeftLowerArm, XrotLeftLowerArm]
    shoulder_dofs = [ZrotRightUpperArm, YrotRightUpperArm, ZrotLeftUpperArm, YrotLeftUpperArm]
    elbow_dofs = [ZrotRightLowerArm, XrotRightLowerArm, ZrotLeftLowerArm, XrotLeftLowerArm]

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

    # Quick kick out
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, key="q", node=Node.ALL, index=[XrotLegs], target=[0], weight=10000, phase=3
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, key="qdot", node=list(range(int(n_shooting[3]/4), n_shooting[3])), index=[XrotLegs], target=[0], weight=100000, phase=3,
    )
    # objective_functions.add(
    #     ObjectiveFcn.Mayer.MINIMIZE_STATE, key="qdot", node=Node.START, index=[Xrot], weight=10, phase=0, quadratic=False
    # )
    objective_functions.add(ObjectiveFcn.Mayer.PROPORTIONAL_STATE, key="q", node=list(range(0, int(n_shooting[3]/4))), first_dof=XrotLegs, second_dof=Zrot,
                            coef=2.5/(0.5*np.pi), first_dof_intercept=-2.5, second_dof_intercept=3*np.pi, weight=1000000, phase=3, quadratic=True)

    # Minimize wobbling
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, key="q", node=Node.ALL, index=[Yrot], target=[0], weight=10000, phase=2
    )

    if WITH_VISUAL_CRITERIA:

        # Spotting
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_SEGMENT_VELOCITY, segment="Head", weight=100, phase=3)
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_SEGMENT_VELOCITY, segment="Head", weight=100, phase=4)

        # Self-motion detection
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key='qdot', index=[ZrotEyes, XrotEyes], weight=100, phase=0)

        # Keeping the trampoline bed in the peripheral vision
        objective_functions.add(custom_trampoline_bed_in_peripheral_vision, custom_type=ObjectiveFcn.Lagrange, weight=1, phase=0)
        objective_functions.add(custom_trampoline_bed_in_peripheral_vision, custom_type=ObjectiveFcn.Lagrange, weight=1, phase=2)
        objective_functions.add(custom_trampoline_bed_in_peripheral_vision, custom_type=ObjectiveFcn.Lagrange, weight=100, phase=3)

        # Quiet eye
        objective_functions.add(ObjectiveFcn.Lagrange.TRACK_VECTOR_ORIENTATIONS_FROM_MARKERS,
                                vector_0_marker_0="eyes_vect_start",
                                vector_0_marker_1="eyes_vect_end",
                                vector_1_marker_0="eyes_vect_start",
                                vector_1_marker_1="fixation_center",
                                weight=10, phase=0)
        objective_functions.add(ObjectiveFcn.Lagrange.TRACK_VECTOR_ORIENTATIONS_FROM_MARKERS,
                                vector_0_marker_0="eyes_vect_start",
                                vector_0_marker_1="eyes_vect_end",
                                vector_1_marker_0="eyes_vect_start",
                                vector_1_marker_1="fixation_center",
                                weight=10, phase=1)
        objective_functions.add(ObjectiveFcn.Lagrange.TRACK_VECTOR_ORIENTATIONS_FROM_MARKERS,
                                vector_0_marker_0="eyes_vect_start",
                                vector_0_marker_1="eyes_vect_end",
                                vector_1_marker_0="eyes_vect_start",
                                vector_1_marker_1="fixation_center",
                                weight=10, phase=2)
        objective_functions.add(ObjectiveFcn.Lagrange.TRACK_VECTOR_ORIENTATIONS_FROM_MARKERS,
                                vector_0_marker_0="eyes_vect_start",
                                vector_0_marker_1="eyes_vect_end",
                                vector_1_marker_0="eyes_vect_start",
                                vector_1_marker_1="fixation_center",
                                weight=10, phase=3)
        objective_functions.add(ObjectiveFcn.Lagrange.TRACK_VECTOR_ORIENTATIONS_FROM_MARKERS,
                                vector_0_marker_0="eyes_vect_start",
                                vector_0_marker_1="eyes_vect_end",
                                vector_1_marker_0="eyes_vect_start",
                                vector_1_marker_1="fixation_front",
                                weight=10000, phase=4)

        # Avoid extreme eye angles
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", index=[ZrotEyes, XrotEyes], target=[0], weight=1, phase=0)
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", index=[ZrotEyes, XrotEyes], target=[0], weight=1, phase=1)
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", index=[ZrotEyes, XrotEyes], target=[0], weight=1, phase=2)
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", index=[ZrotEyes, XrotEyes], target=[0], weight=1, phase=3)
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", index=[ZrotEyes, XrotEyes], target=[0], weight=1, phase=4)


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
    x_bounds[0][YrotRightUpperArm, START] = 2.9
    x_bounds[0][ZrotRightUpperArm, START] = 0
    # Left arm
    x_bounds[0][YrotLeftUpperArm, START] = -2.9
    x_bounds[0][ZrotLeftUpperArm, START] = 0

    # Right elbow
    x_bounds[0][ZrotRightLowerArm : XrotRightLowerArm + 1, START] = 0
    # Left elbow
    x_bounds[0][ZrotLeftLowerArm : XrotLeftLowerArm + 1, START] = 0

    # Hip flexion
    x_bounds[0][XrotLegs, START] = 0
    x_bounds[0].min[XrotLegs, MIDDLE:] = -0.2
    x_bounds[0].max[XrotLegs, MIDDLE:] = 0.2
    # Hip sides
    x_bounds[0][YrotLegs, START] = 0

    vzinit = 9.81 / 2 * final_time

    # Shift the initial vertical speed at the CoM
    CoM_Q_sym = cas.MX.sym("CoM", nb_q)
    CoM_Q_init = x_bounds[0].min[:nb_q, START]
    CoM_Q_func = cas.Function("CoM_Q_func", [CoM_Q_sym], [biorbd_model[0].center_of_mass(CoM_Q_sym)])
    bassin_Q_func = cas.Function(
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
    x_bounds[0].min[vZrotRightUpperArm : vYrotRightUpperArm + 1, :] = -100
    x_bounds[0].max[vZrotRightUpperArm : vYrotRightUpperArm + 1, :] = 100
    x_bounds[0][vZrotRightUpperArm : vYrotRightUpperArm + 1, START] = 0
    # Left arm
    x_bounds[0].min[vZrotLeftUpperArm : vYrotLeftUpperArm + 1, :] = -100
    x_bounds[0].max[vZrotLeftUpperArm : vYrotLeftUpperArm + 1, :] = 100
    x_bounds[0][vZrotLeftUpperArm : vYrotLeftUpperArm + 1, START] = 0

    # Right elbow
    x_bounds[0].min[vZrotRightLowerArm : vYrotRightLowerArm + 1, :] = -100
    x_bounds[0].max[vZrotRightLowerArm : vYrotRightLowerArm + 1, :] = 100
    x_bounds[0][vZrotRightLowerArm : vYrotRightLowerArm + 1, START] = 0
    # Left elbow
    x_bounds[0].min[vZrotRightLowerArm : vYrotLeftLowerArm + 1, :] = -100
    x_bounds[0].max[vZrotRightLowerArm : vYrotLeftLowerArm + 1, :] = 100
    x_bounds[0][vZrotLeftLowerArm : vYrotLeftLowerArm + 1, START] = 0

    # Hip flexion
    x_bounds[0].min[vXrotLegs, :] = -100
    x_bounds[0].max[vXrotLegs, :] = 100
    x_bounds[0][vXrotLegs, START] = 0
    # Hip sides
    x_bounds[0].min[vYrotLegs, :] = -100
    x_bounds[0].max[vYrotLegs, :] = 100
    x_bounds[0][vYrotLegs, START] = 0

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
    x_bounds[1].min[XrotLegs, START] = -0.2
    x_bounds[1].max[XrotLegs, START] = 0.2
    x_bounds[1].min[XrotLegs, MIDDLE] = -2.5 - 0.2
    x_bounds[1].max[XrotLegs, MIDDLE] = 0.2
    x_bounds[1].min[XrotLegs, END] = -2.5 - 0.2
    x_bounds[1].max[XrotLegs, END] = -2.5 + 0.2
    # Hips sides
    x_bounds[1].min[YrotLegs, END] = -0.1
    x_bounds[1].max[YrotLegs, END] = 0.1

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
    x_bounds[1].min[vZrotRightUpperArm : vYrotRightUpperArm + 1, :] = -100
    x_bounds[1].max[vZrotRightUpperArm : vYrotRightUpperArm + 1, :] = 100
    # Left elbow
    x_bounds[1].min[vZrotLeftUpperArm : vYrotLeftUpperArm + 1, :] = -100
    x_bounds[1].max[vZrotLeftUpperArm : vYrotLeftUpperArm + 1, :] = 100

    # Right elbow
    x_bounds[1].min[vZrotRightLowerArm : vYrotRightLowerArm + 1, :] = -100
    x_bounds[1].max[vZrotRightLowerArm : vYrotRightLowerArm + 1, :] = 100
    # Left elbow
    x_bounds[1].min[vZrotRightLowerArm : vYrotLeftLowerArm + 1, :] = -100
    x_bounds[1].max[vZrotRightLowerArm : vYrotLeftLowerArm + 1, :] = 100

    # Hip flexion
    x_bounds[1].min[vXrotLegs, :] = -100
    x_bounds[1].max[vXrotLegs, :] = 100
    # Hip sides
    x_bounds[1].min[vYrotLegs, :] = -100
    x_bounds[1].max[vYrotLegs, :] = 100

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
    x_bounds[2].min[XrotLegs, :] = -2.5 - 0.2
    x_bounds[2].max[XrotLegs, :] = -2.5 + 0.2
    # Hips sides
    x_bounds[2].min[YrotLegs, :] = -0.1
    x_bounds[2].max[YrotLegs, :] = 0.1

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
    x_bounds[2].min[vZrotRightUpperArm : vYrotRightUpperArm + 1, :] = -100
    x_bounds[2].max[vZrotRightUpperArm : vYrotRightUpperArm + 1, :] = 100
    # Left arm
    x_bounds[2].min[vZrotLeftUpperArm : vYrotLeftUpperArm + 1, :] = -100
    x_bounds[2].max[vZrotLeftUpperArm : vYrotLeftUpperArm + 1, :] = 100

    # Right elbow
    x_bounds[2].min[vZrotRightLowerArm : vYrotRightLowerArm + 1, :] = -100
    x_bounds[2].max[vZrotRightLowerArm : vYrotRightLowerArm + 1, :] = 100
    # Left elbow
    x_bounds[2].min[vZrotRightLowerArm : vYrotLeftLowerArm + 1, :] = -100
    x_bounds[2].max[vZrotRightLowerArm : vYrotLeftLowerArm + 1, :] = 100

    # Hip flexion
    x_bounds[2].min[vXrotLegs, :] = -100
    x_bounds[2].max[vXrotLegs, :] = 100
    # Hip sides
    x_bounds[2].min[vYrotLegs, :] = -100
    x_bounds[2].max[vYrotLegs, :] = 100

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
    x_bounds[3].min[XrotLegs, START] = -2.5 - 0.2
    x_bounds[3].max[XrotLegs, START] = -2.5 + 0.2
    x_bounds[3].min[XrotLegs, MIDDLE] = -2.5 - 0.2
    x_bounds[3].max[XrotLegs, MIDDLE] = 0.2
    x_bounds[3].min[XrotLegs, END] = -0.2
    x_bounds[3].max[XrotLegs, END] = 0.2

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
    x_bounds[3].min[vZrotRightUpperArm : vYrotRightUpperArm + 1, :] = -100
    x_bounds[3].max[vZrotRightUpperArm : vYrotRightUpperArm + 1, :] = 100
    # Left arm
    x_bounds[3].min[vZrotLeftUpperArm : vYrotLeftUpperArm + 1, :] = -100
    x_bounds[3].max[vZrotLeftUpperArm : vYrotLeftUpperArm + 1, :] = 100

    # Right elbow
    x_bounds[3].min[vZrotRightLowerArm : vYrotRightLowerArm + 1, :] = -100
    x_bounds[3].max[vZrotRightLowerArm : vYrotRightLowerArm + 1, :] = 100
    # Left elbow
    x_bounds[3].min[vZrotRightLowerArm : vYrotLeftLowerArm + 1, :] = -100
    x_bounds[3].max[vZrotRightLowerArm : vYrotLeftLowerArm + 1, :] = 100

    # Hip flexion
    x_bounds[3].min[vXrotLegs, :] = -100
    x_bounds[3].max[vXrotLegs, :] = 100
    # Hip sides
    x_bounds[3].min[vYrotLegs, :] = -100
    x_bounds[3].max[vYrotLegs, :] = 100

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
    x_bounds[4].min[YrotRightUpperArm, END] = 2.9 - 0.1
    x_bounds[4].max[YrotRightUpperArm, END] = 2.9 + 0.1
    x_bounds[4].min[ZrotRightUpperArm, END] = -0.1
    x_bounds[4].max[ZrotRightUpperArm, END] = 0.1
    # Left arm
    x_bounds[4].min[YrotLeftUpperArm, END] = -2.9 - 0.1
    x_bounds[4].max[YrotLeftUpperArm, END] = -2.9 + 0.1
    x_bounds[4].min[ZrotLeftUpperArm, END] = -0.1
    x_bounds[4].max[ZrotLeftUpperArm, END] = 0.1

    # Right elbow
    x_bounds[4].min[ZrotRightLowerArm : XrotRightLowerArm + 1, END] = -0.1
    x_bounds[4].max[ZrotRightLowerArm : XrotRightLowerArm + 1, END] = 0.1
    # Left elbow
    x_bounds[4].min[ZrotLeftLowerArm : XrotLeftLowerArm + 1, END] = -0.1
    x_bounds[4].max[ZrotLeftLowerArm : XrotLeftLowerArm + 1, END] = 0.1

    # Hips flexion
    x_bounds[4].min[XrotLegs, :] = -0.4
    x_bounds[4].min[XrotLegs, END] = -0.60
    x_bounds[4].max[XrotLegs, END] = -0.40
    # Hips sides
    x_bounds[4].min[YrotLegs, END] = -0.1
    x_bounds[4].max[YrotLegs, END] = 0.1

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
    x_bounds[4].min[vZrotRightUpperArm : vYrotRightUpperArm + 1, :] = -100
    x_bounds[4].max[vZrotRightUpperArm : vYrotRightUpperArm + 1, :] = 100
    # Left arm
    x_bounds[4].min[vZrotLeftUpperArm : vYrotLeftUpperArm + 1, :] = -100
    x_bounds[4].max[vZrotLeftUpperArm : vYrotLeftUpperArm + 1, :] = 100

    # Right elbow
    x_bounds[4].min[vZrotRightLowerArm : vYrotRightLowerArm + 1, :] = -100
    x_bounds[4].max[vZrotRightLowerArm : vYrotRightLowerArm + 1, :] = 100
    # Left elbow
    x_bounds[4].min[vZrotRightLowerArm : vYrotLeftLowerArm + 1, :] = -100
    x_bounds[4].max[vZrotRightLowerArm : vYrotLeftLowerArm + 1, :] = 100

    # Hip flexion
    x_bounds[4].min[vXrotLegs, :] = -100
    x_bounds[4].max[vXrotLegs, :] = 100
    # Hip sides
    x_bounds[4].min[vYrotLegs, :] = -100
    x_bounds[4].max[vYrotLegs, :] = 100

    # ------------------------------- Initial guesses ------------------------------- #

    x0 = np.vstack((np.zeros((nb_q, 2)), np.zeros((nb_qdot, 2))))
    x1 = np.vstack((np.zeros((nb_q, 2)), np.zeros((nb_qdot, 2))))
    x2 = np.vstack((np.zeros((nb_q, 2)), np.zeros((nb_qdot, 2))))
    x3 = np.vstack((np.zeros((nb_q, 2)), np.zeros((nb_qdot, 2))))
    x4 = np.vstack((np.zeros((nb_q, 2)), np.zeros((nb_qdot, 2))))

    x0[Xrot] = np.array([0, -np.pi / 2])
    x0[Zrot] = np.array([0, 2 * np.pi * num_twists])
    x0[ZrotLeftUpperArm] = -0.75
    x0[ZrotRightUpperArm] = 0.75
    x0[YrotLeftUpperArm, 0] = -2.9
    x0[YrotRightUpperArm, 0] = 2.9
    # x0[YrotLeftUpperArm, 1] = -1.35
    # x0[YrotRightUpperArm, 1] = 1.35
    # x0[XrotLegs, 0] = -.5
    x0[vXrot] = - 4 * np.pi

    x1[Xrot] = np.array([-np.pi / 2, -3 / 4 * np.pi])
    x1[Zrot] = np.array([2 * np.pi * num_twists, 2 * np.pi * num_twists + np.pi])
    # x1[ZrotLeftUpperArm] = -.75
    # x1[ZrotRightUpperArm] = .75
    # x1[YrotLeftUpperArm] = -1.35
    # x1[YrotRightUpperArm] = 1.35
    x1[XrotLegs] = np.array([0, -2.5])

    x2[Xrot] = np.array([-3 / 4 * np.pi, -3 * np.pi])
    x2[Zrot] = np.array([2 * np.pi * num_twists + np.pi, 2 * np.pi * num_twists + np.pi])
    # x2[ZrotLeftUpperArm, 0] = -.75
    # x2[ZrotRightUpperArm, 0] = .75
    # x2[YrotLeftUpperArm, 0] = -1.35
    # x2[YrotRightUpperArm, 0] = 1.35
    x2[XrotLegs, 0] = -2.5

    x3[Xrot] = np.array([-3 * np.pi, -2 * np.pi - 5 / 4 * np.pi])
    x3[Zrot] = np.array([2 * np.pi * num_twists + np.pi, 2 * np.pi * num_twists + 2 * np.pi])
    x3[XrotLegs] = np.array([-2.5, 0])

    x4[Xrot] = np.array([-2 * np.pi - 5 / 4 * np.pi, -4 * np.pi + 0.5])
    x4[Zrot] = np.array([2 * np.pi * num_twists + 2 * np.pi, 2 * np.pi * num_twists + 2 * np.pi])
    x4[XrotLegs] = np.array([0, -0.5])

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

    if WITH_VISUAL_CRITERIA:
        # constraints.add(....)
        print('add constraints')
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
    Prepares and solves an ocp for a 831< with and without visual criteria.
    """

    WITH_VISUAL_CRITERIA = False # True

    biorbd_model_path = "/home/charbie/Documents/Programmation/VisionOCP/models/SoMe.bioMod"
    n_shooting = (40, 100, 100, 100, 40)
    num_twists = 1
    ocp = prepare_ocp(biorbd_model_path, n_shooting=n_shooting, num_twists=num_twists, n_threads=7, WITH_VISUAL_CRITERIA=WITH_VISUAL_CRITERIA)
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
