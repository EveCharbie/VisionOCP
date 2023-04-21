"""
The goal of this program is to get the joint and gaze angles to compare with the ones obtained from the OCP
"""

import numpy as np
import pickle
from IPython import embed
import matplotlib.pyplot as plt
import os
from scipy import signal

import biorbd
import bioviz
from bioptim import (
    BiorbdModel,
    OptimalControlProgram,
    DynamicsList,
    DynamicsFcn,
    BoundsList,
    InitialGuessList,
    ObjectiveList,
    ObjectiveFcn,
    Node,
    Solver,
    InterpolationType,
)


def prepare_optimal_estimation(biorbd_model_path, markers_xsens, x_init_kalman):

    biorbd_model = BiorbdModel(biorbd_model_path)
    n_shooting = markers_xsens.shape[2] - 1
    final_time = n_shooting * 1/200

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(
        ObjectiveFcn.Lagrange.TRACK_MARKERS,
        node=Node.ALL,
        weight=100,
        target=markers_xsens,
    )
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="qddot_joints")

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.JOINTS_ACCELERATION_DRIVEN)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=biorbd_model.bounds_from_ranges(["q", "qdot"]))

    qddot_joints_min, qddot_joints_max, qddot_joints_init = -1000, 1000, 0
    nb_u = biorbd_model.nb_q - biorbd_model.nb_root
    u_bounds = BoundsList()
    u_bounds.add([qddot_joints_min] * nb_u, [qddot_joints_max] * nb_u)

    # Initial guess
    x_init = InitialGuessList()
    x_init.add(x_init_kalman, interpolation=InterpolationType.EACH_FRAME)

    u_init = InitialGuessList()
    u_init.add([qddot_joints_init] * nb_u)

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
    )

def recons_kalman(num_frames, num_markers, markers_xsens, model):
    markersOverFrames = []
    for i in range(num_frames):
        node_segment = []
        for j in range(num_markers):
            node_segment.append(biorbd.NodeSegment(markers_xsens[:, j, i].T))
        markersOverFrames.append(node_segment)

    # Create a Kalman filter structure
    freq = 200  # Hz
    params = biorbd.KalmanParam(freq)
    kalman = biorbd.KalmanReconsMarkers(model, params)

    # Perform the kalman filter for each frame (the first frame is much longer than the next)
    Q = biorbd.GeneralizedCoordinates(model)
    Qdot = biorbd.GeneralizedVelocity(model)
    Qddot = biorbd.GeneralizedAcceleration(model)
    q_recons = np.ndarray((model.nbQ(), len(markersOverFrames)))
    qdot_recons = np.ndarray((model.nbQ(), len(markersOverFrames)))
    for i, targetMarkers in enumerate(markersOverFrames):
        kalman.reconstructFrame(model, targetMarkers, Q, Qdot, Qddot)
        q_recons[:, i] = Q.to_array()
        qdot_recons[:, i] = Qdot.to_array()
    return q_recons, qdot_recons


def recons_ik_trf(markers_xsens, model):
    ik = biorbd.InverseKinematics(model, markers_xsens)
    ik.solve("trf")
    q_recons_wrapped = ik.q
    q_recons = np.unwrap(q_recons_wrapped)
    return q_recons

def smooth_q_recons(q_recons):
    b, a = signal.ellip(4, 0.01, 120, 0.125)  # Filter to be applied: 4th order, 0.125 normalized cutoff frequency
    q_recons_smoothed = np.zeros(np.shape(q_recons))
    for i in range(28):
        q_recons_smoothed[i, :] = signal.filtfilt(b, a, q_recons[i, :], method="gust")
    return q_recons_smoothed


FLAG_SHOW_BIOVIZ = False  # True
RECONSTRUCTION_METHOD = "OCP"  # "OCP", "KALMAN" or "IK_TRF"

move_path = "/home/charbie/disk/Eye-tracking/Results/SoMe/831</"
save_path = "/home/charbie/Documents/Programmation/VisionOCP/Kalman_recons/"

# loop over the files in the folder
for filename in os.listdir(move_path):
    if filename[-24:] == "_eyetracking_metrics.pkl":
        move_filename = move_path + filename
    else:
        continue

    biorbd_model_path = "models/SoMe_reconstruction.bioMod"

    with open(move_filename, "rb") as f:
        data = pickle.load(f)
        subject_name = data["subject_name"]
        gaze_position_temporal_evolution_projected_facing_front_wall = data["gaze_position_temporal_evolution_projected_facing_front_wall"]
        move_orientation = data["move_orientation"]
        Xsens_head_position_calculated = data["Xsens_head_position_calculated"]
        eye_position = data["eye_position"]
        gaze_orientation = data["gaze_orientation"]
        EulAngles_head_global = data["EulAngles_head_global"]
        EulAngles_neck = data["EulAngles_neck"]
        eye_angles = data["eye_angles"]
        Xsens_orthogonal_thorax_position = data["Xsens_orthogonal_thorax_position"]
        Xsens_orthogonal_head_position = data["Xsens_orthogonal_head_position"]
        Xsens_position_no_level_CoM_corrected_rotated_per_move = data["Xsens_position_no_level_CoM_corrected_rotated_per_move"]


    # get markers position from the biorbd model
    model = biorbd.Model(biorbd_model_path)
    markers_tuple = model.markers(np.zeros((model.nbQ())))
    num_markers = len(markers_tuple)


    # get joint positions from the xsens model
    num_joints = int(Xsens_position_no_level_CoM_corrected_rotated_per_move.shape[1]/3)
    num_frames = Xsens_position_no_level_CoM_corrected_rotated_per_move.shape[0]
    JCS_xsens = np.zeros((3, num_joints, num_frames))
    for j in range(num_frames):
        for i in range(num_joints):
            JCS_xsens[:, i, j] = Xsens_position_no_level_CoM_corrected_rotated_per_move[j, i*3:(i+1)*3]


    # Put markers in the same order
    markers_xsens = np.zeros((3, num_markers, num_frames))
    markers_xsens[:, 0, :] = JCS_xsens[:, 2, :]  # L3_marker # T12_marker
    markers_xsens[:, 1, :] = JCS_xsens[:, 6, :]  # C1_marker
    markers_xsens[:, 2, :] = np.transpose(eye_position)  # eyes_vect_start
    markers_xsens[:, 3, :] = np.transpose(eye_position + (gaze_orientation - eye_position)/10)   # eyes_vect_end
    markers_xsens[:, 4, :] = JCS_xsens[:, 8, :]  # RightUpperArm_marker
    markers_xsens[:, 5, :] = JCS_xsens[:, 9, :]  # RightForeArm_marker
    markers_xsens[:, 6, :] = JCS_xsens[:, 10, :]  # RightHand_marker
    markers_xsens[:, 7, :] = JCS_xsens[:, 12, :]  # LeftUpperArm_marker
    markers_xsens[:, 8, :] = JCS_xsens[:, 13, :]  # LeftForeArm_marker
    markers_xsens[:, 9, :] = JCS_xsens[:, 14, :]  # LeftHand_marker
    markers_xsens[:, 10, :] = JCS_xsens[:, 0, :]  # Hip_marker
    markers_xsens[:, 11, :] = JCS_xsens[:, 15, :]  # RightUpperLeg_marker
    markers_xsens[:, 12, :] = JCS_xsens[:, 19, :]  # LeftUpperLeg_marker
    markers_xsens[:, 13, :] = JCS_xsens[:, 16, :]  # RightLowerLeg_marker
    markers_xsens[:, 14, :] = JCS_xsens[:, 20, :]  # LeftLowerLeg_marker
    markers_xsens[:, 15, :] = JCS_xsens[:, 17, :]  # RightFoot_marker
    markers_xsens[:, 16, :] = JCS_xsens[:, 21, :]  # LeftFoot_marker

    if RECONSTRUCTION_METHOD == "KALMAN":
        q_recons, _ = recons_kalman(num_frames, num_markers, markers_xsens, model)

    elif RECONSTRUCTION_METHOD == "IK_TRF":
        q_recons = recons_ik_trf(markers_xsens, model)

    elif RECONSTRUCTION_METHOD == "OCP":
        q_init, qdot_init = recons_kalman(num_frames, num_markers, markers_xsens, model)
        q_init_smoothed = smooth_q_recons(q_init)
        x_init_kalman = np.vstack((q_init_smoothed, qdot_init))

        ocp = prepare_optimal_estimation(biorbd_model_path, markers_xsens, x_init_kalman)
        solver = Solver.IPOPT(show_online_optim=True, show_options=dict(show_bounds=True))
        solver.set_linear_solver("ma57")
        solver.set_maximum_iterations(10000)
        solver.set_convergence_tolerance(1e-4)
        sol = ocp.solve(solver)
        q_recons = sol.states["q"]

    if FLAG_SHOW_BIOVIZ:
        print(filename)
        b = bioviz.Viz(biorbd_model_path)
        b.load_movement(q_recons)
        b.load_experimental_markers(markers_xsens)
        b.exec()



    save_name = save_path + filename[:-24] + 'kalman_recons'
    fig, axs = plt.subplots(7, 4, figsize=(20, 10))
    axs = axs.ravel()
    for i in range(28):
        axs[i].plot(q_recons_smoothed[i, :], 'b')
        axs[i].plot(q_recons[i, :], '--r')
        axs[i].set_title(f"{model.nameDof()[i].to_string()}")
    plt.tight_layout()
    plt.suptitle("Kalman reconstruction (r = Q_recons, b = Q_recons_smoothed)")
    plt.savefig(save_name + ".png", dpi=300)
    # plt.show()