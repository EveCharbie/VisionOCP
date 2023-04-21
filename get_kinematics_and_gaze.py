"""
The goal of this program is to get the joint and gaze angles to compare with the ones obtained from the OCP
"""

import numpy as np
import pickle
from IPython import embed
import biorbd
import bioviz
import matplotlib.pyplot as plt
import os
from scipy import signal

FLAG_SHOW_BIOVIZ = False # True # False

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
    for i, targetMarkers in enumerate(markersOverFrames):
        kalman.reconstructFrame(model, targetMarkers, Q, Qdot, Qddot)
        q_recons[:, i] = Q.to_array()


    # # create a 3D plot of the markers
    # colors = plt.cm.get_cmap('viridis', num_markers)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(markers_xsens[0, 0, 0], markers_xsens[1, 0, 0], markers_xsens[2, 0, 0], c=colors(0/16), marker='o', label='Pelvis_marker')
    # ax.scatter(markers_xsens[0, 1, 0], markers_xsens[1, 1, 0], markers_xsens[2, 1, 0], c=colors(1/16), marker='o', label='T12_marker')
    # ax.scatter(markers_xsens[0, 2, 0], markers_xsens[1, 2, 0], markers_xsens[2, 2, 0], c=colors(2/16), marker='o', label='C1_marker')
    # ax.scatter(markers_xsens[0, 3, 0], markers_xsens[1, 3, 0], markers_xsens[2, 3, 0], c=colors(3/16), marker='o', label='eyes_vect_start')
    # ax.scatter(markers_xsens[0, 4, 0], markers_xsens[1, 4, 0], markers_xsens[2, 4, 0], c=colors(4/16), marker='o', label='eyes_vect_end')
    # ax.scatter(markers_xsens[0, 5, 0], markers_xsens[1, 5, 0], markers_xsens[2, 5, 0], c=colors(5/16), marker='o', label='RightUpperArm_marker')
    # ax.scatter(markers_xsens[0, 6, 0], markers_xsens[1, 6, 0], markers_xsens[2, 6, 0], c=colors(6/16), marker='o', label='RightForeArm_marker')
    # ax.scatter(markers_xsens[0, 7, 0], markers_xsens[1, 7, 0], markers_xsens[2, 7, 0], c=colors(7/16), marker='o', label='RightHand_marker')
    # ax.scatter(markers_xsens[0, 8, 0], markers_xsens[1, 8, 0], markers_xsens[2, 8, 0], c=colors(8/16), marker='o', label='LeftUpperArm_marker')
    # ax.scatter(markers_xsens[0, 9, 0], markers_xsens[1, 9, 0], markers_xsens[2, 9, 0], c=colors(9/16), marker='o', label='LeftForeArm_marker')
    # ax.scatter(markers_xsens[0, 10, 0], markers_xsens[1, 10, 0], markers_xsens[2, 10, 0], c=colors(10/16), marker='o', label='RightHand_marker')
    # ax.scatter(markers_xsens[0, 11, 0], markers_xsens[1, 11, 0], markers_xsens[2, 11, 0], c=colors(11/16), marker='o', label='RightUpperLeg_marker')
    # ax.scatter(markers_xsens[0, 12, 0], markers_xsens[1, 12, 0], markers_xsens[2, 12, 0], c=colors(12/16), marker='o', label='LeftUpperLeg_marker')
    # ax.scatter(markers_xsens[0, 13, 0], markers_xsens[1, 13, 0], markers_xsens[2, 13, 0], c=colors(13/16), marker='o', label='RightLowerLeg_marker')
    # ax.scatter(markers_xsens[0, 14, 0], markers_xsens[1, 14, 0], markers_xsens[2, 14, 0], c=colors(14/16), marker='o', label='LeftLowerLeg_marker')
    # ax.scatter(markers_xsens[0, 15, 0], markers_xsens[1, 15, 0], markers_xsens[2, 15, 0], c=colors(15/16), marker='o', label='RightFoot_marker')
    # ax.scatter(markers_xsens[0, 16, 0], markers_xsens[1, 16, 0], markers_xsens[2, 16, 0], c=colors(16/16), marker='o', label='LeftFoot_marker')
    # ax.set_xlabel('X [m]')
    # ax.set_ylabel('Y [m]')
    # ax.set_zlabel('Z [m]')
    # ax.legend()
    # plt.savefig("markers_and_joints.png")
    # # plt.show()

    if FLAG_SHOW_BIOVIZ:
        print(filename)
        b = bioviz.Viz(biorbd_model_path)
        b.load_movement(q_recons)
        b.load_experimental_markers(markers_xsens)
        b.exec()

    # Smooth the q_recons
    b, a = signal.ellip(4, 0.01, 120, 0.125)  # Filter to be applied: 4th order, 0.125 normalized cutoff frequency
    q_recons_smoothed = np.zeros(np.shape(q_recons))
    for i in range(28):
        q_recons_smoothed[i, :] = signal.filtfilt(b, a, q_recons[i, :], method="gust")

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