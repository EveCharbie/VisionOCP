"""
The goal of this program is to compare the athlete kinematics with the optimal kinematics with and without visual criteria.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
import pickle
import sys
sys.path.append("/home/charbie/Documents/Programmation/BiorbdOptim")
import bioptim
import biorbd
from IPython import embed
import os

sys.path.append("/home/charbie/Documents/Programmation/biorbd-viz")
import bioviz

ACROBATICS = ["42", "831"]


biorbd_model_path_831_with_visual_criteria = "models/SoMe_with_visual_criteria.bioMod"
model_831_with_visual_criteria = biorbd.Model(biorbd_model_path_831_with_visual_criteria)
biorbd_model_path_831 = "models/SoMe.bioMod"
model_831 = biorbd.Model(biorbd_model_path_831)
biorbd_model_path_831_both = "models/SoMe_with_and_without_visual_criteria.bioMod"
n_shooting_831 = (40, 40, 40, 40, 40, 40)
biorbd_model_path_42_with_visual_criteria = "models/SoMe_42_with_visual_criteria.bioMod"
model_42_with_visual_criteria = biorbd.Model(biorbd_model_path_42_with_visual_criteria)
biorbd_model_path_42 = "models/SoMe_42.bioMod"
model_42 = biorbd.Model(biorbd_model_path_42)
biorbd_model_path_42_both = "models/SoMe_42_with_and_without_visual_criteria.bioMod"
n_shooting_42 = (100, 40)


# ---------------------------------------- Load optimal kinematics ---------------------------------------- #
weights = [0.0, 0.25, 0.5, 1.0, 1.75, 2.0]  # [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
listdir = os.listdir("Solutions/")
solution_file_names = [current_dir for current_dir in listdir if ".pkl" in current_dir and current_dir[:2] != "q_"]

# # Branch Multi-start, commit fd1886ca828af2fa56b93e8221a0840f88b2058d, May 4th 2023
# for i_weight, weight in enumerate(weights):
#
#     for file_name in solution_file_names:
#         if "42" in file_name and str(weight).replace('.', 'p') in file_name:
#             file_name_42 = file_name
#
#     with open("Solutions/" + file_name_42, "rb") as f:
#         data = pickle.load(f)
#         sol = data[0]
#         q_per_phase = data[1]
#         qs = data[2]
#         qdots = data[3]
#         qddots = data[4]
#         time_parameters = data[5]
#         q_reintegrated = data[6]
#         qdot_reintegrated = data[7]
#         time_vector = data[8]
#         interpolated_states = data[9]
#
#     data_without_sol = {"q_per_phase": q_per_phase,
#                         "qs": qs,
#                         "qdots": qdots,
#                         "qddots": qddots,
#                         "time_parameters": time_parameters,
#                         "q_reintegrated": q_reintegrated,
#                         "qdot_reintegrated": qdot_reintegrated,
#                         "time_vector": time_vector,
#                         "interpolated_states": interpolated_states}
#
#     with open("Solutions/q_" + file_name_42, "wb") as f:
#         pickle.dump(data_without_sol, f)


# # Branch Total_time_constraint, commit 7e3a4b17531e359138737714216eda41a3b316f7, October 26th 2023
# for i_weight, weight in enumerate(weights):
#
#     if i_weight == 0:
#         continue
#     for file_name in solution_file_names:
#         if "831" in file_name and str(weight).replace('.', 'p') in file_name:
#             file_name_831 = file_name
#
#     with open("Solutions/" + file_name_831, "rb") as f:
#         data = pickle.load(f)
#         sol = data[0]
#         q_per_phase = data[1]
#         qs = data[2]
#         qdots = data[3]
#         qddots = data[4]
#         time_parameters = data[5]
#         q_reintegrated = data[6]
#         qdot_reintegrated = data[7]
#         time_vector = data[8]
#         interpolated_states = data[9]
#
#     data_without_sol = {"q_per_phase": q_per_phase,
#                         "qs": qs,
#                         "qdots": qdots,
#                         "qddots": qddots,
#                         "time_parameters": time_parameters,
#                         "q_reintegrated": q_reintegrated,
#                         "qdot_reintegrated": qdot_reintegrated,
#                         "time_vector": time_vector,
#                         "interpolated_states": interpolated_states}
#
#     with open("Solutions/q_" + file_name_831, "wb") as f:
#         pickle.dump(data_without_sol, f)


# # Branch release_3_2, commit 17e3628b1e4fdf5aa09b47e30f13abae2e1ecae4, November 1st 2023
# weight = 0.0
# for file_name in solution_file_names:
#     if "831" in file_name and str(weight).replace('.', 'p') in file_name:
#         file_name_831 = file_name
#
# with open("Solutions/" + file_name_831, "rb") as f:
#     data = pickle.load(f)
#     sol = data[0]
#     q_per_phase = data[1]
#     qs = data[2]
#     qdots = data[3]
#     qddots = data[4]
#     time_parameters = data[5]
#     q_reintegrated = data[6]
#     qdot_reintegrated = data[7]
#     time_vector = data[8]
#     interpolated_states = data[9]
#
# data_without_sol = {"q_per_phase": q_per_phase,
#                     "qs": qs,
#                     "qdots": qdots,
#                     "qddots": qddots,
#                     "time_parameters": time_parameters,
#                     "q_reintegrated": q_reintegrated,
#                     "qdot_reintegrated": qdot_reintegrated,
#                     "time_vector": time_vector,
#                     "interpolated_states": interpolated_states}
#
# with open("Solutions/q_" + file_name_831, "wb") as f:
#     pickle.dump(data_without_sol, f)


def find_eye_markers_index(model):
    marker_names = model.markerNames()
    for i_marker, marker in enumerate(marker_names):
        if marker.to_string() == "eyes_vect_start":
            eyes_start_idx = i_marker
        elif marker.to_string() == "eyes_vect_end":
            eyes_end_idx = i_marker
    return eyes_start_idx, eyes_end_idx


def custom_trampoline_bed_in_peripheral_vision(model, q):
    """
    This function aims to encourage the avatar to keep the trampoline bed in his peripheral vision.
    It is adapted from the OCP.
    """

    a = 1.07  # Trampoline with/2
    b = 2.14  # Trampoline length/2
    n = 6  # order of the polynomial for the trampoline bed rectangle equation

    marker_names = [model.markerNames()[i].to_string() for i in range(model.nbMarkers())]

    val = np.zeros((q.shape[1]))
    for i_node in range(q.shape[1]):
        # Get the gaze vector
        eyes_vect_start_marker_idx, eyes_vect_end_marker_idx = find_eye_markers_index(model)
        gaze_vector = model.markers(q[:, i_node])[eyes_vect_end_marker_idx].to_array() - model.markers(q[:, i_node])[eyes_vect_start_marker_idx].to_array()

        point_in_the_plane = np.array([1, 2, -0.83])
        vector_normal_to_the_plane = np.array([0, 0, 1])
        obj = 0
        for i_r in range(11):
            for i_th in range(10):

                # Get this vector from the vision cone
                marker_idx = marker_names.index(f'cone_approx_{i_r}_{i_th}')
                vector_origin = model.markers(q[:, i_node])[eyes_vect_start_marker_idx].to_array()
                vector_end = model.markers(q[:, i_node])[marker_idx].to_array()
                vector = vector_end - vector_origin

                # Get the intersection between the vector and the trampoline plane
                t = (np.dot(point_in_the_plane, vector_normal_to_the_plane) - np.dot(vector_normal_to_the_plane, vector_origin)) / np.dot(
                    vector, vector_normal_to_the_plane
                )
                point_projection = vector_origin + vector * np.abs(t)

                # Determine if the point is inside the trampoline bed
                # Rectangle equation : (x/a)**n + (y/b)**n = 1
                # The function is convoluted with tanh to make it:
                # 1. Continuous
                # 2. Not encourage to look to the middle of the trampoline bed
                # 3. Largely penalized when outside the trampoline bed
                # 4. Equaly penalized when looking upward
                obj += np.tanh(((point_projection[0]/a)**n + (point_projection[1]/b)**n) - 1) + 1

        if gaze_vector[2] > -0.01:
            val[i_node] = 2*10*11
        elif np.abs(gaze_vector[0]/gaze_vector[2]) > np.tan(3*np.pi/8):
            val[i_node] = 2*10*11
        elif np.abs(gaze_vector[1]/gaze_vector[2]) > np.tan(3*np.pi/8):
            val[i_node] = 2*10*11
        else:
            val[i_node] = obj

    return val

def minimize_segment_velocity(model, q , qdot):
    """
    Track the head velocity.
    It is adapted from the OCP.
    """
    segment_angular_velocity = np.zeros((3, q.shape[1]))
    for i_node in range(q.shape[1]):
        segment_angular_velocity[:, i_node] = model.segmentAngularVelocity(q[:, i_node], qdot[:, i_node], 1).to_array()
    return segment_angular_velocity

def track_vector_orientations_from_markers(model, q):
    """
    Aligns the gaze vector with the vector joining the trampoline target with the eyes.
    It is adapted from the OCP.
    """

    marker_names = [model.markerNames()[i].to_string() for i in range(model.nbMarkers())]
    eyes_vect_start = marker_names.index('eyes_vect_start')
    eyes_vect_end = marker_names.index('eyes_vect_end')
    fixation_front = marker_names.index('fixation_front')

    angle = np.zeros((q.shape[1]))
    for i_node in range(q.shape[1]):
        eyes_vect_start_position = model.marker(q[:, i_node], eyes_vect_start).to_array()
        eyes_vect_end_position = model.marker(q[:, i_node], eyes_vect_end).to_array()
        fixation_front_position = model.marker(q[:, i_node], fixation_front).to_array()

        vector_0 = eyes_vect_end_position - eyes_vect_start_position
        vector_1 = fixation_front_position - eyes_vect_start_position
        cross_prod = np.cross(vector_0, vector_1)
        cross_prod_norm = np.sqrt(cross_prod[0] ** 2 + cross_prod[1] ** 2 + cross_prod[2] ** 2)
        angle[i_node] = np.arctan2(cross_prod_norm, np.dot(vector_0, vector_1))

    return angle

def hand_leg_distance(model, q):
    marker_names = [model.markerNames()[i].to_string() for i in range(model.nbMarkers())]
    right_hand = marker_names.index('MidMainD')
    left_hand = marker_names.index('MidMainG')
    right_leg = marker_names.index('CibleMainD')
    left_leg = marker_names.index('CibleMainG')
    right_hand_marker = model.markers(q)[right_hand].to_array()
    left_hand_marker = model.markers(q)[left_hand].to_array()
    right_leg_marker = model.markers(q)[right_leg].to_array()
    left_leg_marker = model.markers(q)[left_leg].to_array()
    distance = (right_hand_marker - right_leg_marker) ** 2 + (left_hand_marker - left_leg_marker) ** 2
    return distance


# ---------------------------------------- Plot comparison ---------------------------------------- #
colors = [cm.magma(i/(len(weights)-1)) for i in range(len(weights))]
fig_cost, axs_cost = plt.subplots(6, 2, figsize=(10, 16))

# 42 plots
detailed_cost_function_42 = {"qddot_joints": {}, "qddot_joints_derivative": {}, "time": {}, "shoulders_dof": {}, "final_tilt": {}, "peripheral": {}, "spotting": {}, "self_motion_detection": {}, "trampo_fixation": {}, "neck": {}, "eyes": {}}
fig_root, axs_root = plt.subplots(2, 3, figsize=(15, 6))
fig_joints, axs_joints = plt.subplots(2, 2, figsize=(10, 6))
fig_somersault_twist, axs_somersault_twist = plt.subplots(1, 1, figsize=(5, 3))
for i_weight, weight in enumerate(weights):

    for file_name in solution_file_names:
        if "42" in file_name and str(weight).replace('.', 'p') in file_name:
            file_name_42 = file_name

    with open("Solutions/q_" + file_name_42, "rb") as f:
        data = pickle.load(f)
        q_per_phase = data["q_per_phase"]
        qs = data["qs"]
        qdots = data["qdots"]
        qddots = data["qddots"]
        time_parameters = data["time_parameters"]
        q_reintegrated = data["q_reintegrated"]
        qdot_reintegrated = data["qdot_reintegrated"]
        time_vector = data["time_vector"]
        interpolated_states = data["interpolated_states"]

    axs_root[0, 0].plot(time_vector, qs[3, :] * 180/np.pi, color=colors[i_weight], label=str(weight))
    axs_root[0, 1].plot(time_vector, qs[4, :] * 180/np.pi, color=colors[i_weight])
    axs_root[0, 2].plot(time_vector, qs[5, :] * 180/np.pi, color=colors[i_weight])
    axs_root[1, 0].plot(time_vector, qdots[3, :] * 180/np.pi, color=colors[i_weight], label=str(weight))
    axs_root[1, 1].plot(time_vector, qdots[4, :] * 180/np.pi, color=colors[i_weight])
    axs_root[1, 2].plot(time_vector, qdots[5, :] * 180/np.pi, color=colors[i_weight])
    axs_somersault_twist.plot(qs[3, :] * 180/np.pi, qs[5, :] * 180/np.pi, color=colors[i_weight])

    if weight == 0.0:
        right_arm_indices = [6, 7]
        left_arm_indices = [8, 9]
    else:
        right_arm_indices = [10, 11]
        left_arm_indices = [12, 13]

    # Right arm
    axs_joints[0, 1].plot(time_vector, qs[right_arm_indices[0], :] * 180/np.pi, color=colors[i_weight], label=str(weight))
    axs_joints[1, 1].plot(time_vector, qs[right_arm_indices[1], :] * 180/np.pi, color=colors[i_weight])

    # Left arm
    axs_joints[0, 0].plot(time_vector, -qs[left_arm_indices[0], :] * 180/np.pi, color=colors[i_weight])
    axs_joints[1, 0].plot(time_vector, -qs[left_arm_indices[1], :] * 180/np.pi, color=colors[i_weight])

    # Cost function
    dt_lagrange = time_vector[1:] - time_vector[:-1]
    model = biorbd.Model(biorbd_model_path_42_with_visual_criteria)
    if weight == 0:
        qs_tempo = np.zeros(qs.shape)
        qs_tempo[:, :] = qs[:, :]
        qs = np.zeros((14, qs_tempo.shape[1]))
        qs[:6, :] = qs_tempo[:6, :]
        qs[10:, :] = qs_tempo[6:, :]
        qdots_tempo = np.zeros(qdots.shape)
        qdots_tempo[:, :] = qdots[:, :]
        qdots = np.zeros((14, qdots_tempo.shape[1]))
        qdots[:6, :] = qdots_tempo[:6, :]
        qdots[10:, :] = qdots_tempo[6:, :]
        for j in range(6):
            axs_cost[j, 0].set_xlim(0, time_vector[-1])

    peripheral = custom_trampoline_bed_in_peripheral_vision(model, qs)
    axs_cost[0, 0].plot(time_vector, peripheral, color=colors[i_weight], label=str(weight))
    detailed_cost_function_42["peripheral"][str(weight)] = peripheral[:-1]**2 * dt_lagrange

    head_velocity = minimize_segment_velocity(model, qs, qdots) * 180/np.pi
    head_velocity_norm = head_velocity[0, :]**2 + head_velocity[1, :]**2 + head_velocity[2, :]**2
    axs_cost[1, 0].plot(time_vector, head_velocity_norm, color=colors[i_weight])
    detailed_cost_function_42["spotting"][str(weight)] = head_velocity_norm[:-1] * dt_lagrange

    self_motion_detection = qdots[8:10, :] * 180/np.pi
    self_motion_detection_norm = self_motion_detection[0, :]**2 + self_motion_detection[1, :]**2
    axs_cost[2, 0].plot(time_vector, self_motion_detection_norm, color=colors[i_weight])
    detailed_cost_function_42["self_motion_detection"][str(weight)] = self_motion_detection_norm[:-1] * dt_lagrange

    angle_quiet_eye = track_vector_orientations_from_markers(model, qs) * 180/np.pi
    axs_cost[3, 0].plot(time_vector, angle_quiet_eye, color=colors[i_weight])
    detailed_cost_function_42["trampo_fixation"][str(weight)] = angle_quiet_eye[:-1]**2 * dt_lagrange

    neck_rotations = qs[6:8, :] * 180/np.pi
    neck_rotations_norm = neck_rotations[0, :]**2 + neck_rotations[1, :]**2
    axs_cost[4, 0].plot(time_vector, neck_rotations_norm, color=colors[i_weight])
    detailed_cost_function_42["neck"][str(weight)] = neck_rotations_norm[:-1] * dt_lagrange

    eyes_rotations = qs[8:10, :] * 180/np.pi
    eyes_rotations_norm = eyes_rotations[0, :]**2 + eyes_rotations[1, :]**2
    axs_cost[5, 0].plot(time_vector, eyes_rotations_norm, color=colors[i_weight])
    detailed_cost_function_42["eyes"][str(weight)] = eyes_rotations_norm[:-1] * dt_lagrange

    # Kinematic costs
    detailed_cost_function_42["qddot_joints"][str(weight)] = np.sum(qddots[:, :-1]**2, axis=0) * dt_lagrange
    detailed_cost_function_42["qddot_joints_derivative"][str(weight)] = np.sum((qddots[:, 1:-1] - qddots[:, :-2])**2, axis=0) * dt_lagrange[-1]
    detailed_cost_function_42["time"][str(weight)] = time_vector[-1]
    detailed_cost_function_42["shoulders_dof"][str(weight)] = (qs[11, :-1]**2 + qs[13, :-1]**2) * dt_lagrange
    detailed_cost_function_42["final_tilt"][str(weight)] = qs[5, -1]**2

current_time = 0
for i, time in enumerate(time_parameters[:-1]):
    current_time += time
    for j in range(6):
        axs_cost[j, 0].plot([current_time, current_time], [-250, 1e+8], '-k', alpha=0.3, linewidth=0.5)
axs_cost[0, 0].set_ylim(-10, 250)
axs_cost[1, 0].set_ylim(-100, 3.5e+6)
axs_cost[2, 0].set_ylim(-100, 8e+7)
axs_cost[3, 0].set_ylim(-10, 200)
axs_cost[4, 0].set_ylim(-100, 10000)
axs_cost[5, 0].set_ylim(-10, 1500)

# show legend below figure
axs_root[0, 0].legend(bbox_to_anchor=[3.7, 1.0], frameon=False)
axs_joints[0, 1].legend(bbox_to_anchor=[1.1, 0.5], frameon=False)
fig_joints.subplots_adjust(hspace=0.35, right=0.85)

axs_joints[0, 1].set_title("Change in elevation plane R")
axs_joints[1, 1].set_title("Elevation R")
axs_joints[0, 0].set_title("Change in elevation plane L")
axs_joints[1, 0].set_title("Elevation L")

axs_cost[0, 0].set_ylabel("Peripheral vision \n", fontsize=13)
axs_cost[1, 0].set_ylabel("Spotting\n", fontsize=13)
axs_cost[2, 0].set_ylabel("Eyes velocity\n", fontsize=13)
axs_cost[3, 0].set_ylabel("Trampo fixation\n", fontsize=13)
axs_cost[4, 0].set_ylabel("Neck angles\n", fontsize=13)
axs_cost[5, 0].set_ylabel("Eyes angles\n", fontsize=13)
axs_cost[5, 0].set_xlabel("Time [s]", fontsize=15)

axs_somersault_twist = plt.gca()
axs_somersault_twist.set_xlim(axs_somersault_twist.get_xlim()[::-1])
axs_somersault_twist.set_xlabel("Somersault [°]", fontsize=15)
axs_somersault_twist.set_ylabel("Twist [°]", fontsize=15)
fig_somersault_twist.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)

fig_root.savefig("Graphs/compare_42_root.png", dpi=300)
fig_joints.savefig("Graphs/compare_42_dofs.png", dpi=300)
fig_somersault_twist.savefig("Graphs/compare_42_somersault_twist.png", dpi=300)



# 831 plots
detailed_cost_function_831 = {"qddot_joints": {},
                          "qddot_joints_derivative": {},
                          "time": {},
                          "shoulders_dof": {},
                          "final_tilt": {},
                          "peripheral": {},
                          "spotting": {},
                          "self_motion_detection": {},
                          "trampo_fixation": {},
                          "neck": {},
                          "eyes": {},
                          "superimpose_markers": {},
                          "elbow_dof": {},
                          "arm_dof": {},
                          "Xrot_legs_dof": {},
                          "wobbling": {},
                          }

fig_root, axs_root = plt.subplots(2, 3, figsize=(15, 6))
fig_joints, axs_joints = plt.subplots(2, 3, figsize=(15, 6))
fig_somersault_twist, axs_somersault_twist = plt.subplots(1, 1, figsize=(5, 3))
for i_weight, weight in enumerate(weights):

    for file_name in solution_file_names:
        if "831" in file_name and str(weight).replace('.', 'p') in file_name:
            file_name_831 = file_name

    with open("Solutions/q_" + file_name_831, "rb") as f:
        data = pickle.load(f)
        q_per_phase = data["q_per_phase"]
        qs = data["qs"]
        qdots = data["qdots"]
        qddots = data["qddots"]
        time_parameters = data["time_parameters"]
        q_reintegrated = data["q_reintegrated"]
        qdot_reintegrated = data["qdot_reintegrated"]
        time_vector = data["time_vector"]
        interpolated_states = data["interpolated_states"]


    axs_root[0, 0].plot(time_vector, qs[3, :] * 180/np.pi, color=colors[i_weight], label=str(weight))
    axs_root[0, 1].plot(time_vector, qs[4, :] * 180/np.pi, color=colors[i_weight])
    axs_root[0, 2].plot(time_vector, qs[5, :] * 180/np.pi, color=colors[i_weight])
    axs_root[1, 0].plot(time_vector, qdots[3, :] * 180/np.pi, color=colors[i_weight], label=str(weight))
    axs_root[1, 1].plot(time_vector, qdots[4, :] * 180/np.pi, color=colors[i_weight])
    axs_root[1, 2].plot(time_vector, qdots[5, :] * 180/np.pi, color=colors[i_weight])
    axs_somersault_twist.plot(qs[3, :] * 180/np.pi, qs[5, :] * 180/np.pi, color=colors[i_weight])

    if weight == 0.0:
        right_arm_indices = [6, 7]
        left_arm_indices = [10, 11]
        hips_indices = [14, 15]
    else:
        right_arm_indices = [10, 11]
        left_arm_indices = [14, 15]
        hips_indices = [18, 19]

    # Right arm
    axs_joints[0, 1].plot(time_vector, qs[right_arm_indices[0], :] * 180/np.pi, color=colors[i_weight], label=str(weight))
    axs_joints[1, 1].plot(time_vector, qs[right_arm_indices[1], :] * 180/np.pi, color=colors[i_weight])

    # Left arm
    axs_joints[0, 0].plot(time_vector, -qs[left_arm_indices[0], :] * 180/np.pi, color=colors[i_weight])
    axs_joints[1, 0].plot(time_vector, -qs[left_arm_indices[1], :] * 180/np.pi, color=colors[i_weight])

    # Hips
    axs_joints[0, 2].plot(time_vector, -qs[hips_indices[0], :] * 180/np.pi, color=colors[i_weight])
    axs_joints[1, 2].plot(time_vector, qs[hips_indices[1], :] * 180/np.pi, color=colors[i_weight])

    # Cost function
    dt_lagrange = time_vector[1:] - time_vector[:-1]
    model = biorbd.Model(biorbd_model_path_831_with_visual_criteria)
    if weight == 0:
        qs_tempo = np.zeros(qs.shape)
        qs_tempo[:, :] = qs[:, :]
        qs = np.zeros((20, qs_tempo.shape[1]))
        qs[:6, :] = qs_tempo[:6, :]
        qs[10:, :] = qs_tempo[6:, :]
        qdots_tempo = np.zeros(qdots.shape)
        qdots_tempo[:, :] = qdots[:, :]
        qdots = np.zeros((20, qdots_tempo.shape[1]))
        qdots[:6, :] = qdots_tempo[:6, :]
        qdots[10:, :] = qdots_tempo[6:, :]
        for j in range(6):
            axs_cost[j, 1].set_xlim(0, time_vector[-1])

    peripheral = custom_trampoline_bed_in_peripheral_vision(model, qs)
    axs_cost[0, 1].plot(time_vector, peripheral, color=colors[i_weight], label=str(weight))
    detailed_cost_function_831["peripheral"][str(weight)] = peripheral[:-1]**2 * dt_lagrange

    head_velocity = minimize_segment_velocity(model, qs, qdots) * 180/np.pi
    head_velocity_norm = head_velocity[0, :]**2 + head_velocity[1, :]**2 + head_velocity[2, :]**2
    axs_cost[1, 1].plot(time_vector, head_velocity_norm, color=colors[i_weight])
    detailed_cost_function_831["spotting"][str(weight)] = head_velocity_norm[:-1] * dt_lagrange

    self_motion_detection = qdots[8:10, :] * 180/np.pi
    self_motion_detection_norm = self_motion_detection[0, :]**2 + self_motion_detection[1, :]**2
    axs_cost[2, 1].plot(time_vector, self_motion_detection_norm, color=colors[i_weight])
    detailed_cost_function_831["self_motion_detection"][str(weight)] = self_motion_detection_norm[:-1] * dt_lagrange

    angle_quiet_eye = track_vector_orientations_from_markers(model, qs) * 180/np.pi
    axs_cost[3, 1].plot(time_vector, angle_quiet_eye, color=colors[i_weight])
    detailed_cost_function_831["trampo_fixation"][str(weight)] = angle_quiet_eye[:-1]**2 * dt_lagrange

    neck_rotations = qs[6:8, :] * 180/np.pi
    neck_rotations_norm = neck_rotations[0, :]**2 + neck_rotations[1, :]**2
    axs_cost[4, 1].plot(time_vector, neck_rotations_norm, color=colors[i_weight])
    detailed_cost_function_831["neck"][str(weight)] = neck_rotations_norm[:-1] * dt_lagrange

    eyes_rotations = qs[8:10, :] * 180/np.pi
    eyes_rotations_norm = eyes_rotations[0, :]**2 + eyes_rotations[1, :]**2
    axs_cost[5, 1].plot(time_vector, eyes_rotations_norm, color=colors[i_weight])
    detailed_cost_function_831["eyes"][str(weight)] = eyes_rotations_norm[:-1] * dt_lagrange

    # Kinematic costs
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
    arm_dofs = [ZrotRightUpperArm, YrotRightUpperArm, ZrotRightLowerArm, XrotRightLowerArm, ZrotLeftUpperArm, YrotLeftUpperArm, ZrotLeftLowerArm, XrotLeftLowerArm]
    shoulder_dofs = [ZrotRightUpperArm, YrotRightUpperArm, ZrotLeftUpperArm, YrotLeftUpperArm]
    elbow_dofs = [ZrotRightLowerArm, XrotRightLowerArm, ZrotLeftLowerArm, XrotLeftLowerArm]

    detailed_cost_function_831["qddot_joints"][str(weight)] = np.sum(qddots[:, :-1]**2, axis=0) * dt_lagrange
    detailed_cost_function_831["qddot_joints_derivative"][str(weight)] = np.sum((qddots[:, 1:-1] - qddots[:, :-2])**2, axis=0) * dt_lagrange[-1]
    detailed_cost_function_831["time"][str(weight)] = time_vector[-1]
    detailed_cost_function_831["superimpose_markers"][str(weight)] = hand_leg_distance(model_831_with_visual_criteria, qs[:, -1])
    detailed_cost_function_831["elbow_dof"][str(weight)] = np.sum(qs[elbow_dofs, :-1]**2, axis=0) * dt_lagrange
    detailed_cost_function_831["arm_dof"][str(weight)] = np.sum(qs[arm_dofs, :-1]**2) * dt_lagrange
    detailed_cost_function_831["shoulders_dof"][str(weight)] = np.sum(qs[shoulder_dofs, :-1]**2, axis=0) * dt_lagrange
    detailed_cost_function_831["Xrot_legs_dof"][str(weight)] = qs[XrotLegs, :-1]**2 * dt_lagrange
    detailed_cost_function_831["wobbling"][str(weight)] = qs[5, :-1]**2 * dt_lagrange
    detailed_cost_function_831["final_tilt"][str(weight)] = qs[5, -1]**2



current_time = 0
for i, time in enumerate(time_parameters[:-1]):
    current_time += time
    for j in range(6):
        axs_cost[j, 1].plot([current_time, current_time], [-250, 1e+8], '-k', alpha=0.3, linewidth=0.5)
axs_cost[0, 1].set_ylim(-10, 250)
axs_cost[1, 1].set_ylim(-100, 3.5e+6)
axs_cost[2, 1].set_ylim(-100, 8e+7)
axs_cost[3, 1].set_ylim(-10, 200)
axs_cost[4, 1].set_ylim(-100, 10000)
axs_cost[5, 1].set_ylim(-10, 1500)

# show legend below figure
axs_root[0, 0].legend(bbox_to_anchor=[3.7, 1.0], frameon=False)
axs_joints[0, 1].legend(bbox_to_anchor=[2.5, 0.5], frameon=False)
fig_joints.subplots_adjust(hspace=0.35, right=0.9)

axs_joints[0, 1].set_title("Change in elevation plane R")
axs_joints[1, 1].set_title("Elevation R")
axs_joints[0, 0].set_title("Change in elevation plane L")
axs_joints[1, 0].set_title("Elevation L")
axs_joints[0, 2].set_title("Flexion")
axs_joints[1, 2].set_title("Lateral flexion")

axs_cost[5, 1].set_xlabel("Time [s]", fontsize=15)
fig_cost.tight_layout()

axs_somersault_twist = plt.gca()
axs_somersault_twist.set_xlim(axs_somersault_twist.get_xlim()[::-1])
axs_somersault_twist.set_xlabel("Somersault [°]", fontsize=15)
axs_somersault_twist.set_ylabel("Twist [°]", fontsize=15)
fig_somersault_twist.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)

fig_root.savefig("Graphs/compare_831_root.png", dpi=300)
fig_joints.savefig("Graphs/compare_831_dofs.png", dpi=300)
fig_somersault_twist.savefig("Graphs/compare_831_somersault_twist.png", dpi=300)
fig_cost.savefig("Graphs/compare_cost.png", dpi=300)
plt.show()


viridis_colors = cm.get_cmap("viridis")
phases_42 = [range(0, 100), range(100, 140)]
fig_cost_bar_plot, axs_cost_bar_plot = plt.subplots(2, 2, figsize=(15, 9))
fig_cost_bar_plot_weighted, axs_cost_bar_plot_weighted = plt.subplots(2, 2, figsize=(15, 9))
sum_cost = {key: {weight: 0 for weight in detailed_cost_function_42[key].keys()} for key in detailed_cost_function_42.keys()}
sum_cost_weighted = {key: {weight: 0 for weight in detailed_cost_function_42[key].keys()} for key in detailed_cost_function_42.keys()}
for i_obj, obj in enumerate(detailed_cost_function_42.keys()):
    for i_weight, weight in enumerate(detailed_cost_function_42[obj].keys()):
        weights_42 = {"qddot_joints": [1, 1],
                      "qddot_joints_derivative": [1, 1],
                      "time": [0.00001, 0.00001],
                      "shoulders_dof": [50000, 0],
                      "final_tilt": [0, 1000],
                      "peripheral": [100 * float(weight), 0],
                      "spotting": [0, 10 * float(weight)],
                      "self_motion_detection": [1 * float(weight), 0],
                      "trampo_fixation": [1 * float(weight), 1000 * float(weight)],
                      "neck": [100 * float(weight), 100 * float(weight)],
                      "eyes": [10 * float(weight), 10 * float(weight)]}

        for i_phase in range(2):
            cost_this_time = detailed_cost_function_42[obj][weight]
            if isinstance(cost_this_time, float):
                sum_cost_this_time = cost_this_time
            elif phases_42[i_phase][-1] > len(cost_this_time) -1:
                sum_cost_this_time = np.sum(cost_this_time[phases_42[i_phase][0]:])
            else:
                sum_cost_this_time = np.sum(cost_this_time[phases_42[i_phase]])
            sum_cost_this_time = 0 if weights_42[obj][i_phase] == 0 else sum_cost_this_time
            sum_cost_this_time_weighted = sum_cost_this_time * weights_42[obj][i_phase]
            axs_cost_bar_plot[0, 0].bar(i_weight, sum_cost_this_time, bottom=sum_cost[obj][weight], color=viridis_colors(i_obj/15))
            axs_cost_bar_plot_weighted[0, 0].bar(i_weight, sum_cost_this_time_weighted, bottom=sum_cost_weighted[obj][weight], color=viridis_colors(i_obj/15))
            sum_cost[obj][weight] += sum_cost_this_time
            sum_cost_weighted[obj][weight] += sum_cost_this_time_weighted

            axs_cost_bar_plot[1, 0].bar(i_obj + 0.1 * i_weight, sum_cost_this_time, width=0.08, color=colors[i_weight])
            axs_cost_bar_plot_weighted[1, 0].bar(i_obj + 0.1 * i_weight, sum_cost_this_time_weighted, width=0.08, color=colors[i_weight])


phases_831 = [range(0, 40), range(40, 80), range(80, 120), range(120, 160), range(160, 200), range(200, 240)]
sum_cost = {key: {weight: 0 for weight in detailed_cost_function_831[key].keys()} for key in detailed_cost_function_831.keys()}
sum_cost_weighted = {key: {weight: 0 for weight in detailed_cost_function_831[key].keys()} for key in detailed_cost_function_831.keys()}
for i_obj, obj in enumerate(detailed_cost_function_831.keys()):
    for i_weight, weight in enumerate(detailed_cost_function_831[obj].keys()):
        weights_831 = {"qddot_joints": [1, 1, 1, 1, 1, 1],
                       "qddot_joints_derivative": [1, 1, 1, 1, 1, 1],
                       "time": [1, 100, -0.01, 100, -0.01, -0.01],
                       "shoulders_dof": [0, 0, 50000, 0, 0, 0],
                       "final_tilt": [0, 0, 0, 0, 0, 1000],
                       "peripheral": [100 * float(weight), 0, 0, 100 * float(weight), 100 * float(weight), 100 * float(weight)],
                       "spotting": [10 * float(weight), 0, 0, 0, 0, 10 * float(weight)],
                       "self_motion_detection": [1 * float(weight), 1 * float(weight), 1 * float(weight), 1 * float(weight), 1 * float(weight), 1 * float(weight)],
                       "trampo_fixation": [1 * float(weight), 0, 0, 0, 0, 1000 * float(weight)],
                       "neck": [100, 100, 100, 100, 100, 100],
                       "eyes": [10, 10, 10, 10, 10, 10],
                       "superimpose_markers": [0, 1, 0, 0, 0, 0],
                       "elbow_dof": [50000, 0, 0, 0, 0, 50000],
                       "arm_dof": [0, 0, 0, 0, 50000, 0],
                       "Xrot_legs_dof": [0, 0, 0, 50000, 50000, 50000],
                       "wobbling": [0, 0, 100, 0, 0, 0],
                       }

        for i_phase in range(6):
            cost_this_time = detailed_cost_function_831[obj][weight]
            if isinstance(cost_this_time, float):
                sum_cost_this_time = cost_this_time
            elif phases_831[i_phase][-1] > len(cost_this_time) -1:
                sum_cost_this_time = np.sum(cost_this_time[phases_831[i_phase][0]:])
            else:
                sum_cost_this_time = np.sum(cost_this_time[phases_831[i_phase]])
            sum_cost_this_time = 0 if weights_831[obj][i_phase] == 0 else sum_cost_this_time
            sum_cost_this_time_weighted = sum_cost_this_time * weights_831[obj][i_phase]
            if i_weight == 0 and i_phase == 0:
                axs_cost_bar_plot[0, 1].bar(i_weight, sum_cost_this_time, bottom=sum_cost[obj][weight], color=viridis_colors(i_obj/15), label=obj)
                axs_cost_bar_plot_weighted[0, 1].bar(i_weight, sum_cost_this_time_weighted,
                                                  bottom=sum_cost_weighted[obj][weight],
                                                  color=viridis_colors(i_obj / 15), label=obj)
            else:
                axs_cost_bar_plot[0, 1].bar(i_weight, sum_cost_this_time, bottom=sum_cost[obj][weight], color=viridis_colors(i_obj/15))
                axs_cost_bar_plot_weighted[0, 1].bar(i_weight, sum_cost_this_time_weighted, bottom=sum_cost_weighted[obj][weight], color=viridis_colors(i_obj/15))

            sum_cost[obj][weight] += sum_cost_this_time
            sum_cost_weighted[obj][weight] += sum_cost_this_time_weighted

            if i_obj == 0 and i_phase == 0:
                axs_cost_bar_plot[1, 1].bar(i_obj + 0.1 * i_weight, sum_cost_this_time, width=0.08,
                                            color=colors[i_weight], label=weight)
                axs_cost_bar_plot_weighted[1, 1].bar(i_obj + 0.1 * i_weight, sum_cost_this_time_weighted, width=0.08,
                                                     color=colors[i_weight], label=weight)
            else:
                axs_cost_bar_plot[1, 1].bar(i_obj + 0.1 * i_weight, sum_cost_this_time, width=0.08,
                                            color=colors[i_weight])
                axs_cost_bar_plot_weighted[1, 1].bar(i_obj + 0.1 * i_weight, sum_cost_this_time_weighted, width=0.08,
                                                     color=colors[i_weight])


axs_cost_bar_plot[0, 0].set_xticks(list(range(len(weights))))
axs_cost_bar_plot[0, 0].set_xticklabels([key for key in detailed_cost_function_42["peripheral"].keys()])
axs_cost_bar_plot[0, 1].set_xticks(list(range(len(weights))))
axs_cost_bar_plot[0, 1].set_xticklabels([key for key in detailed_cost_function_42["peripheral"].keys()])
axs_cost_bar_plot[0, 1].legend(bbox_to_anchor=[1.5, 1], frameon=False)
axs_cost_bar_plot[0, 0].set_title("42")
axs_cost_bar_plot[0, 1].set_title("831")
axs_cost_bar_plot[0, 0].set_ylabel("Cost")
axs_cost_bar_plot[1, 0].set_xticks([])
axs_cost_bar_plot[1, 1].set_xticks([])
axs_cost_bar_plot[1, 0].set_yscale('log')
axs_cost_bar_plot[1, 1].set_yscale('log')
axs_cost_bar_plot[1, 1].legend(bbox_to_anchor=[1.5, 1], frameon=False)

axs_cost_bar_plot_weighted[0, 0].set_xticks(list(range(len(weights))))
axs_cost_bar_plot_weighted[0, 0].set_xticklabels([key for key in detailed_cost_function_42["peripheral"].keys()])
axs_cost_bar_plot_weighted[0, 1].set_xticks(list(range(len(weights))))
axs_cost_bar_plot_weighted[0, 1].set_xticklabels([key for key in detailed_cost_function_42["peripheral"].keys()])
axs_cost_bar_plot_weighted[0, 1].legend(bbox_to_anchor=[1.8, 1], frameon=False)
axs_cost_bar_plot_weighted[0, 0].set_title("42")
axs_cost_bar_plot_weighted[0, 1].set_title("831")
axs_cost_bar_plot_weighted[0, 0].set_ylabel("Cost")
axs_cost_bar_plot_weighted[1, 0].set_xticks([])
axs_cost_bar_plot_weighted[1, 1].set_xticks([])
axs_cost_bar_plot_weighted[1, 0].set_yscale('log')
axs_cost_bar_plot_weighted[1, 1].set_yscale('log')
axs_cost_bar_plot_weighted[1, 1].legend(bbox_to_anchor=[1.5, 1], frameon=False)

fig_cost_bar_plot.subplots_adjust(left=0.1, right=0.7, top=0.9, bottom=0.1)
fig_cost_bar_plot_weighted.subplots_adjust(left=0.1, right=0.7, top=0.9, bottom=0.1)
fig_cost_bar_plot.savefig("Graphs/detailed_cost.png", dpi=300)
fig_cost_bar_plot_weighted.savefig("Graphs/detailed_cost_weighted.png", dpi=300)
plt.show()

print('ici')

# file_name_831 = "SoMe_without_mesh_831-(40_40_40_40_40_40)-2023-10-26-1040.pkl"  # Good 831<
# file_name_831 = "old/SoMe_without_mesh_831-(40_40_40_40_40_40)-2023-11-01-1206-0p0_CVG.pkl"  # Good 831<
# file_name_831_with_visual_criteria = "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-2023-10-25-1426.pkl"  # Good 831< with visual criteria
# file_name_831_with_visual_criteria = "old/SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-2023-11-02-1729-1p0_CVG.pkl"  # Good 831< with visual criteria
# file_name_42 = "SoMe_42_without_mesh-(100_40)-2023-10-26-1518.pkl"  # Good 42/
# file_name_42 = "old/SoMe_42_without_mesh-(100_40)-2023-10-28-0825-0p0_CVG.pkl"  # Good 42/
# file_name_42_with_visual_criteria = "SoMe_42_with_visual_criteria_without_mesh-(100_40)-2023-10-26-1533.pkl"   # Good 42/ with visual criteria
# file_name_42_with_visual_criteria = "old/SoMe_42_with_visual_criteria_without_mesh-(100_40)-2023-11-02-1810-1p0_CVG.pkl"   # Good 42/ with visual criteria

#
# # ---------------------------------------- Load data ---------------------------------------- #
#
# with open("Solutions/" + file_name_42, "rb") as f:
#     data = pickle.load(f)
#     sol = data[0]
#     q_per_phase_42 = data[1]
#     qs_42 = data[2]
#     qdots = data[3]
#     qddots = data[4]
#     time_parameters = data[5]
#     q_reintegrated = data[6]
#     qdot_reintegrated = data[7]
#     time_vector_42 = data[8]
#     interpolated_states_42 = data[9]
#
# with open("Solutions/" + file_name_42_with_visual_criteria, "rb") as f:
#     data = pickle.load(f)
#     sol = data[0]
#     q_per_phase_42_with_visual_criteria = data[1]
#     qs_42_with_visual_criteria = data[2]
#     qdots = data[3]
#     qddots = data[4]
#     time_parameters = data[5]
#     q_reintegrated = data[6]
#     qdot_reintegrated = data[7]
#     time_vector_42_with_visual_criteria = data[8]
#     interpolated_states_42_with_visual_criteria = data[9]
#
# with open("Solutions/" + file_name_831, "rb") as f:
#     data = pickle.load(f)
#     sol = data[0]
#     q_per_phase_831 = data[1]
#     qs_831 = data[2]
#     qdots = data[3]
#     qddots = data[4]
#     time_parameters = data[5]
#     q_reintegrated = data[6]
#     qdot_reintegrated = data[7]
#     time_vector_831 = data[8]
#     interpolated_states_831 = data[9]
#
# with open("Solutions/" + file_name_831_with_visual_criteria, "rb") as f:
#     data = pickle.load(f)
#     sol = data[0]
#     q_per_phase_831_with_visual_criteria = data[1]
#     qs_831_with_visual_criteria = data[2]
#     qdots = data[3]
#     qddots = data[4]
#     time_parameters = data[5]
#     q_reintegrated = data[6]
#     qdot_reintegrated = data[7]
#     time_vector_831_with_visual_criteria = data[8]
#     interpolated_states_831_with_visual_criteria = data[9]
#

# ---------------------------------------- Animate comparison ---------------------------------------- #

# q_per_phase_42_combined = [np.vstack((q_per_phase_42_with_visual_criteria[i], q_per_phase_42[i])) for i in range(len(q_per_phase_42))]
# qs_42_combined = np.vstack((qs_42_with_visual_criteria, qs_42))
# b = bioviz.Kinogram(model_path=biorbd_model_path_42_both,
#                    mesh_opacity=0.8,
#                    show_global_center_of_mass=False,
#                    show_gravity_vector=False,
#                    show_segments_center_of_mass=False,
#                    show_global_ref_frame=False,
#                    show_local_ref_frame=False,
#                    experimental_markers_color=(1, 1, 1),
#                    background_color=(1.0, 1.0, 1.0),
#                     )
# b.load_movement(q_per_phase_42_combined)
# b.set_camera_focus_point(0, 0, 2.5)
# b.set_camera_zoom(0.25)
# b.exec(frame_step=20,
#        save_path="Kinograms/42_both.svg")

# qs_42_combined_real_time = [np.vstack((interpolated_states_42_with_visual_criteria[i]["q"], interpolated_states_42[i]["q"])) for i in range(len(q_per_phase_42))]
# b = bioviz.Viz(biorbd_model_path_42_both,
#                mesh_opacity=0.8,
#                show_global_center_of_mass=False,
#                show_gravity_vector=False,
#                show_segments_center_of_mass=False,
#                show_global_ref_frame=False,
#                show_local_ref_frame=False,
#                experimental_markers_color=(1, 1, 1),
#                background_color=(1.0, 1.0, 1.0),
#                )
# b.load_movement(qs_42_combined_real_time)
# b.set_camera_zoom(0.25)
# b.set_camera_focus_point(0, 0, 2.5)
# b.exec()

# qs_42_combined_array = np.hstack((q_per_phase_42_combined[0][:, :-1], q_per_phase_42_combined[1]))
# qs_42_combined_array[15, :] -= 2
#
# b = bioviz.Viz(biorbd_model_path_42_both,
#                mesh_opacity=0.8,
#                show_global_center_of_mass=False,
#                show_gravity_vector=False,
#                show_segments_center_of_mass=False,
#                show_global_ref_frame=False,
#                show_local_ref_frame=False,
#                experimental_markers_color=(1, 1, 1),
#                background_color=(1.0, 1.0, 1.0),
#                )
# b.load_movement(qs_42_combined_array)
# b.set_camera_zoom(0.25)
# b.set_camera_focus_point(0, 0, 1.5) ##
# b.maximize()
# b.update()
# b.start_recording(f"comp_42.ogv")
# for frame in range(qs_42_combined_array.shape[1] + 1):
#     b.movement_slider[0].setValue(frame)
#     b.add_frame()
# b.stop_recording()
# b.quit()

# Animate comparison
# q_per_phase_831_combined = [np.vstack((q_per_phase_831_with_visual_criteria[i], q_per_phase_831[i])) for i in range(len(q_per_phase_831))]
# qs_831_combined = np.vstack((qs_831_with_visual_criteria, qs_831))
# b = bioviz.Kinogram(model_path=biorbd_model_path_831_both,
#                    mesh_opacity=0.8,
#                    show_global_center_of_mass=False,
#                    show_gravity_vector=False,
#                    show_segments_center_of_mass=False,
#                    show_global_ref_frame=False,
#                    show_local_ref_frame=False,
#                    experimental_markers_color=(1, 1, 1),
#                    background_color=(1.0, 1.0, 1.0),
#                     )
# b.load_movement(q_per_phase_831_combined)
# b.set_camera_zoom(0.25)
# b.set_camera_focus_point(0, 0, 2.5)
# b.exec(frame_step=20,
#        save_path="Kinograms/831_both.svg")

# qs_831_combined_real_time = [np.vstack((interpolated_states_831_with_visual_criteria[i]["q"], interpolated_states_831[i]["q"])) for i in range(len(q_per_phase_831))]
# b = bioviz.Viz(biorbd_model_path_831_both,
#                mesh_opacity=0.8,
#                show_global_center_of_mass=False,
#                show_gravity_vector=False,
#                show_segments_center_of_mass=False,
#                show_global_ref_frame=False,
#                show_local_ref_frame=False,
#                experimental_markers_color=(1, 1, 1),
#                background_color=(1.0, 1.0, 1.0),
#                )
# b.load_movement(qs_831_combined_real_time)
# b.set_camera_zoom(0.25)
# b.set_camera_focus_point(0, 0, 2.5)
# b.exec()

#
# # ---------------------------------------- Compare projected gaze orientation ---------------------------------------- #
# """
# These functions are adapted from Trampoline_EyeTracking_IMUs repo.
# """
#
# def get_gaze_position_from_intersection(vector_origin, vector_end, facing_front_wall):
#     def intersection_plane_vector(vector_origin, vector_end, planes_points, planes_normal_vector):
#         vector_orientation = vector_end - vector_origin
#         t = (np.dot(planes_points, planes_normal_vector) - np.dot(planes_normal_vector, vector_origin)) / np.dot(
#             vector_orientation, planes_normal_vector
#         )
#         return vector_origin + vector_orientation * np.abs(t)
#
#     def verify_intersection_position(vector_origin, vector_end, wall_index, bound_side, facing_front_wall):
#         vector_orientation = vector_end - vector_origin
#         if not facing_front_wall:
#             if wall_index == 0:  # trampoline
#                 t = (0 - vector_origin[2]) / vector_orientation[2]
#             elif wall_index == 1:  # wall front
#                 a = (bound_side - -bound_side) / (7.360 - 7.193)
#                 b = bound_side - a * 7.360
#                 t = (b + a * vector_origin[0] - vector_origin[1]) / (vector_orientation[1] - a * vector_orientation[0])
#             elif wall_index == 2:  # ceiling
#                 t = (9.4620 - 1.2192 - vector_origin[2]) / vector_orientation[2]
#             elif wall_index == 3:  # wall back
#                 t = (-8.881 - vector_origin[0]) / vector_orientation[0]
#             elif wall_index == 4:  # bound right
#                 t = (-bound_side - vector_origin[1]) / vector_orientation[1]
#             elif wall_index == 5:  # bound left
#                 t = (bound_side - vector_origin[1]) / vector_orientation[1]
#         else:
#             if wall_index == 0:  # trampoline
#                 t = (0 - vector_origin[2]) / vector_orientation[2]
#             elif wall_index == 1:  # wall front
#                 t = (7.2 - vector_origin[0]) / vector_orientation[0]
#             elif wall_index == 2:  # ceiling
#                 t = (9.4620 - 1.2192 - vector_origin[2]) / vector_orientation[2]
#             elif wall_index == 3:  # wall back
#                 t = (-7.2 - vector_origin[0]) / vector_orientation[0]
#             elif wall_index == 4:  # bound right
#                 t = (-bound_side - vector_origin[1]) / vector_orientation[1]
#             elif wall_index == 5:  # bound left
#                 t = (bound_side - vector_origin[1]) / vector_orientation[1]
#         return vector_origin + vector_orientation * t
#
#     bound_side = 3 + 121 * 0.0254 / 2
#     if not facing_front_wall:
#         # zero is positioned at the center of the trampoline
#         planes_points = np.array(
#             [
#                 [7.193, bound_side, 0],  # trampoline
#                 [7.193, bound_side, 0],  # wall front
#                 [7.193, bound_side, 9.4620 - 1.2192],  # ceiling
#                 [-8.881, bound_side, 0],  # wall back
#                 [7.193, bound_side, 0],  # bound right
#                 [7.360, -bound_side, 0],  # bound left
#             ]
#         )
#
#         planes_normal_vector = np.array(
#             [
#                 [0, 0, 1],  # trampoline
#                 np.cross(
#                     np.array([7.193, bound_side, 0]) - np.array([7.360, -bound_side, 0]), np.array([0, 0, -1])
#                 ).tolist(),  # wall front
#                 [0, 0, -1],  # ceiling
#                 [1, 0, 0],  # wall back
#                 [0, 1, 0],  # bound right
#                 [0, -1, 0],  # bound left
#             ]
#         )
#
#         plane_bounds = [
#             np.array([[-8.881, 7.360], [-bound_side, bound_side], [0, 0]]),
#             np.array([[7.193, 7.360], [-bound_side, bound_side], [0, 9.4620 - 1.2192]]),
#             np.array([[-8.881, 7.360], [-bound_side, bound_side], [9.4620 - 1.2192, 9.4620 - 1.2192]]),
#             np.array([[-8.881, -8.881], [-bound_side, bound_side], [0, 9.4620 - 1.2192]]),
#             np.array([[-8.881, 7.193], [-bound_side, -bound_side], [0, 9.4620 - 1.2192]]),
#             np.array([[-8.881, 7.360], [bound_side, bound_side], [0, 9.4620 - 1.2192]]),
#         ]
#
#     else:
#         # zero is positioned at the center of the trampoline
#         planes_points = np.array(
#             [
#                 [7.2, bound_side, 0],  # trampoline
#                 [7.2, bound_side, 0],  # wall front
#                 [7.2, bound_side, 9.4620 - 1.2192],  # ceiling
#                 [-7.2, bound_side, 0],  # wall back
#                 [7.2, bound_side, 0],  # bound right
#                 [7.2, -bound_side, 0],  # bound left
#             ]
#         )
#
#         planes_normal_vector = np.array(
#             [
#                 [0, 0, 1],  # trampoline
#                 [-1, 0, 0],  # wall front
#                 [0, 0, -1],  # ceiling
#                 [1, 0, 0],  # wall back
#                 [0, 1, 0],  # bound right
#                 [0, -1, 0],  # bound left
#             ]
#         )
#
#         plane_bounds = [
#             np.array([[-7.2, 7.2], [-bound_side, bound_side], [0, 0]]),
#             np.array([[7.2, 7.2], [-bound_side, bound_side], [0, 9.4620 - 1.2192]]),
#             np.array([[-7.2, 7.2], [-bound_side, bound_side], [9.4620 - 1.2192, 9.4620 - 1.2192]]),
#             np.array([[-7.2, -7.2], [-bound_side, bound_side], [0, 9.4620 - 1.2192]]),
#             np.array([[-7.2, 7.2], [-bound_side, -bound_side], [0, 9.4620 - 1.2192]]),
#             np.array([[-7.2, 7.2], [bound_side, bound_side], [0, 9.4620 - 1.2192]]),
#         ]
#
#     gaze_positions = np.zeros((vector_origin.shape[0], 3))
#     wall_indices = np.zeros((vector_origin.shape[0]))
#     for i_node in range(vector_origin.shape[0]):
#         intersection = []
#         wall_index = None
#         intersection_index = np.zeros((len(planes_points)))
#         for i in range(len(planes_points)):
#             current_interaction = intersection_plane_vector(
#                 vector_origin[i_node, :], vector_end[i_node, :], planes_points[i, :], planes_normal_vector[i, :],
#             )
#
#             if current_interaction is not None:
#                 bounds_bool = True
#                 vector_orientation = vector_end[i_node, :] - vector_origin[i_node, :]
#                 potential_gaze_orientation = current_interaction - vector_origin[i_node, :]
#                 cross_condition = np.linalg.norm(np.cross(vector_orientation, potential_gaze_orientation))
#                 dot_condition = np.dot(vector_orientation, potential_gaze_orientation)
#                 if dot_condition > 0:
#                     if cross_condition > -0.01 and cross_condition < 0.01:
#                         for i_bool in range(3):
#                             if (
#                                 current_interaction[i_bool] > plane_bounds[i][i_bool, 0] - 1
#                                     and current_interaction[i_bool] < plane_bounds[i][i_bool, 1] + 1
#                             ):
#                                 a = 1
#                             else:
#                                 bounds_bool = False
#                     else:
#                         bounds_bool = False
#                 else:
#                     bounds_bool = False
#
#             if bounds_bool:
#                 intersection += [current_interaction]
#                 intersection_index[i] = 1
#                 wall_index = i
#
#         if intersection_index.sum() > 1:
#             bound_crossing = np.zeros((len(np.where(intersection_index == 1)[0])))
#             for idx, i in enumerate(np.where(intersection_index == 1)[0]):
#                 for j in range(3):
#                     if plane_bounds[i][j, 0] - intersection[idx][j] > 0:
#                         bound_crossing[idx] += np.abs(plane_bounds[i][j, 0] - intersection[idx][j])
#                     if plane_bounds[i][j, 1] - intersection[idx][j] < 0:
#                         bound_crossing[idx] += np.abs(plane_bounds[i][j, 1] - intersection[idx][j])
#             closest_index = np.argmin(bound_crossing)
#             wall_index = np.where(intersection_index == 1)[0][closest_index]
#
#         if wall_index is not None:
#             gaze_position = verify_intersection_position(vector_origin[i_node, :], vector_end[i_node, :], wall_index, bound_side, facing_front_wall)
#         else:
#             gaze_position = None
#         gaze_positions[i_node, :] = gaze_position
#         wall_indices[i_node] = wall_index
#
#     return gaze_positions, wall_indices
#
#
#
# def plot_gaze_trajectory(
#         gaze_position_temporal_evolution_projected,
#         gaze_position_temporal_evolution_projected_with_visual_criteria,
#         output_file_name,
# ):
#     """
#     This function plots the gaze trajectory and the fixation positions projected on the gymnasium in 3D.
#     """
#
#     def plot_gymnasium_symmetrized(bound_side, ax):
#         """
#         Plot the gymnasium in 3D with the walls and trampoline bed.
#         """
#         ax.set_box_aspect([1, 1, 1])
#         ax.view_init(elev=10.0, azim=-70)
#
#         ax.set_xlim3d([-8.0, 8.0])
#         ax.set_ylim3d([-8.0, 8.0])
#         ax.set_zlim3d([-3.0, 13.0])
#         ax.set_xlabel("X")
#         ax.set_ylabel("Y")
#         ax.set_zlabel("Z")
#
#         # Front right, to front left (bottom)
#         plt.plot(np.array([7.2, 7.2]), np.array([-bound_side, bound_side]), np.array([0, 0]), "-k")
#         # Front right, to back right (bottom)
#         plt.plot(np.array([-7.2, 7.2]), np.array([-bound_side, -bound_side]), np.array([0, 0]), "-k")
#         # Front left, to back left (bottom)
#         plt.plot(np.array([-7.2, 7.2]), np.array([bound_side, bound_side]), np.array([0, 0]), "-k")
#         # Back right, to back left (bottom)
#         plt.plot(np.array([-7.2, -7.2]), np.array([-bound_side, bound_side]), np.array([0, 0]), "-k")
#
#         # Front right, to front left (ceiling)
#         plt.plot(
#             np.array([7.2, 7.2]),
#             np.array([-bound_side, bound_side]),
#             np.array([9.4620 - 1.2192, 9.4620 - 1.2192]),
#             "-k")
#
#         # Front right, to back right (ceiling)
#         plt.plot(
#             np.array([-7.2, 7.2]),
#             np.array([-bound_side, -bound_side]),
#             np.array([9.4620 - 1.2192, 9.4620 - 1.2192]),
#             "-k",
#         )
#         # Front left, to back left (ceiling)
#         plt.plot(
#             np.array([-7.2, 7.2]),
#             np.array([bound_side, bound_side]),
#             np.array([9.4620 - 1.2192, 9.4620 - 1.2192]),
#             "-k",
#         )
#         # Back right, to back left (ceiling)
#         plt.plot(
#             np.array([-7.2, -7.2]),
#             np.array([-bound_side, bound_side]),
#             np.array([9.4620 - 1.2192, 9.4620 - 1.2192]),
#             "-k",
#         )
#
#         # Front right bottom, to front right ceiling
#         plt.plot(np.array([7.2, 7.2]), np.array([-bound_side, -bound_side]), np.array([0, 9.4620 - 1.2192]), "-k")
#         # Front left bottom, to front left ceiling
#         plt.plot(np.array([7.2, 7.2]), np.array([bound_side, bound_side]), np.array([0, 9.4620 - 1.2192]), "-k")
#         # Back right bottom, to back right ceiling
#         plt.plot(np.array([-7.2, -7.2]), np.array([-bound_side, -bound_side]), np.array([0, 9.4620 - 1.2192]), "-k")
#         # Back left bottom, to back left ceiling
#         plt.plot(np.array([-7.2, -7.2]), np.array([bound_side, bound_side]), np.array([0, 9.4620 - 1.2192]), "-k")
#
#         # Trampoline
#         X, Y = np.meshgrid([-7 * 0.3048, 7 * 0.3048], [-3.5 * 0.3048, 3.5 * 0.3048])
#         Z = np.zeros(X.shape)
#         ax.plot_surface(X, Y, Z, color="k", alpha=0.4)
#         return
#
#     bound_side = 3 + 121 * 0.0254 / 2
#
#     fig = plt.figure()
#     ax = fig.add_subplot(projection='3d')
#
#     plot_gymnasium_symmetrized(bound_side, ax)
#
#     N = len(gaze_position_temporal_evolution_projected[:, 0]) - 1
#     for j in range(N):
#         ax.plot(
#             gaze_position_temporal_evolution_projected[j: j + 2, 0],
#             gaze_position_temporal_evolution_projected[j: j + 2, 1],
#             gaze_position_temporal_evolution_projected[j: j + 2, 2],
#             color=plt.cm.winter(j / N),
#         )
#         ax.plot(
#             gaze_position_temporal_evolution_projected_with_visual_criteria[j: j + 2, 0],
#             gaze_position_temporal_evolution_projected_with_visual_criteria[j: j + 2, 1],
#             gaze_position_temporal_evolution_projected_with_visual_criteria[j: j + 2, 2],
#             color=plt.cm.autumn(j / N),
#         )
#         # spring for real athlete
#
#     ax.set_title("Gaze trajectory")
#
#     plt.savefig(output_file_name + "_gaze_trajectory.png", dpi=300)
#     # plt.show()
#     # plt.close("all")
#     return
#
# def plot_unwrapped_trajectories(
#         gaze_position,
#         gaze_position_with_visual_criteria,
#         wall_index,
#         wall_index_with_visual_criteria,
#         output_filename,
# ):
#     def plot_gymnasium_unwrapped(axs):
#
#         bound_side = 3 + 121 * 0.0254 / 2
#
#         # Plot trampo bed
#         axs.add_patch(Rectangle((-7 * 0.3048, -3.5 * 0.3048), 14 * 0.3048, 7 * 0.3048, facecolor='k', alpha=0.2))
#         # Plot vertical lines of the symmetrized gymnasium
#         axs.plot(np.array([-7.2, -7.2]), np.array([-bound_side, bound_side]), '-k')
#         axs.plot(np.array([7.2, 7.2]), np.array([-bound_side, bound_side]), '-k')
#         axs.plot(np.array([-7.2 - (9.4620 - 1.2192), -7.2 - (9.4620 - 1.2192)]), np.array([-bound_side, bound_side]),
#                     '-k')
#         axs.plot(np.array([7.2 + 9.4620 - 1.2192, 7.2 + 9.4620 - 1.2192]), np.array([-bound_side, bound_side]), '-k')
#         axs.plot(np.array([-7.2 - (9.4620 - 1.2192) - 2 * 7.2, -7.2 - (9.4620 - 1.2192) - 2 * 7.2]),
#                     np.array([-bound_side, bound_side]), '-k')
#         axs.plot(np.array([-7.2, -7.2]), np.array([-bound_side - (9.4620 - 1.2192), bound_side]), '-k')
#         axs.plot(np.array([7.2, 7.2]), np.array([-bound_side - (9.4620 - 1.2192), bound_side]), '-k')
#         axs.plot(np.array([-7.2, -7.2]), np.array([bound_side, bound_side + 9.4620 - 1.2192]), '-k')
#         axs.plot(np.array([7.2, 7.2]), np.array([bound_side, bound_side + 9.4620 - 1.2192]), '-k')
#         # Plot horizontal lines of the symmetrized gymnasium
#         axs.plot(np.array([-7.2, 7.2]), np.array([-bound_side, -bound_side]), '-k')
#         axs.plot(np.array([-7.2, 7.2]), np.array([bound_side, bound_side]), '-k')
#         axs.plot(np.array([-7.2, 7.2]), np.array([-bound_side - (9.4620 - 1.2192), -bound_side - (9.4620 - 1.2192)]),
#                     '-k')
#         axs.plot(np.array([-7.2, 7.2]), np.array([bound_side + 9.4620 - 1.2192, bound_side + 9.4620 - 1.2192]), '-k')
#         axs.plot(np.array([7.2, 7.2 + 9.4620 - 1.2192]), np.array([-bound_side, -bound_side]), '-k')
#         axs.plot(np.array([7.2, 7.2 + 9.4620 - 1.2192]), np.array([bound_side, bound_side]), '-k')
#         axs.plot(np.array([-7.2 - (9.4620 - 1.2192), 7.2]), np.array([-bound_side, -bound_side]), '-k')
#         axs.plot(np.array([-7.2 - (9.4620 - 1.2192), 7.2]), np.array([bound_side, bound_side]), '-k')
#         axs.plot(np.array([-7.2 - (9.4620 - 1.2192) - 2 * 7.2, -7.2 - (9.4620 - 1.2192)]),
#                     np.array([-bound_side, -bound_side]), '-k')
#         axs.plot(np.array([-7.2 - (9.4620 - 1.2192) - 2 * 7.2, -7.2 - (9.4620 - 1.2192)]),
#                     np.array([bound_side, bound_side]), '-k')
#
#         axs.text(-7.2 - (9.4620 - 1.2192) - 2 * 7.2 + 7.2 / 2 + 1, bound_side + 0.2, "Ceiling", fontsize=10)
#         axs.text(-7.2 - (9.4620 - 1.2192) + 1, bound_side + 0.2, "Back wall", fontsize=10)
#         axs.text(7.2 + 1, bound_side + 0.2, "Front wall", fontsize=10)
#         axs.text(-7.2 + 7.2 / 2 + 1, bound_side + 9.4620 - 1.2192 + 0.2, "Left wall", fontsize=10)
#         axs.text(-7.2 + 7.2 / 2 + 0.5, -bound_side - (9.4620 - 1.2192) - 1.2, "Right wall", fontsize=10)
#
#         return
#
#     def unwrap_gaze_positions(gaze_position, wall_index):
#         bound_side = 3 + 121 * 0.0254 / 2
#
#         gaze_position_x_y = np.zeros((2, np.shape(wall_index)[0]))
#         gaze_position_x_y[:, :] = np.nan
#         for i in range(len(wall_index)):
#             if wall_index[i] == 0:  # trampoline
#                 gaze_position_x_y[:, i] = gaze_position[i][:2]
#             if wall_index[i] == 1:  # wall front
#                 gaze_position_x_y[:, i] = [gaze_position[i][2] + 7.2, gaze_position[i][1]]
#             elif wall_index[i] == 2:  # ceiling
#                 gaze_position_x_y[:, i] = [-7.2 - (9.4620 - 1.2192) - 7.2 - gaze_position[i][0], gaze_position[i][1]]
#             elif wall_index[i] == 3:  # wall back
#                 gaze_position_x_y[:, i] = [-7.2 - gaze_position[i][2], gaze_position[i][1]]
#             elif wall_index[i] == 4:  # bound right
#                 gaze_position_x_y[:, i] = [gaze_position[i][0], -bound_side - gaze_position[i][2]]
#             elif wall_index[i] == 5:  # bound left
#                 gaze_position_x_y[:, i] = [gaze_position[i][0], bound_side + gaze_position[i][2]]
#         return gaze_position_x_y
#
#     fig, axs = plt.subplots(1, 1, figsize=(9, 6))
#
#     unwrapped_gaze_position = unwrap_gaze_positions(gaze_position, wall_index)
#     unwrapped_gaze_position_with_visual_criteria = unwrap_gaze_positions(gaze_position_with_visual_criteria, wall_index_with_visual_criteria)
#
#     axs.scatter(unwrapped_gaze_position[0, :],
#                 unwrapped_gaze_position[1, :],
#                 c=np.linspace(0, 1, unwrapped_gaze_position.shape[1]),
#                 cmap='winter', marker='.')
#     axs.scatter(unwrapped_gaze_position_with_visual_criteria[0, :],
#                 unwrapped_gaze_position_with_visual_criteria[1, :],
#                 c=np.linspace(0, 1, unwrapped_gaze_position_with_visual_criteria.shape[1]),
#                 cmap='autumn', marker='.')
#
#     plot_gymnasium_unwrapped(axs)
#     axs.axis('equal')
#
#     plt.subplots_adjust(right=0.8)
#     plt.savefig(output_filename, dpi=300)
#     # plt.show()
#     # plt.close('all')
#     return
#
#
# model_42_eyes_start_idx, model_42_eyes_end_idx = find_eye_markers_index(model_42)
# model_42_with_visual_criteria_eyes_start_idx, model_42_with_visual_criteria_eyes_end_idx = find_eye_markers_index(model_42_with_visual_criteria)
# model_831_eyes_start_idx, model_831_eyes_end_idx = find_eye_markers_index(model_831)
# model_831_with_visual_criteria_eyes_start_idx, model_831_with_visual_criteria_eyes_end_idx = find_eye_markers_index(model_831_with_visual_criteria)
#
# # Coordinate system of the gymnasium (x-orientation for xsens is front wall, x-orientation for OCP is right wall)
# rotation_matrix = biorbd.Rotation.fromEulerAngles(np.array([np.pi/2]), 'z').to_array()
#
# vector_origin_42 = np.zeros((time_vector_42.shape[0], 3))
# vector_end_42 = np.zeros((time_vector_42.shape[0], 3))
# vector_origin_42_with_visual_criteria = np.zeros((time_vector_42_with_visual_criteria.shape[0], 3))
# vector_end_42_with_visual_criteria = np.zeros((time_vector_42_with_visual_criteria.shape[0], 3))
# for i_node in range(time_vector_42.shape[0]):
#     vector_origin_42[i_node, :] = rotation_matrix @ model_42.markers(qs_42[:, i_node])[model_42_eyes_start_idx].to_array()
#     vector_end_42[i_node, :] = rotation_matrix @ model_42.markers(qs_42[:, i_node])[model_42_eyes_end_idx].to_array()
#     vector_origin_42_with_visual_criteria[i_node, :] = rotation_matrix @ model_42_with_visual_criteria.markers(qs_42_with_visual_criteria[:, i_node])[model_42_with_visual_criteria_eyes_start_idx].to_array()
#     vector_end_42_with_visual_criteria[i_node, :] = rotation_matrix @ model_42_with_visual_criteria.markers(qs_42_with_visual_criteria[:, i_node])[model_42_with_visual_criteria_eyes_end_idx].to_array()
#
# gaze_position_42, wall_index_42 = get_gaze_position_from_intersection(vector_origin_42, vector_end_42, True)
# gaze_position_42_with_visual_criteria, wall_index_42_with_visual_criteria = get_gaze_position_from_intersection(vector_origin_42_with_visual_criteria, vector_end_42_with_visual_criteria, True)
#
# plot_gaze_trajectory(gaze_position_42,
#                      gaze_position_42_with_visual_criteria,
#                      "Graphs/compare_42")
#
# plot_unwrapped_trajectories(gaze_position_42,
#                             gaze_position_42_with_visual_criteria,
#                             wall_index_42,
#                             wall_index_42_with_visual_criteria,
#                             "Graphs/compare_42_unwrapped")
#
#
# vector_origin_831 = np.zeros((time_vector_831.shape[0], 3))
# vector_end_831 = np.zeros((time_vector_831.shape[0], 3))
# vector_origin_831_with_visual_criteria = np.zeros((time_vector_831_with_visual_criteria.shape[0], 3))
# vector_end_831_with_visual_criteria = np.zeros((time_vector_831_with_visual_criteria.shape[0], 3))
# for i_node in range(time_vector_831.shape[0]):
#     vector_origin_831[i_node, :] = rotation_matrix @ model_831.markers(qs_831[:, i_node])[model_831_eyes_start_idx].to_array()
#     vector_end_831[i_node, :] = rotation_matrix @ model_831.markers(qs_831[:, i_node])[model_831_eyes_end_idx].to_array()
#     vector_origin_831_with_visual_criteria[i_node, :] = rotation_matrix @ model_831_with_visual_criteria.markers(qs_831_with_visual_criteria[:, i_node])[model_831_with_visual_criteria_eyes_start_idx].to_array()
#     vector_end_831_with_visual_criteria[i_node, :] = rotation_matrix @ model_831_with_visual_criteria.markers(qs_831_with_visual_criteria[:, i_node])[model_831_with_visual_criteria_eyes_end_idx].to_array()
#
# gaze_position_831, wall_index_831 = get_gaze_position_from_intersection(vector_origin_831, vector_end_831, True)
# gaze_position_831_with_visual_criteria, wall_index_831_with_visual_criteria = get_gaze_position_from_intersection(vector_origin_831_with_visual_criteria, vector_end_831_with_visual_criteria, True)
#
# plot_gaze_trajectory(gaze_position_831,
#                      gaze_position_831_with_visual_criteria,
#                      "Graphs/compare_42")
#
# plot_unwrapped_trajectories(gaze_position_831,
#                             gaze_position_831_with_visual_criteria,
#                             wall_index_831,
#                             wall_index_831_with_visual_criteria,
#                             "Graphs/compare_831_unwrapped")
#
# plt.show()
