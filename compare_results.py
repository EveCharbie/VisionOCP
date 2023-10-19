"""
The goal of this program is to compare the athlete kinematics with the optimal kinematics with and without visual criteria.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import bioviz
import bioptim
import biorbd

ACROBATICS = ["42", "831"]


biorbd_model_path_831_with_visual_criteria = "models/SoMe_with_visual_criteria.bioMod"
biorbd_model_path_831 = "models/SoMe.bioMod"
n_shooting_831 = (40, 100, 100, 100, 40)
biorbd_model_path_42_with_visual_criteria = "models/SoMe_42_with_visual_criteria.bioMod"
biorbd_model_path_42 = "models/SoMe_42.bioMod"
biorbd_model_path_42_both = "models/SoMe_42_with_and_without_visual_criteria.bioMod"
n_shooting_42 = (100, 40)

file_name_831 = "SoMe-1-(40_100_100_100_40)-2023-04-17-2319.pkl"  # Good 831< without visual criteria
file_name_831_with_visual_criteria = "todo"
file_name_42 = "SoMe_42_without_mesh-(100_40)-2023-10-19-1319.pkl"  # Good 42/
file_name_42_with_visual_criteria = "SoMe_42_with_visual_criteria_without_mesh-(100_40)-2023-10-19-1314.pkl"   # Good 42/ with visual criteria

with open("Solutions/" + file_name_42, "rb") as f:
    data = pickle.load(f)
    # sol = data[0]
    qs_42 = data[1]
    # qdots = data[2]
    # qddots = data[3]
    # time_parameters = data[4]
    # q_reintegrated = data[5]
    # qdot_reintegrated = data[6]
    time_vector_42 = data[7]

with open("Solutions/" + file_name_42_with_visual_criteria, "rb") as f:
    data = pickle.load(f)
    # sol = data[0]
    qs_42_with_visual_criteria = data[1]
    # qdots = data[2]
    # qddots = data[3]
    # time_parameters = data[4]
    # q_reintegrated = data[5]
    # qdot_reintegrated = data[6]
    time_vector_42_with_visual_criteria = data[7]

# with open("Solutions/" + file_name_831, "rb") as f:
#     data = pickle.load(f)
#     # sol = data[0]
#     qs_831 = data[1]
#     # qdots = data[2]
#     # qddots = data[3]
#     # time_parameters = data[4]
#     # q_reintegrated = data[5]
#     # qdot_reintegrated = data[6]
#     time_vector_831 = data[7]

# with open("Solutions/" + file_name_831_with_visual_criteria, "rb") as f:
#     data = pickle.load(f)
#     # sol = data[0]
#     qs_831_with_visual_criteria = data[1]
#     # qdots = data[2]
#     # qddots = data[3]
#     # time_parameters = data[4]
#     # q_reintegrated = data[5]
#     # qdot_reintegrated = data[6]
#     time_vector_831_with_visual_criteria = data[7]

# Animate comparison
qs_42_combined = np.vstack((qs_42_with_visual_criteria, qs_42))
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
# b.load_movement(qs_42_combined)
# b.exec()




fig, axs = plt.subplots(1, 3, figsize=(15, 3))
for i in range(3):
    axs[i].plot(time_vector_42, qs_42[i+3, :], 'tab:blue')
    axs[i].plot(time_vector_42_with_visual_criteria, qs_42_with_visual_criteria[i+3, :], 'tab:red')
plt.savefig("Graphs/compare_42_root.png", dpi=300)
plt.show()

fig, axs = plt.subplots(2, 2)

axs[0, 0].plot(time_vector_42, qs_42[6, :], 'tab:blue', label="OCP without vision")
axs[0, 0].plot(time_vector_42_with_visual_criteria, qs_42_with_visual_criteria[6+4, :], 'tab:red', label="OCP with vision")
axs[0, 0].set_title("Change in elevation plane R")

axs[0, 1].plot(time_vector_42, qs_42[7, :], 'tab:blue')
axs[0, 1].plot(time_vector_42_with_visual_criteria, qs_42_with_visual_criteria[7+4, :], 'tab:red')
axs[0, 1].set_title("Elevation R")

axs[1, 0].plot(time_vector_42, qs_42[8, :], 'tab:blue')
axs[1, 0].plot(time_vector_42_with_visual_criteria, qs_42_with_visual_criteria[8+4, :], 'tab:red')
axs[1, 0].set_title("Change in elevation plane L")

axs[1, 1].plot(time_vector_42, qs_42[9, :], 'tab:blue')
axs[1, 1].plot(time_vector_42_with_visual_criteria, qs_42_with_visual_criteria[9+4, :], 'tab:red')
axs[1, 1].set_title("Elevation L")

# show legend below figure
axs[0, 0].legend(bbox_to_anchor=[2.0, -1.5], ncols=2, frameon=False)
plt.subplots_adjust(hspace=0.35)
plt.savefig("Graphs/compare_42_dofs.png", dpi=300)
plt.show()


# model = biorbd.Model(biorbd_model_path)
# num_frames = qs.shape[1]
# obj = np.zeros(num_frames)
# first_condition = np.zeros(num_frames)
# second_condition = np.zeros(num_frames)
# third_condition = np.zeros(num_frames)
# for i in range(num_frames):
#     obj[i], first_condition[i], second_condition[i], third_condition[i] = custom_trampoline_bed_in_peripheral_vision(model, qs[:, i])
#
# plt.figure()
# plt.plot(obj, 'k')
# plt.plot(first_condition * 220, 'r')
# plt.plot(second_condition * 220, 'g')
# plt.plot(third_condition * 220, 'b')
# plt.show()