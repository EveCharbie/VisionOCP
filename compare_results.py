"""
The goal of this program is to compare the athlete kinematics with the optimal kinematics with and without visual criteria.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pickle
# import bioviz
import sys
sys.path.append("/home/charbie/Documents/Programmation/BiorbdOptim")
import bioptim
import biorbd
from IPython import embed

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

file_name_831 = "SoMe_without_mesh_831-(40_40_40_40_40_40)-2023-10-26-1040.pkl"  # Good 831<
file_name_831_with_visual_criteria = "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-2023-10-25-1426.pkl"  # Good 831< with visual criteria
file_name_42 = "SoMe_42_without_mesh-(100_40)-2023-10-20-1652.pkl"  # Good 42/
file_name_42_with_visual_criteria = "SoMe_42_with_visual_criteria_without_mesh-(100_40)-2023-10-20-1631.pkl"   # Good 42/ with visual criteria


# ---------------------------------------- Load data ---------------------------------------- #

with open("Solutions/" + file_name_42, "rb") as f:
    data = pickle.load(f)
    # sol = data[0]
    q_per_phase_42 = data[1]
    qs_42 = data[2]
    # qdots = data[3]
    # qddots = data[4]
    # time_parameters = data[5]
    # q_reintegrated = data[6]
    # qdot_reintegrated = data[7]
    time_vector_42 = data[8]
    interpolated_states_42 = data[9]

with open("Solutions/" + file_name_42_with_visual_criteria, "rb") as f:
    data = pickle.load(f)
    # sol = data[0]
    q_per_phase_42_with_visual_criteria = data[1]
    qs_42_with_visual_criteria = data[2]
    # qdots = data[3]
    # qddots = data[4]
    # time_parameters = data[5]
    # q_reintegrated = data[6]
    # qdot_reintegrated = data[7]
    time_vector_42_with_visual_criteria = data[8]
    interpolated_states_42_with_visual_criteria = data[9]

with open("Solutions/" + file_name_831, "rb") as f:
    data = pickle.load(f)
    # sol = data[0]
    q_per_phase_831 = data[1]
    qs_831 = data[2]
    # qdots = data[3]
    # qddots = data[4]
    # time_parameters = data[5]
    # q_reintegrated = data[6]
    # qdot_reintegrated = data[7]
    time_vector_831 = data[8]
    interpolated_states_831 = data[9]

with open("Solutions/" + file_name_831_with_visual_criteria, "rb") as f:
    data = pickle.load(f)
    # sol = data[0]
    q_per_phase_831_with_visual_criteria = data[1]
    qs_831_with_visual_criteria = data[2]
    # qdots = data[3]
    # qddots = data[4]
    # time_parameters = data[5]
    # q_reintegrated = data[6]
    # qdot_reintegrated = data[7]
    time_vector_831_with_visual_criteria = data[8]
    interpolated_states_831_with_visual_criteria = data[9]


# ---------------------------------------- Animate comparison ---------------------------------------- #

q_per_phase_42_combined = [np.vstack((q_per_phase_42_with_visual_criteria[i], q_per_phase_42[i])) for i in range(len(q_per_phase_42))]
qs_42_combined = np.vstack((qs_42_with_visual_criteria, qs_42))
b = bioviz.Kinogram(model_path=biorbd_model_path_42_both,
                   mesh_opacity=0.8,
                   show_global_center_of_mass=False,
                   show_gravity_vector=False,
                   show_segments_center_of_mass=False,
                   show_global_ref_frame=False,
                   show_local_ref_frame=False,
                   experimental_markers_color=(1, 1, 1),
                   background_color=(1.0, 1.0, 1.0),
                    )
b.load_movement(q_per_phase_42_combined)
b.set_camera_zoom(0.25)
b.exec(frame_step=20,
       save_path="Kinograms/42_both.svg")

qs_42_combined_real_time = [np.vstack((interpolated_states_42_with_visual_criteria[i]["q"], interpolated_states_42[i]["q"])) for i in range(len(q_per_phase_42))]
b = bioviz.Viz(biorbd_model_path_42_both,
               mesh_opacity=0.8,
               show_global_center_of_mass=False,
               show_gravity_vector=False,
               show_segments_center_of_mass=False,
               show_global_ref_frame=False,
               show_local_ref_frame=False,
               experimental_markers_color=(1, 1, 1),
               background_color=(1.0, 1.0, 1.0),
               )
b.load_movement(qs_42_combined_real_time)
b.set_camera_zoom(0.25)
b.exec()

# # Animate comparison
q_per_phase_831_combined = [np.vstack((q_per_phase_831_with_visual_criteria[i], q_per_phase_831[i])) for i in range(len(q_per_phase_831))]
qs_831_combined = np.vstack((qs_831_with_visual_criteria, qs_831))
b = bioviz.Kinogram(model_path=biorbd_model_path_831_both,
                   mesh_opacity=0.8,
                   show_global_center_of_mass=False,
                   show_gravity_vector=False,
                   show_segments_center_of_mass=False,
                   show_global_ref_frame=False,
                   show_local_ref_frame=False,
                   experimental_markers_color=(1, 1, 1),
                   background_color=(1.0, 1.0, 1.0),
                    )
b.load_movement(q_per_phase_831_combined)
b.set_camera_zoom(0.25)
b.exec(frame_step=20,
       save_path="Kinograms/831_both.svg")

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
# b.exec()


# ---------------------------------------- Plot comparison ---------------------------------------- #

# 42 plots
fig, axs = plt.subplots(1, 3, figsize=(15, 3))
for i in range(3):
    axs[i].plot(time_vector_42, qs_42[i+3, :], 'tab:blue')
    axs[i].plot(time_vector_42_with_visual_criteria, qs_42_with_visual_criteria[i+3, :], 'tab:red')
plt.savefig("Graphs/compare_42_root.png", dpi=300)
# plt.show()

fig, axs = plt.subplots(2, 2)
# Right arm
axs[0, 1].plot(time_vector_42, qs_42[6, :], 'tab:blue', label="OCP without vision")
axs[0, 1].plot(time_vector_42_with_visual_criteria, qs_42_with_visual_criteria[6+4, :], 'tab:red', label="OCP with vision")
axs[0, 1].set_title("Change in elevation plane R")

axs[1, 1].plot(time_vector_42, qs_42[7, :], 'tab:blue')
axs[1, 1].plot(time_vector_42_with_visual_criteria, qs_42_with_visual_criteria[7+4, :], 'tab:red')
axs[1, 1].set_title("Elevation R")

# Left arm
axs[0, 0].plot(time_vector_42, -qs_42[8, :], 'tab:blue')
axs[0, 0].plot(time_vector_42_with_visual_criteria, -qs_42_with_visual_criteria[8+4, :], 'tab:red')
axs[0, 0].set_title("Change in elevation plane L")

axs[1, 0].plot(time_vector_42, -qs_42[9, :], 'tab:blue')
axs[1, 0].plot(time_vector_42_with_visual_criteria, -qs_42_with_visual_criteria[9+4, :], 'tab:red')
axs[1, 0].set_title("Elevation L")

# show legend below figure
axs[0, 1].legend(bbox_to_anchor=[0.8, -1.5], ncols=2, frameon=False)
plt.subplots_adjust(hspace=0.35)
plt.savefig("Graphs/compare_42_dofs.png", dpi=300)
# plt.show()


# 831 plots
fig, axs = plt.subplots(1, 3, figsize=(15, 3))
for i in range(3):
    axs[i].plot(time_vector_831, qs_831[i+3, :], 'tab:blue')
    axs[i].plot(time_vector_831_with_visual_criteria, qs_831_with_visual_criteria[i+3, :], 'tab:red')
plt.savefig("Graphs/compare_831_root.png", dpi=300)
plt.show()

fig, axs = plt.subplots(2, 3)
# Right arm
axs[0, 1].plot(time_vector_831, qs_831[6, :], 'tab:blue', label="OCP without vision")
axs[0, 1].plot(time_vector_831_with_visual_criteria, qs_831_with_visual_criteria[6+4, :], 'tab:red', label="OCP with vision")
axs[0, 1].set_title("Change in elevation plane R")

axs[1, 1].plot(time_vector_831, qs_831[7, :], 'tab:blue')
axs[1, 1].plot(time_vector_831_with_visual_criteria, qs_831_with_visual_criteria[7+4, :], 'tab:red')
axs[1, 1].set_title("Elevation R")

# Left arm
axs[0, 0].plot(time_vector_831, qs_831[10, :], 'tab:blue')
axs[0, 0].plot(time_vector_831_with_visual_criteria, qs_831_with_visual_criteria[10+4, :], 'tab:red')
axs[0, 0].set_title("Change in elevation plane L")

axs[1, 0].plot(time_vector_831, qs_831[11, :], 'tab:blue')
axs[1, 0].plot(time_vector_831_with_visual_criteria, qs_831_with_visual_criteria[11+4, :], 'tab:red')
axs[1, 0].set_title("Elevation L")

# Hips
axs[0, 2].plot(time_vector_831, qs_831[12, :], 'tab:blue')
axs[0, 2].plot(time_vector_831_with_visual_criteria, qs_831_with_visual_criteria[12+4, :], 'tab:red')
axs[0, 2].set_title("Flexion")

axs[1, 2].plot(time_vector_831, qs_831[13, :], 'tab:blue')
axs[1, 2].plot(time_vector_831_with_visual_criteria, qs_831_with_visual_criteria[13+4, :], 'tab:red')
axs[1, 2].set_title("Lateral flexion")

# show legend below figure
axs[0, 0].legend(bbox_to_anchor=[3.0, -1.5], ncols=2, frameon=False)
plt.subplots_adjust(hspace=0.35)
plt.savefig("Graphs/compare_831_dofs.png", dpi=300)
plt.show()


# ---------------------------------------- Compare projected gaze orientation ---------------------------------------- #
"""
These functions are adapted from Trampoline_EyeTracking_IMUs repo.
"""

def get_gaze_position_from_intersection(vector_origin, vector_end, facing_front_wall):
    def intersection_plane_vector(vector_origin, vector_end, planes_points, planes_normal_vector):
        vector_orientation = vector_end - vector_origin
        t = (np.dot(planes_points, planes_normal_vector) - np.dot(planes_normal_vector, vector_origin)) / np.dot(
            vector_orientation, planes_normal_vector
        )
        return vector_origin + vector_orientation * np.abs(t)

    def verify_intersection_position(vector_origin, vector_end, wall_index, bound_side, facing_front_wall):
        vector_orientation = vector_end - vector_origin
        if not facing_front_wall:
            if wall_index == 0:  # trampoline
                t = (0 - vector_origin[2]) / vector_orientation[2]
            elif wall_index == 1:  # wall front
                a = (bound_side - -bound_side) / (7.360 - 7.193)
                b = bound_side - a * 7.360
                t = (b + a * vector_origin[0] - vector_origin[1]) / (vector_orientation[1] - a * vector_orientation[0])
            elif wall_index == 2:  # ceiling
                t = (9.4620 - 1.2192 - vector_origin[2]) / vector_orientation[2]
            elif wall_index == 3:  # wall back
                t = (-8.881 - vector_origin[0]) / vector_orientation[0]
            elif wall_index == 4:  # bound right
                t = (-bound_side - vector_origin[1]) / vector_orientation[1]
            elif wall_index == 5:  # bound left
                t = (bound_side - vector_origin[1]) / vector_orientation[1]
        else:
            if wall_index == 0:  # trampoline
                t = (0 - vector_origin[2]) / vector_orientation[2]
            elif wall_index == 1:  # wall front
                t = (7.2 - vector_origin[0]) / vector_orientation[0]
            elif wall_index == 2:  # ceiling
                t = (9.4620 - 1.2192 - vector_origin[2]) / vector_orientation[2]
            elif wall_index == 3:  # wall back
                t = (-7.2 - vector_origin[0]) / vector_orientation[0]
            elif wall_index == 4:  # bound right
                t = (-bound_side - vector_origin[1]) / vector_orientation[1]
            elif wall_index == 5:  # bound left
                t = (bound_side - vector_origin[1]) / vector_orientation[1]
        return vector_origin + vector_orientation * t

    bound_side = 3 + 121 * 0.0254 / 2
    if not facing_front_wall:
        # zero is positioned at the center of the trampoline
        planes_points = np.array(
            [
                [7.193, bound_side, 0],  # trampoline
                [7.193, bound_side, 0],  # wall front
                [7.193, bound_side, 9.4620 - 1.2192],  # ceiling
                [-8.881, bound_side, 0],  # wall back
                [7.193, bound_side, 0],  # bound right
                [7.360, -bound_side, 0],  # bound left
            ]
        )

        planes_normal_vector = np.array(
            [
                [0, 0, 1],  # trampoline
                np.cross(
                    np.array([7.193, bound_side, 0]) - np.array([7.360, -bound_side, 0]), np.array([0, 0, -1])
                ).tolist(),  # wall front
                [0, 0, -1],  # ceiling
                [1, 0, 0],  # wall back
                [0, 1, 0],  # bound right
                [0, -1, 0],  # bound left
            ]
        )

        plane_bounds = [
            np.array([[-8.881, 7.360], [-bound_side, bound_side], [0, 0]]),
            np.array([[7.193, 7.360], [-bound_side, bound_side], [0, 9.4620 - 1.2192]]),
            np.array([[-8.881, 7.360], [-bound_side, bound_side], [9.4620 - 1.2192, 9.4620 - 1.2192]]),
            np.array([[-8.881, -8.881], [-bound_side, bound_side], [0, 9.4620 - 1.2192]]),
            np.array([[-8.881, 7.193], [-bound_side, -bound_side], [0, 9.4620 - 1.2192]]),
            np.array([[-8.881, 7.360], [bound_side, bound_side], [0, 9.4620 - 1.2192]]),
        ]

    else:
        # zero is positioned at the center of the trampoline
        planes_points = np.array(
            [
                [7.2, bound_side, 0],  # trampoline
                [7.2, bound_side, 0],  # wall front
                [7.2, bound_side, 9.4620 - 1.2192],  # ceiling
                [-7.2, bound_side, 0],  # wall back
                [7.2, bound_side, 0],  # bound right
                [7.2, -bound_side, 0],  # bound left
            ]
        )

        planes_normal_vector = np.array(
            [
                [0, 0, 1],  # trampoline
                [-1, 0, 0],  # wall front
                [0, 0, -1],  # ceiling
                [1, 0, 0],  # wall back
                [0, 1, 0],  # bound right
                [0, -1, 0],  # bound left
            ]
        )

        plane_bounds = [
            np.array([[-7.2, 7.2], [-bound_side, bound_side], [0, 0]]),
            np.array([[7.2, 7.2], [-bound_side, bound_side], [0, 9.4620 - 1.2192]]),
            np.array([[-7.2, 7.2], [-bound_side, bound_side], [9.4620 - 1.2192, 9.4620 - 1.2192]]),
            np.array([[-7.2, -7.2], [-bound_side, bound_side], [0, 9.4620 - 1.2192]]),
            np.array([[-7.2, 7.2], [-bound_side, -bound_side], [0, 9.4620 - 1.2192]]),
            np.array([[-7.2, 7.2], [bound_side, bound_side], [0, 9.4620 - 1.2192]]),
        ]

    gaze_positions = np.zeros((vector_origin.shape[0], 3))
    wall_indices = np.zeros((vector_origin.shape[0]))
    for i_node in range(vector_origin.shape[0]):
        intersection = []
        wall_index = None
        intersection_index = np.zeros((len(planes_points)))
        for i in range(len(planes_points)):
            current_interaction = intersection_plane_vector(
                vector_origin[i_node, :], vector_end[i_node, :], planes_points[i, :], planes_normal_vector[i, :],
            )

            if current_interaction is not None:
                bounds_bool = True
                vector_orientation = vector_end[i_node, :] - vector_origin[i_node, :]
                potential_gaze_orientation = current_interaction - vector_origin[i_node, :]
                cross_condition = np.linalg.norm(np.cross(vector_orientation, potential_gaze_orientation))
                dot_condition = np.dot(vector_orientation, potential_gaze_orientation)
                if dot_condition > 0:
                    if cross_condition > -0.01 and cross_condition < 0.01:
                        for i_bool in range(3):
                            if (
                                current_interaction[i_bool] > plane_bounds[i][i_bool, 0] - 1
                                    and current_interaction[i_bool] < plane_bounds[i][i_bool, 1] + 1
                            ):
                                a = 1
                            else:
                                bounds_bool = False
                    else:
                        bounds_bool = False
                else:
                    bounds_bool = False

            if bounds_bool:
                intersection += [current_interaction]
                intersection_index[i] = 1
                wall_index = i

        if intersection_index.sum() > 1:
            bound_crossing = np.zeros((len(np.where(intersection_index == 1)[0])))
            for idx, i in enumerate(np.where(intersection_index == 1)[0]):
                for j in range(3):
                    if plane_bounds[i][j, 0] - intersection[idx][j] > 0:
                        bound_crossing[idx] += np.abs(plane_bounds[i][j, 0] - intersection[idx][j])
                    if plane_bounds[i][j, 1] - intersection[idx][j] < 0:
                        bound_crossing[idx] += np.abs(plane_bounds[i][j, 1] - intersection[idx][j])
            closest_index = np.argmin(bound_crossing)
            wall_index = np.where(intersection_index == 1)[0][closest_index]

        if wall_index is not None:
            gaze_position = verify_intersection_position(vector_origin[i_node, :], vector_end[i_node, :], wall_index, bound_side, facing_front_wall)
        else:
            gaze_position = None
        gaze_positions[i_node, :] = gaze_position
        wall_indices[i_node] = wall_index

    return gaze_positions, wall_indices



def plot_gaze_trajectory(
        gaze_position_temporal_evolution_projected,
        gaze_position_temporal_evolution_projected_with_visual_criteria,
        output_file_name,
):
    """
    This function plots the gaze trajectory and the fixation positions projected on the gymnasium in 3D.
    """

    def plot_gymnasium_symmetrized(bound_side, ax):
        """
        Plot the gymnasium in 3D with the walls and trampoline bed.
        """
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=10.0, azim=-70)

        ax.set_xlim3d([-8.0, 8.0])
        ax.set_ylim3d([-8.0, 8.0])
        ax.set_zlim3d([-3.0, 13.0])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # Front right, to front left (bottom)
        plt.plot(np.array([7.2, 7.2]), np.array([-bound_side, bound_side]), np.array([0, 0]), "-k")
        # Front right, to back right (bottom)
        plt.plot(np.array([-7.2, 7.2]), np.array([-bound_side, -bound_side]), np.array([0, 0]), "-k")
        # Front left, to back left (bottom)
        plt.plot(np.array([-7.2, 7.2]), np.array([bound_side, bound_side]), np.array([0, 0]), "-k")
        # Back right, to back left (bottom)
        plt.plot(np.array([-7.2, -7.2]), np.array([-bound_side, bound_side]), np.array([0, 0]), "-k")

        # Front right, to front left (ceiling)
        plt.plot(
            np.array([7.2, 7.2]),
            np.array([-bound_side, bound_side]),
            np.array([9.4620 - 1.2192, 9.4620 - 1.2192]),
            "-k")

        # Front right, to back right (ceiling)
        plt.plot(
            np.array([-7.2, 7.2]),
            np.array([-bound_side, -bound_side]),
            np.array([9.4620 - 1.2192, 9.4620 - 1.2192]),
            "-k",
        )
        # Front left, to back left (ceiling)
        plt.plot(
            np.array([-7.2, 7.2]),
            np.array([bound_side, bound_side]),
            np.array([9.4620 - 1.2192, 9.4620 - 1.2192]),
            "-k",
        )
        # Back right, to back left (ceiling)
        plt.plot(
            np.array([-7.2, -7.2]),
            np.array([-bound_side, bound_side]),
            np.array([9.4620 - 1.2192, 9.4620 - 1.2192]),
            "-k",
        )

        # Front right bottom, to front right ceiling
        plt.plot(np.array([7.2, 7.2]), np.array([-bound_side, -bound_side]), np.array([0, 9.4620 - 1.2192]), "-k")
        # Front left bottom, to front left ceiling
        plt.plot(np.array([7.2, 7.2]), np.array([bound_side, bound_side]), np.array([0, 9.4620 - 1.2192]), "-k")
        # Back right bottom, to back right ceiling
        plt.plot(np.array([-7.2, -7.2]), np.array([-bound_side, -bound_side]), np.array([0, 9.4620 - 1.2192]), "-k")
        # Back left bottom, to back left ceiling
        plt.plot(np.array([-7.2, -7.2]), np.array([bound_side, bound_side]), np.array([0, 9.4620 - 1.2192]), "-k")

        # Trampoline
        X, Y = np.meshgrid([-7 * 0.3048, 7 * 0.3048], [-3.5 * 0.3048, 3.5 * 0.3048])
        Z = np.zeros(X.shape)
        ax.plot_surface(X, Y, Z, color="k", alpha=0.4)
        return

    bound_side = 3 + 121 * 0.0254 / 2

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    plot_gymnasium_symmetrized(bound_side, ax)

    N = len(gaze_position_temporal_evolution_projected[:, 0]) - 1
    for j in range(N):
        ax.plot(
            gaze_position_temporal_evolution_projected[j: j + 2, 0],
            gaze_position_temporal_evolution_projected[j: j + 2, 1],
            gaze_position_temporal_evolution_projected[j: j + 2, 2],
            color=plt.cm.winter(j / N),
        )
        ax.plot(
            gaze_position_temporal_evolution_projected_with_visual_criteria[j: j + 2, 0],
            gaze_position_temporal_evolution_projected_with_visual_criteria[j: j + 2, 1],
            gaze_position_temporal_evolution_projected_with_visual_criteria[j: j + 2, 2],
            color=plt.cm.autumn(j / N),
        )
        # spring for real athlete

    ax.set_title("Gaze trajectory")

    plt.savefig(output_file_name + "_gaze_trajectory.png", dpi=300)
    # plt.show()
    # plt.close("all")
    return

def plot_unwrapped_trajectories(
        gaze_position,
        gaze_position_with_visual_criteria,
        wall_index,
        wall_index_with_visual_criteria,
        output_filename,
):
    def plot_gymnasium_unwrapped(axs):

        bound_side = 3 + 121 * 0.0254 / 2

        # Plot trampo bed
        axs.add_patch(Rectangle((-7 * 0.3048, -3.5 * 0.3048), 14 * 0.3048, 7 * 0.3048, facecolor='k', alpha=0.2))
        # Plot vertical lines of the symmetrized gymnasium
        axs.plot(np.array([-7.2, -7.2]), np.array([-bound_side, bound_side]), '-k')
        axs.plot(np.array([7.2, 7.2]), np.array([-bound_side, bound_side]), '-k')
        axs.plot(np.array([-7.2 - (9.4620 - 1.2192), -7.2 - (9.4620 - 1.2192)]), np.array([-bound_side, bound_side]),
                    '-k')
        axs.plot(np.array([7.2 + 9.4620 - 1.2192, 7.2 + 9.4620 - 1.2192]), np.array([-bound_side, bound_side]), '-k')
        axs.plot(np.array([-7.2 - (9.4620 - 1.2192) - 2 * 7.2, -7.2 - (9.4620 - 1.2192) - 2 * 7.2]),
                    np.array([-bound_side, bound_side]), '-k')
        axs.plot(np.array([-7.2, -7.2]), np.array([-bound_side - (9.4620 - 1.2192), bound_side]), '-k')
        axs.plot(np.array([7.2, 7.2]), np.array([-bound_side - (9.4620 - 1.2192), bound_side]), '-k')
        axs.plot(np.array([-7.2, -7.2]), np.array([bound_side, bound_side + 9.4620 - 1.2192]), '-k')
        axs.plot(np.array([7.2, 7.2]), np.array([bound_side, bound_side + 9.4620 - 1.2192]), '-k')
        # Plot horizontal lines of the symmetrized gymnasium
        axs.plot(np.array([-7.2, 7.2]), np.array([-bound_side, -bound_side]), '-k')
        axs.plot(np.array([-7.2, 7.2]), np.array([bound_side, bound_side]), '-k')
        axs.plot(np.array([-7.2, 7.2]), np.array([-bound_side - (9.4620 - 1.2192), -bound_side - (9.4620 - 1.2192)]),
                    '-k')
        axs.plot(np.array([-7.2, 7.2]), np.array([bound_side + 9.4620 - 1.2192, bound_side + 9.4620 - 1.2192]), '-k')
        axs.plot(np.array([7.2, 7.2 + 9.4620 - 1.2192]), np.array([-bound_side, -bound_side]), '-k')
        axs.plot(np.array([7.2, 7.2 + 9.4620 - 1.2192]), np.array([bound_side, bound_side]), '-k')
        axs.plot(np.array([-7.2 - (9.4620 - 1.2192), 7.2]), np.array([-bound_side, -bound_side]), '-k')
        axs.plot(np.array([-7.2 - (9.4620 - 1.2192), 7.2]), np.array([bound_side, bound_side]), '-k')
        axs.plot(np.array([-7.2 - (9.4620 - 1.2192) - 2 * 7.2, -7.2 - (9.4620 - 1.2192)]),
                    np.array([-bound_side, -bound_side]), '-k')
        axs.plot(np.array([-7.2 - (9.4620 - 1.2192) - 2 * 7.2, -7.2 - (9.4620 - 1.2192)]),
                    np.array([bound_side, bound_side]), '-k')

        axs.text(-7.2 - (9.4620 - 1.2192) - 2 * 7.2 + 7.2 / 2 + 1, bound_side + 0.2, "Ceiling", fontsize=10)
        axs.text(-7.2 - (9.4620 - 1.2192) + 1, bound_side + 0.2, "Back wall", fontsize=10)
        axs.text(7.2 + 1, bound_side + 0.2, "Front wall", fontsize=10)
        axs.text(-7.2 + 7.2 / 2 + 1, bound_side + 9.4620 - 1.2192 + 0.2, "Left wall", fontsize=10)
        axs.text(-7.2 + 7.2 / 2 + 0.5, -bound_side - (9.4620 - 1.2192) - 1.2, "Right wall", fontsize=10)

        return

    def unwrap_gaze_positions(gaze_position, wall_index):
        bound_side = 3 + 121 * 0.0254 / 2

        gaze_position_x_y = np.zeros((2, np.shape(wall_index)[0]))
        gaze_position_x_y[:, :] = np.nan
        for i in range(len(wall_index)):
            if wall_index[i] == 0:  # trampoline
                gaze_position_x_y[:, i] = gaze_position[i][:2]
            if wall_index[i] == 1:  # wall front
                gaze_position_x_y[:, i] = [gaze_position[i][2] + 7.2, gaze_position[i][1]]
            elif wall_index[i] == 2:  # ceiling
                gaze_position_x_y[:, i] = [-7.2 - (9.4620 - 1.2192) - 7.2 - gaze_position[i][0], gaze_position[i][1]]
            elif wall_index[i] == 3:  # wall back
                gaze_position_x_y[:, i] = [-7.2 - gaze_position[i][2], gaze_position[i][1]]
            elif wall_index[i] == 4:  # bound right
                gaze_position_x_y[:, i] = [gaze_position[i][0], -bound_side - gaze_position[i][2]]
            elif wall_index[i] == 5:  # bound left
                gaze_position_x_y[:, i] = [gaze_position[i][0], bound_side + gaze_position[i][2]]
        return gaze_position_x_y

    fig, axs = plt.subplots(1, 1, figsize=(9, 6))

    unwrapped_gaze_position = unwrap_gaze_positions(gaze_position, wall_index)
    unwrapped_gaze_position_with_visual_criteria = unwrap_gaze_positions(gaze_position_with_visual_criteria, wall_index_with_visual_criteria)

    axs.scatter(unwrapped_gaze_position[0, :],
                unwrapped_gaze_position[1, :],
                c=np.linspace(0, 1, unwrapped_gaze_position.shape[1]),
                cmap='winter', marker='.')
    axs.scatter(unwrapped_gaze_position_with_visual_criteria[0, :],
                unwrapped_gaze_position_with_visual_criteria[1, :],
                c=np.linspace(0, 1, unwrapped_gaze_position_with_visual_criteria.shape[1]),
                cmap='autumn', marker='.')

    plot_gymnasium_unwrapped(axs)
    axs.axis('equal')

    plt.subplots_adjust(right=0.8)
    plt.savefig(output_filename, dpi=300)
    # plt.show()
    # plt.close('all')
    return

def find_eye_markers_index(model):
    marker_names = model.markerNames()
    for i_marker, marker in enumerate(marker_names):
        if marker.to_string() == "eyes_vect_start":
            eyes_start_idx = i_marker
        elif marker.to_string() == "eyes_vect_end":
            eyes_end_idx = i_marker
    return eyes_start_idx, eyes_end_idx


model_42_eyes_start_idx, model_42_eyes_end_idx = find_eye_markers_index(model_42)
model_42_with_visual_criteria_eyes_start_idx, model_42_with_visual_criteria_eyes_end_idx = find_eye_markers_index(model_42_with_visual_criteria)
model_831_eyes_start_idx, model_831_eyes_end_idx = find_eye_markers_index(model_831)
model_831_with_visual_criteria_eyes_start_idx, model_831_with_visual_criteria_eyes_end_idx = find_eye_markers_index(model_831_with_visual_criteria)

# Coordinate system of the gymnasium (x-orientation for xsens is front wall, x-orientation for OCP is right wall)
rotation_matrix = biorbd.Rotation.fromEulerAngles(np.array([np.pi/2]), 'z').to_array()

vector_origin_42 = np.zeros((time_vector_42.shape[0], 3))
vector_end_42 = np.zeros((time_vector_42.shape[0], 3))
vector_origin_42_with_visual_criteria = np.zeros((time_vector_42_with_visual_criteria.shape[0], 3))
vector_end_42_with_visual_criteria = np.zeros((time_vector_42_with_visual_criteria.shape[0], 3))
for i_node in range(time_vector_42.shape[0]):
    vector_origin_42[i_node, :] = rotation_matrix @ model_42.markers(qs_42[:, i_node])[model_42_eyes_start_idx].to_array()
    vector_end_42[i_node, :] = rotation_matrix @ model_42.markers(qs_42[:, i_node])[model_42_eyes_end_idx].to_array()
    vector_origin_42_with_visual_criteria[i_node, :] = rotation_matrix @ model_42_with_visual_criteria.markers(qs_42_with_visual_criteria[:, i_node])[model_42_with_visual_criteria_eyes_start_idx].to_array()
    vector_end_42_with_visual_criteria[i_node, :] = rotation_matrix @ model_42_with_visual_criteria.markers(qs_42_with_visual_criteria[:, i_node])[model_42_with_visual_criteria_eyes_end_idx].to_array()

gaze_position_42, wall_index_42 = get_gaze_position_from_intersection(vector_origin_42, vector_end_42, True)
gaze_position_42_with_visual_criteria, wall_index_42_with_visual_criteria = get_gaze_position_from_intersection(vector_origin_42_with_visual_criteria, vector_end_42_with_visual_criteria, True)

plot_gaze_trajectory(gaze_position_42,
                     gaze_position_42_with_visual_criteria,
                     "Graphs/compare_42")

plot_unwrapped_trajectories(gaze_position_42,
                            gaze_position_42_with_visual_criteria,
                            wall_index_42,
                            wall_index_42_with_visual_criteria,
                            "Graphs/compare_42_unwrapped")


vector_origin_831 = np.zeros((time_vector_831.shape[0], 3))
vector_end_831 = np.zeros((time_vector_831.shape[0], 3))
vector_origin_831_with_visual_criteria = np.zeros((time_vector_831_with_visual_criteria.shape[0], 3))
vector_end_831_with_visual_criteria = np.zeros((time_vector_831_with_visual_criteria.shape[0], 3))
for i_node in range(time_vector_831.shape[0]):
    vector_origin_831[i_node, :] = rotation_matrix @ model_831.markers(qs_831[:, i_node])[model_831_eyes_start_idx].to_array()
    vector_end_831[i_node, :] = rotation_matrix @ model_831.markers(qs_831[:, i_node])[model_831_eyes_end_idx].to_array()
    vector_origin_831_with_visual_criteria[i_node, :] = rotation_matrix @ model_831_with_visual_criteria.markers(qs_831_with_visual_criteria[:, i_node])[model_831_with_visual_criteria_eyes_start_idx].to_array()
    vector_end_831_with_visual_criteria[i_node, :] = rotation_matrix @ model_831_with_visual_criteria.markers(qs_831_with_visual_criteria[:, i_node])[model_831_with_visual_criteria_eyes_end_idx].to_array()

gaze_position_831, wall_index_831 = get_gaze_position_from_intersection(vector_origin_831, vector_end_831, True)
gaze_position_831_with_visual_criteria, wall_index_831_with_visual_criteria = get_gaze_position_from_intersection(vector_origin_831_with_visual_criteria, vector_end_831_with_visual_criteria, True)

plot_gaze_trajectory(gaze_position_831,
                     gaze_position_831_with_visual_criteria,
                     "Graphs/compare_42")

plot_unwrapped_trajectories(gaze_position_831,
                            gaze_position_831_with_visual_criteria,
                            wall_index_831,
                            wall_index_831_with_visual_criteria,
                            "Graphs/compare_831_unwrapped")

plt.show()






