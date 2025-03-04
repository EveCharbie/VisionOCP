"""
The goal of this program is to show an animation of the solution of the optimal control problem.
"""

import numpy as np
import pickle

import bioviz
import biorbd


def combine_real_time(interpolated_states_with_visual_criteria, interpolated_states):
    qs_combined_real_time = None
    for i in range(len(interpolated_states_with_visual_criteria)):
        # Make qs the right size to be concatenated (removing one frame at the end is necessary)
        q_with_visual_criteria = interpolated_states_with_visual_criteria[i]["q"]
        q = interpolated_states[i]["q"]
        if q_with_visual_criteria.shape[1] > q.shape[1]:
            q_with_visual_criteria = q_with_visual_criteria[:, :q.shape[1]]
        elif q_with_visual_criteria.shape[1] < q.shape[1]:
            q = q[:, :q_with_visual_criteria.shape[1]]
        # Combining them into one matrix
        if qs_combined_real_time is None:
            qs_combined_real_time = np.vstack((q_with_visual_criteria, q))
        else:
            qs_combined_real_time = np.hstack((qs_combined_real_time,
                                               np.vstack((q_with_visual_criteria, q))))
    return qs_combined_real_time


# Models
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
biorbd_model_path_831_with_visual_criteria_red = "models/SoMe_with_visual_criteria_red.bioMod"
biorbd_model_path_42_with_visual_criteria_red = "models/SoMe_42_with_visual_criteria_red.bioMod"
biorbd_model_path_831_blue = "models/SoMe_blue.bioMod"
biorbd_model_path_42_blue = "models/SoMe_42_blue.bioMod"


# Paths to techniques
file_name_831 = "Solutions/q_SoMe_without_mesh_831-(40_40_40_40_40_40)-2023-11-13-1528-0p0_CVG.pkl"  # Good 831<
file_name_831_with_visual_criteria = "Solutions/q_SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-2023-11-07-2257-0p25_CVG.pkl"  # Good 831< with visual criteria
file_name_42 = "Solutions/q_SoMe_42_without_mesh-(100_40)-2023-11-01-0932-0p0_CVG.pkl"  # Good 42/
file_name_42_with_visual_criteria = "Solutions/q_SoMe_42_with_visual_criteria_without_mesh-(100_40)-2023-11-01-0942-0p25_CVG.pkl"   # Good 42/ with visual criteria


# Flags
animate_comparison_FLAG = False
animate_independently_FLAG = True

# ---------------------------------------- Load data ---------------------------------------- #

with open(file_name_42, "rb") as f:
    data = pickle.load(f)
    q_per_phase_42 = data["q_per_phase"]
    qs_42 = data["qs"]
    qdots_42 = data["qdots"]
    qddots_42 = data["qddots"]
    time_parameters_42 = data["time_parameters"]
    q_reintegrated_42 = data["q_reintegrated"]
    qdot_reintegrated_42 = data["qdot_reintegrated"]
    time_vector_42 = data["time_vector"]
    interpolated_states_42 = data["interpolated_states"]
    q_interpolated_42 = np.hstack((data["interpolated_states"][0]["q"], data["interpolated_states"][1]["q"]))

with open(file_name_42_with_visual_criteria, "rb") as f:
    data = pickle.load(f)
    q_per_phase_42_with_visual_criteria = data["q_per_phase"]
    qs_42_with_visual_criteria = data["qs"]
    qdots_42_with_visual_criteria = data["qdots"]
    qddots_42_with_visual_criteria = data["qddots"]
    time_parameters_42_with_visual_criteria = data["time_parameters"]
    q_reintegrated_42_with_visual_criteria = data["q_reintegrated"]
    qdot_reintegrated_42_with_visual_criteria = data["qdot_reintegrated"]
    time_vector_42_with_visual_criteria = data["time_vector"]
    interpolated_states_42_with_visual_criteria = data["interpolated_states"]
    q_interpolated_42_with_visual_criteria = np.hstack((data["interpolated_states"][0]["q"], data["interpolated_states"][1]["q"]))

with open(file_name_831, "rb") as f:
    data = pickle.load(f)
    q_per_phase_831 = data["q_per_phase"]
    qs_831 = data["qs"]
    qdots_831 = data["qdots"]
    qddots_831 = data["qddots"]
    time_parameters_831 = data["time_parameters"]
    q_reintegrated_831 = data["q_reintegrated"]
    qdot_reintegrated_831 = data["qdot_reintegrated"]
    time_vector_831 = data["time_vector"]
    interpolated_states_831 = data["interpolated_states"]
    q_interpolated_831 = np.hstack((data["interpolated_states"][0]["q"],
                                    data["interpolated_states"][1]["q"],
                                    data["interpolated_states"][2]["q"],
                                    data["interpolated_states"][3]["q"],
                                    data["interpolated_states"][4]["q"],
                                    data["interpolated_states"][5]["q"]))

with open(file_name_831_with_visual_criteria, "rb") as f:
    data = pickle.load(f)
    q_per_phase_831_with_visual_criteria = data["q_per_phase"]
    qs_831_with_visual_criteria = data["qs"]
    qdots_831_with_visual_criteria = data["qdots"]
    qddots_831_with_visual_criteria = data["qddots"]
    time_parameters_831_with_visual_criteria = data["time_parameters"]
    q_reintegrated_831_with_visual_criteria = data["q_reintegrated"]
    qdot_reintegrated_831_with_visual_criteria = data["qdot_reintegrated"]
    time_vector_831_with_visual_criteria = data["time_vector"]
    interpolated_states_831_with_visual_criteria = data["interpolated_states"]
    q_interpolated_831_with_visual_criteria = np.hstack((data["interpolated_states"][0]["q"],
                                                        data["interpolated_states"][1]["q"],
                                                        data["interpolated_states"][2]["q"],
                                                        data["interpolated_states"][3]["q"],
                                                        data["interpolated_states"][4]["q"],
                                                        data["interpolated_states"][5]["q"]))


# ---------------------------------------- Animate comparison ---------------------------------------- #
if animate_comparison_FLAG:
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
    b.set_camera_focus_point(0, 0, 2.5)
    b.set_camera_zoom(0.25)
    b.exec(frame_step=20,
           save_path="Kinograms/42_both.svg")

    qs_42_combined_real_time = combine_real_time(interpolated_states_42_with_visual_criteria, interpolated_states_42)
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
    b.set_camera_focus_point(0, 0, 2.5)
    b.exec()

    qs_42_combined_array = np.hstack((q_per_phase_42_combined[0][:, :-1], q_per_phase_42_combined[1]))
    qs_42_combined_array[15, :] -= 2

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
    b.load_movement(qs_42_combined_array)
    b.set_camera_zoom(0.25)
    b.set_camera_focus_point(0, 0, 1.5) ##
    b.maximize()
    b.update()
    b.start_recording(f"comp_42.ogv")
    for frame in range(qs_42_combined_array.shape[1] + 1):
        b.movement_slider[0].setValue(frame)
        b.add_frame()
    b.stop_recording()
    b.quit()

    # Animate comparison
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
    b.set_camera_focus_point(0, 0, 2.5)
    b.exec(frame_step=20,
           save_path="Kinograms/831_both.svg")

    qs_831_combined_real_time = combine_real_time(interpolated_states_831_with_visual_criteria, interpolated_states_831)
    b = bioviz.Viz(biorbd_model_path_831_both,
                   mesh_opacity=0.8,
                   show_global_center_of_mass=False,
                   show_gravity_vector=False,
                   show_segments_center_of_mass=False,
                   show_global_ref_frame=False,
                   show_local_ref_frame=False,
                   experimental_markers_color=(1, 1, 1),
                   background_color=(1.0, 1.0, 1.0),
                   )
    b.load_movement(qs_831_combined_real_time)
    b.set_camera_zoom(0.25)
    b.set_camera_focus_point(0, 0, 2.5)
    b.exec()



# ---------------------------------------- Animate independently ----------------------------------------------------- #
if animate_independently_FLAG:

    b = bioviz.Kinogram(model_path=biorbd_model_path_42_blue,
                       mesh_opacity=0.8,
                       show_global_center_of_mass=False,
                       show_gravity_vector=False,
                       show_segments_center_of_mass=False,
                       show_global_ref_frame=False,
                       show_local_ref_frame=False,
                       experimental_markers_color=(1, 1, 1),
                       background_color=(1.0, 1.0, 1.0),
                        )
    b.load_movement(q_interpolated_42)
    b.set_camera_focus_point(0, 0, 2.5)
    b.set_camera_zoom(0.25)
    b.exec(frame_step=10,
           save_path=f"Kinograms/{file_name_42}.svg")

    b = bioviz.Kinogram(model_path=biorbd_model_path_42_with_visual_criteria_red,
                       mesh_opacity=0.8,
                       show_global_center_of_mass=False,
                       show_gravity_vector=False,
                       show_segments_center_of_mass=False,
                       show_global_ref_frame=False,
                       show_local_ref_frame=False,
                       experimental_markers_color=(1, 1, 1),
                       background_color=(1.0, 1.0, 1.0),
                        )
    b.load_movement(q_interpolated_42_with_visual_criteria)
    b.set_camera_focus_point(0, 0, 2.5)
    b.set_camera_zoom(0.25)
    b.exec(frame_step=10,
           save_path=f"Kinograms/{file_name_42_with_visual_criteria}.svg")

    b = bioviz.Kinogram(model_path=biorbd_model_path_831_blue,
                       mesh_opacity=0.8,
                       show_global_center_of_mass=False,
                       show_gravity_vector=False,
                       show_segments_center_of_mass=False,
                       show_global_ref_frame=False,
                       show_local_ref_frame=False,
                       experimental_markers_color=(1, 1, 1),
                       background_color=(1.0, 1.0, 1.0),
                        )
    b.load_movement(q_interpolated_831)
    b.set_camera_focus_point(0, 0, 2.5)
    b.set_camera_zoom(0.25)
    b.exec(frame_step=10,
           save_path=f"Kinograms/{file_name_831}.svg")

    b = bioviz.Kinogram(model_path=biorbd_model_path_831_with_visual_criteria_red,
                       mesh_opacity=0.8,
                       show_global_center_of_mass=False,
                       show_gravity_vector=False,
                       show_segments_center_of_mass=False,
                       show_global_ref_frame=False,
                       show_local_ref_frame=False,
                       experimental_markers_color=(1, 1, 1),
                       background_color=(1.0, 1.0, 1.0),
                        )
    b.load_movement(q_interpolated_831_with_visual_criteria)
    b.set_camera_focus_point(0, 0, 2.5)
    b.set_camera_zoom(0.25)
    b.exec(frame_step=10,
           save_path=f"Kinograms/{file_name_831_with_visual_criteria}.svg")
