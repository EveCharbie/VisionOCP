"""
The goal of this program is to show an animation of the solution of the optimal control problem.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import IPython
import time

import bioviz
import bioptim
import biorbd

WITH_VISUAL_CRITERIA = True
ACROBATICS = "831"

if ACROBATICS == "831":
    if WITH_VISUAL_CRITERIA:
        biorbd_model_path = "models/SoMe_with_visual_criteria.bioMod"
    else:
        biorbd_model_path = "models/SoMe.bioMod"
    n_shooting = (40, 100, 100, 100, 40)
    num_twists = 1
    name = "SoMe"
elif ACROBATICS == "42":
    if WITH_VISUAL_CRITERIA:
        biorbd_model_path = "models/SoMe_42_with_visual_criteria.bioMod"
    else:
        biorbd_model_path = "models/SoMe_42.bioMod"
    n_shooting = (100, 40)
    num_twists = 1
    name = "SoMe"

# file_name = "SoMe_831-(40_100_100_100_40)-2023-10-19-1601.pkl"  # Good 831<
file_name = "SoMe_with_visual_criteria_831-(40_100_100_100_40)-2023-10-21-0915.pkl"  # Good 831< with visual criteria
# file_name = "SoMe_42_without_mesh-(100_40)-2023-10-20-1652.pkl"  # Good 42/
# file_name = "SoMe_42_with_visual_criteria_without_mesh-(100_40)-2023-10-20-1631.pkl"  # Good 42/ with visual criteria

with open("Solutions/" + file_name, "rb") as f:
    data = pickle.load(f)
    sol = data[0]
    q_per_phase = data[1]
    qs = data[2]
    qdots = data[3]
    qddots = data[4]
    time_parameters = data[5]
    q_reintegrated = data[6]
    qdot_reintegrated = data[7]
    time_vector = data[8]


# if ACROBATICS == "831":
#     from TechOpt831 import prepare_ocp
#     sol.ocp = prepare_ocp(biorbd_model_path, n_shooting=n_shooting, num_twists=num_twists, n_threads=7, WITH_VISUAL_CRITERIA=WITH_VISUAL_CRITERIA)
# elif ACROBATICS == "42":
#     from TechOpt42 import prepare_ocp
#
#     sol.ocp = prepare_ocp(biorbd_model_path, n_shooting=n_shooting, num_twists=num_twists, n_threads=7,
#                 WITH_VISUAL_CRITERIA=WITH_VISUAL_CRITERIA)

b = bioviz.Viz(biorbd_model_path)
b.load_movement(qs)
b.exec()

# sol.graphs(show_bounds=True)


# measure the value of the custom_trampoline_bed_in_peripheral_vision function
def custom_trampoline_bed_in_peripheral_vision(model, q):

    a = 1.07  # Trampoline with/2
    b = 2.14  # Trampoline length/2
    n = 6  # order of the polynomial for the trampoline bed rectangle equation

    # Get the gaze vector
    eyes_vect_start_marker_idx = 6
    eyes_vect_end_marker_idx = 7
    gaze_vector = model.markers(q)[eyes_vect_end_marker_idx].to_array() - model.markers(q)[eyes_vect_start_marker_idx].to_array()

    first_condition = 0
    second_condition = 0
    third_condition = 0
    point_in_the_plane = np.array([1, 2, -0.83])
    vector_normal_to_the_plane = np.array([0, 0, 1])
    obj = 0
    for i_r in range(11):
        for i_th in range(10):

            # Get this vector from the vision cone
            marker_idx = 9 + i_r * 10 + i_th
            vector_origin = model.markers(q)[eyes_vect_start_marker_idx].to_array()
            vector_end = model.markers(q)[marker_idx].to_array()
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
        val = 2*10*11
        first_condition = 1
    elif np.abs(gaze_vector[0]/gaze_vector[2]) > np.tan(3*np.pi/8):
        val = 2*10*11
        second_condition = 1
    elif np.abs(gaze_vector[1]/gaze_vector[2]) > np.tan(3*np.pi/8):
        val = 2*10*11
        third_condition = 1
    else:
        val = obj

    return val, first_condition, second_condition, third_condition


model = biorbd.Model(biorbd_model_path)
num_frames = qs.shape[1]
obj = np.zeros(num_frames)
first_condition = np.zeros(num_frames)
second_condition = np.zeros(num_frames)
third_condition = np.zeros(num_frames)
for i in range(num_frames):
    obj[i], first_condition[i], second_condition[i], third_condition[i] = custom_trampoline_bed_in_peripheral_vision(model, qs[:, i])

plt.figure()
plt.plot(obj, 'k')
plt.plot(first_condition * 220, 'r')
plt.plot(second_condition * 220, 'g')
plt.plot(third_condition * 220, 'b')
plt.show()