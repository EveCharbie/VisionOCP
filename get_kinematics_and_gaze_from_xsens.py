"""
The goal of this program is to convert the joint angles from xsens to biorbd.
"""

import numpy as np
import pickle
from IPython import embed
import matplotlib.pyplot as plt
import os
import biorbd
import bioviz

joint_labels = [
    "jL5S1_x",  # 0
    "jL5S1_y",  # 1
    "jL5S1_z",  # 2
    "jL4L3_x",  # 3
    "jL4L3_y",  # 4
    "jL4L3_z",  # 5
    "jL1T12_x",  # 6
    "jL1T12_y",  # 7
    "jL1T12_z",  # 8
    "jT9T8_x",  # 9
    "jT9T8_y",  # 10
    "jT9T8_z",  # 11
    "jT1C7_x",  # 12
    "jT1C7_y",  # 13
    "jT1C7_z",  # 14
    "jC1Head_x",  # 15
    "jC1Head_y",  # 16
    "jC1Head_z",  # 17
    "jRightT4Shoulder…",  # 18
    "jRightT4Shoulder…",  # 19
    "jRightT4Shoulder…",  # 20
    "jRightShoulder_x",  # 21
    "jRightShoulder_y",  # 22
    "jRightShoulder_z",  # 23
    "jRightElbow_x",  # 24
    "jRightElbow_y",  # 25
    "jRightElbow_z",  # 26
    "jRightWrist_x",  # 27
    "jRightWrist_y",  # 28
    "jRightWrist_z",  # 29
    "jLeftT4Shoulder_x",  # 30
    "jLeftT4Shoulder_y",  # 31
    "jLeftT4Shoulder_z",  # 32
    "jLeftShoulder_x",  # 33
    "jLeftShoulder_y",  # 34
    "jLeftShoulder_z",  # 35
    "jLeftElbow_x",  # 36
    "jLeftElbow_y",  # 37
    "jLeftElbow_z",  # 38
    "jLeftWrist_x",  # 39
    "jLeftWrist_y",  # 40
    "jLeftWrist_z",  # 41
    "jRightHip_x",  # 42
    "jRightHip_y",  # 43
    "jRightHip_z",  # 44
    "jRightKnee_x",  # 45
    "jRightKnee_y",  # 46
    "jRightKnee_z",  # 47
    "jRightAnkle_x",  # 48
    "jRightAnkle_y",  # 49
    "jRightAnkle_z",  # 50
    "jRightBallFoot_x",  # 51
    "jRightBallFoot_y",  # 52
    "jRightBallFoot_z",  # 53
    "jLeftHip_x",  # 54
    "jLeftHip_y",  # 55
    "jLeftHip_z",  # 56
    "jLeftKnee_x",  # 57
    "jLeftKnee_y",  # 58
    "jLeftKnee_z",  # 59
    "jLeftAnkle_x",  # 60
    "jLeftAnkle_y",  # 61
    "jLeftAnkle_z",  # 62
    "jLeftBallFoot_x",  # 63
    "jLeftBallFoot_y",  # 64
    "jLeftBallFoot_z",  # 65
]  # 66


# Load the data
move_path = "/home/charbie/disk/Eye-tracking/Results_831/SoMe/42/"
save_path = "/home/charbie/Documents/Programmation/VisionOCP/Kalman_recons/"

# for filename in os.listdir(move_path):
for i in range(1):
    filename = "a62d4691_0_0-45_796__42__0__eyetracking_metrics.pkl"
    if filename[-24:] == "_eyetracking_metrics.pkl":
        move_filename = move_path + filename

    biorbd_model_path = "models/SoMe_Xsens_Model.bioMod"

    with open(move_filename, "rb") as f:
        data = pickle.load(f)
        subject_name = data["subject_name"]
        gaze_position_temporal_evolution_projected_facing_front_wall = data[
            "gaze_position_temporal_evolution_projected_facing_front_wall"]
        move_orientation = data["move_orientation"]
        Xsens_head_position_calculated = data["Xsens_head_position_calculated"]
        eye_position = data["eye_position"]
        gaze_orientation = data["gaze_orientation"]
        EulAngles_head_global = data["EulAngles_head_global"]
        EulAngles_neck = data["EulAngles_neck"]
        eye_angles = data["eye_angles"]
        Xsens_orthogonal_thorax_position = data["Xsens_orthogonal_thorax_position"]
        Xsens_orthogonal_head_position = data["Xsens_orthogonal_head_position"]
        Xsens_position_no_level_CoM_corrected_rotated_per_move = data[
            "Xsens_position_no_level_CoM_corrected_rotated_per_move"]
        Xsens_jointAngle_per_move = data["Xsens_jointAngle_per_move"]

    # get markers position from the biorbd model
    model = biorbd.Model(biorbd_model_path)
    num_dofs = model.nbQ()


    DoFs = np.zeros((num_dofs, len(Xsens_jointAngle_per_move)))
    # DoFs[0, :] = Xsens_jointAngle_per_move[:, 0]  # Trans X
    # DoFs[1, :] = Xsens_jointAngle_per_move[:, 1]  # Trans Y
    # DoFs[2, :] = Xsens_jointAngle_per_move[:, 2]  # Trans Z
    DoFs[3, :] = Xsens_jointAngle_per_move[:, 3]  # Rot X
    # DoFs[4, :] = Xsens_jointAngle_per_move[:, 4]  # Rot Y
    # DoFs[5, :] = Xsens_jointAngle_per_move[:, 5]  # Rot Z

    b = bioviz.Viz(model_path=biorbd_model_path)
    b.load_movement(DoFs)
    b.exec()


body_height = 1.545
shoulder_height = 1.248
hip_height = 0.79
knee_height = 0.48
ankle_height = 0.07
foot_length = 0.21
hip_width = 0.27
shoulder_width = 0.39
elbow_span = 0.80
wrist_span = 1.215
arm_span = 1.525
eye_height = 5.64622475933951
eye_depth = 8.52029331945667

T8_height = (shoulder_height - hip_height) / 2
C7_height = (body_height - shoulder_height) / 3
Shoulder_height = T8_height
Shoulder_lateral_position = (shoulder_width - elbow_span) / 2
forearm_length = (elbow_span - wrist_span) / 2
hand_length = (wrist_span - arm_span) / 2
upper_leg_lateral_position = -0.5*hip_width
lower_leg_height = -(hip_height-knee_height)
foot_height = -(knee_height-ankle_height)






