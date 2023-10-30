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

def get_q(Xsens_orientation_per_move):
    """
    This function returns de generalized coordinates in the sequence XYZ (biorbd) from the quaternion of the orientation
    of the Xsens segments.
    The translation is left empty as it has to be computed otherwise.
    I am not sure if I would use this for kinematics analysis, but for visualisation it is not that bad.
    """

    parent_idx_list = {"Pelvis": None,  # 0
                       "L5": [0, "Pelvis"],  # 1
                       "L3": [1, "L5"],  # 2
                       "T12": [2, "L3"],  # 3
                       "T8": [3, "T12"],  # 4
                       "Neck": [4, "T8"],  # 5
                       "Head": [5, "Neck"],  # 6
                       "ShoulderR": [4, "T8"],  # 7
                       "UpperArmR": [7, "ShoulderR"],  # 8
                       "LowerArmR": [8, "UpperArmR"],  # 9
                       "HandR": [9, "LowerArmR"],  # 10
                       "ShoulderL": [4, "T8"],  # 11
                       "UpperArmL": [11, "ShoulderR"],  # 12
                       "LowerArmL": [12, "UpperArmR"],  # 13
                       "HandL": [13, "LowerArmR"],  # 14
                       "UpperLegR": [0, "Pelvis"],  # 15
                       "LowerLegR": [15, "UpperLegR"],  # 16
                       "FootR": [16, "LowerLegR"],  # 17
                       "ToesR": [17, "FootR"],  # 18
                       "UpperLegL": [0, "Pelvis"],  # 19
                       "LowerLegL": [19, "UpperLegL"],  # 20
                       "FootL": [20, "LowerLegL"],  # 21
                       "ToesL": [21, "FootL"],  # 22
                       }

    nb_frames = Xsens_orientation_per_move.shape[0]
    Q = np.zeros((23*3, nb_frames))
    rotation_matrices = np.zeros((23, nb_frames, 3, 3))
    for i_segment, key in enumerate(parent_idx_list):
        for i_frame in range(nb_frames):
            Quat_normalized = Xsens_orientation_per_move[i_frame, i_segment*4: (i_segment+1)*4] / np.linalg.norm(
                Xsens_orientation_per_move[i_frame, i_segment*4: (i_segment+1)*4]
            )
            Quat = biorbd.Quaternion(Quat_normalized[0],
                                     Quat_normalized[1],
                                     Quat_normalized[2],
                                     Quat_normalized[3])

            RotMat_current = biorbd.Quaternion.toMatrix(Quat).to_array()
            z_rotation = biorbd.Rotation.fromEulerAngles(np.array([-np.pi/2]), 'z').to_array()
            RotMat_current = z_rotation @ RotMat_current

            if parent_idx_list[key] is None:
                RotMat = np.eye(3)
            else:
                RotMat = rotation_matrices[parent_idx_list[key][0], i_frame, :, :]

            RotMat_between = np.linalg.inv(RotMat) @ RotMat_current
            RotMat_between = biorbd.Rotation(RotMat_between[0, 0], RotMat_between[0, 1], RotMat_between[0, 2],
                            RotMat_between[1, 0], RotMat_between[1, 1], RotMat_between[1, 2],
                            RotMat_between[2, 0], RotMat_between[2, 1], RotMat_between[2, 2])
            Q[i_segment*3:(i_segment+1)*3, i_frame] = biorbd.Rotation.toEulerAngles(RotMat_between, 'xyz').to_array()

            rotation_matrices[i_segment, i_frame, :, :] = RotMat_current
    return Q


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

    biorbd_model_path = "models/SoMe_Xsens_Model_rotated.bioMod"

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
        Xsens_orientation_per_move = data["Xsens_orientation_per_move"]

    # get markers position from the biorbd model
    model = biorbd.Model(biorbd_model_path)
    num_dofs = model.nbQ()

    DoFs = np.zeros((num_dofs, len(Xsens_jointAngle_per_move)))
    DoFs[3:-3, :] = get_q(Xsens_orientation_per_move)
    for i in range(DoFs.shape[0]):
        DoFs[i, :] = np.unwrap(DoFs[i, :])

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
eye_height = 0.0564622475933951
eye_depth = 0.0852029331945667

T8_height = (shoulder_height - hip_height) / 2
C7_height = (body_height - shoulder_height) / 3
Shoulder_height = T8_height
Shoulder_lateral_position = (shoulder_width - elbow_span) / 2
forearm_length = (elbow_span - wrist_span) / 2
hand_length = (wrist_span - arm_span) / 2
upper_leg_lateral_position = -0.5*hip_width
lower_leg_height = -(hip_height-knee_height)
foot_height = -(knee_height-ankle_height)
ball = foot_length * 4/5






