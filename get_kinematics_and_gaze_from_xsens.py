"""
The goal of this program is to convert the joint angles from xsens to biorbd.
"""

import numpy as np
import pickle
import os
import biorbd
import bioviz
import scipy

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
move_paths = ["/home/charbie/disk/Eye-tracking/Results_831/SoMe/831</", "/home/charbie/disk/Eye-tracking/Results_831/SoMe/42/"]
save_path = "/home/charbie/Documents/Programmation/VisionOCP/Xsens_recons/"

Generate_videos_FLAG = False
Generate_kinograms_FLAG = True
AngMom_42 = []
AngMom_831 = []
for move_idx, move_path in enumerate(move_paths):
    for filename in os.listdir(move_path):
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
                Xsens_CoM_per_move = data["Xsens_CoM_per_move"]
                time_vector_pupil_per_move = data["time_vector_pupil_per_move"]

            # get markers position from the biorbd model
            model = biorbd.Model(biorbd_model_path)
            num_dofs = model.nbQ()

            eye_angles_without_nans = np.zeros(eye_angles.shape)
            eye_angles_without_nans[:, :] = eye_angles[:, :]
            blink_index = np.isnan(eye_angles[0, :]).astype(int)
            end_of_blinks = np.where(blink_index[1:] - blink_index[:-1] == -1)[0]
            start_of_blinks = np.where(blink_index[1:] - blink_index[:-1] == 1)[0]
            for i in range(len(start_of_blinks)):
                eye_angles_without_nans[:, start_of_blinks[i]+1:end_of_blinks[i]+1] = np.linspace(eye_angles_without_nans[:, start_of_blinks[i]], eye_angles_without_nans[:, end_of_blinks[i]+1], end_of_blinks[i]-start_of_blinks[i]).T

            DoFs = np.zeros((num_dofs, len(Xsens_jointAngle_per_move)))
            DoFs[3:-2, :] = get_q(Xsens_orientation_per_move)
            DoFs[-2:, :] = eye_angles_without_nans
            for i in range(DoFs.shape[0]):
                DoFs[i, :] = np.unwrap(DoFs[i, :])

            time_vector_pupil_per_move = time_vector_pupil_per_move - time_vector_pupil_per_move[0]
            duration = time_vector_pupil_per_move[-1]
            vz_init = 9.81 * duration / 2

            trans = np.zeros((3, len(Xsens_jointAngle_per_move)))
            trans[2, :] = vz_init * time_vector_pupil_per_move - 0.5 * 9.81 * time_vector_pupil_per_move ** 2

            model = biorbd.Model(biorbd_model_path)
            for i in range(DoFs.shape[1]):
                CoM = model.CoM(DoFs[:, i]).to_array()
                trans[:, i] = trans[:, i] - CoM
            DoFs[:3, :] = trans

            # get the twisting angular momentum at take-off
            qdot = (DoFs[:, 1:] - DoFs[:, :-1]) / (time_vector_pupil_per_move[1:] - time_vector_pupil_per_move[:-1])
            AngMom_this_frame = np.zeros((3, 10))
            for i_frame in range(10):
                AngMom_this_frame[:, i_frame] = model.angularMomentum(DoFs[:, i_frame], qdot[:, i_frame]).to_array()
            if move_idx == 0:  # 42/
                AngMom_42.append(np.mean(AngMom_this_frame, axis=1))
            else:  # 831/
                AngMom_831.append(np.mean(AngMom_this_frame, axis=1))

            # real-time video
            if Generate_videos_FLAG or Generate_kinograms_FLAG:
                fps = 60
                n_frames = round(duration * fps)
                time_vector = np.linspace(0, duration, n_frames)
                interpolated_DoFs = np.zeros((num_dofs, n_frames))
                for i in range(DoFs.shape[0]):
                    interp = scipy.interpolate.interp1d(time_vector_pupil_per_move, DoFs[i, :])
                    interpolated_DoFs[i, :] = interp(time_vector)

                with open(save_path + filename[:-25] + "_DoFs.pkl", "wb") as f:
                    pickle.dump({"DoFs": DoFs, "interpolated_DoFs": interpolated_DoFs}, f)

                print(f"Videos/official/{filename[:-25]}.ogv")
                model_type = ["with_cone", "without_cone"]
                biorbd_model_paths = ["models/SoMe_Xsens_Model_rotated.bioMod", "models/SoMe_Xsens_Model_rotated_without_cone.bioMod"]
                if Generate_videos_FLAG:
                    for i in range(2):
                        b = bioviz.Viz(biorbd_model_paths[i],
                                       mesh_opacity=0.8,
                                       show_global_center_of_mass=False,
                                       show_gravity_vector=False,
                                       show_segments_center_of_mass=False,
                                       show_global_ref_frame=False,
                                       show_local_ref_frame=False,
                                       experimental_markers_color=(1, 1, 1),
                                       background_color=(1.0, 1.0, 1.0),
                                       )
                        b.set_camera_zoom(0.25)
                        b.set_camera_focus_point(0, 0, 2.5)
                        b.maximize()
                        b.update()
                        b.load_movement(interpolated_DoFs)

                        b.set_camera_zoom(0.25)
                        b.set_camera_focus_point(0, 0, 2.5)
                        b.maximize()
                        b.update()

                        b.start_recording(f"Videos/official/{filename[:-25]}_{model_type[i]}.ogv")
                        for frame in range(interpolated_DoFs.shape[1] + 1):
                            b.movement_slider[0].setValue(frame)
                            b.add_frame()
                        b.stop_recording()
                        b.quit()

                if Generate_kinograms_FLAG:
                       b = bioviz.Kinogram(model_path=biorbd_model_paths[0],
                                          mesh_opacity=0.8,
                                          show_global_center_of_mass=False,
                                          show_gravity_vector=False,
                                          show_segments_center_of_mass=False,
                                          show_global_ref_frame=False,
                                          show_local_ref_frame=False,
                                          experimental_markers_color=(1, 1, 1),
                                          background_color=(1.0, 1.0, 1.0),
                                           )
                       b.load_movement(interpolated_DoFs)
                       b.set_camera_focus_point(0, 0, 2.5)
                       b.set_camera_zoom(0.25)
                       b.exec(frame_step=10,
                              save_path=f"Kinograms/{filename}.svg")


AngMom_angle_42 = []
for i in range(len(AngMom_42)):
    atan = np.arctan2(AngMom_42[i][2], AngMom_42[i][0]) * 180 / np.pi
    if atan > 90:
        atan = 180 - atan
    AngMom_angle_42 += [atan]
print("42 Angular momentum tilt angle [deg]: ", np.mean(np.array(AngMom_angle_42)))
print("angular momentum 42/ : ", AngMom_42)

AngMom_angle_831 = []
for i in range(len(AngMom_831)):
    atan = np.arctan2(AngMom_831[i][2], AngMom_831[i][0]) * 180 / np.pi
    if atan > 90:
        atan = 180 - atan
    AngMom_angle_831 += [atan]
print("831< Angular momentum tilt angle [deg]: ", np.mean(np.array(AngMom_angle_831)))
print("angular momentum 831 : ", AngMom_831)