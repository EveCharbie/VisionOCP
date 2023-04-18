
import bioviz
import numpy as np

angle_cone = np.pi / 8
crop_distance = 3
radial_divisions = 10
angular_divisions = 10

gaze_vector = np.array([0, -1, 0])
base_radius = crop_distance * np.tan(angle_cone)

radial_step = base_radius / radial_divisions
angular_step = 2 * np.pi / angular_divisions
r_values = np.arange(0, base_radius + radial_step, radial_step)
theta_values = np.arange(0, 2 * np.pi, angular_step)

for i_r, r in enumerate(r_values):
    for i_th, theta in enumerate(theta_values):
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        print(f'marker cone_approx_{i_r}_{i_th}')
        print('\tparent Eyes')
        print(f'\tposition {x} -3 {y}')
        print('endmarker\n')


biorbd_model_path = "models/SoMe_with_visual_criteria.bioMod"
b = bioviz.Viz(biorbd_model_path)
b.exec()


