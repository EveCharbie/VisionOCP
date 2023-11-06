
"""
This is the main file to run, to replicate the results from this study.
It runs OCP for 42/ and 831< with global weights for the vision objectives.
"""

import TechOpt42 as ocp_module_42
import TechOpt831 as ocp_module_831

vision_weights = [
    # 0.0,
    # 0.25,
    # 0.5,
    # 0.75,
    # 1.0,
    1.25,
    # 1.5,
    # 1.75,
    # 2.0,
]

for weight in vision_weights:
    if weight == 0.0:
        WITH_VISUAL_CRITERIA = False
    else:
        WITH_VISUAL_CRITERIA = True
    # ocp_module_42.main(WITH_VISUAL_CRITERIA=WITH_VISUAL_CRITERIA, visual_weight=weight)
    ocp_module_831.main(WITH_VISUAL_CRITERIA=WITH_VISUAL_CRITERIA, visual_weight=weight)
    print(f"\n\n\n\n I am done with {weight} :) \n\n\n\n")


