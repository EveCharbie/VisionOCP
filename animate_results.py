"""
The goal of this program is to show an animation of the solution of the optimal control problem.
"""

import numpy as np
import pickle
import IPython
import time

import bioviz
from bioptim import (
    OptimalControlProgram,
    DynamicsList,
    DynamicsFcn,
    ObjectiveList,
    ObjectiveFcn,
    BoundsList,
    InitialGuessList,
    InterpolationType,
    OdeSolver,
    Node,
    Solver,
    BiMappingList,
    CostType,
    ConstraintList,
    ConstraintFcn,
    PenaltyNodeList,
    BiorbdModel,
)

from TechOpt831 import prepare_ocp

biorbd_model_path = "models/SoMe.bioMod"

n_shooting = (40, 100, 100, 100, 40)
num_twists = 1
name = "SoMe"
file_name = "SoMe-1-(40_100_100_100_40)-2023-04-17-2319.pkl"

with open("Solutions/" + file_name, "rb") as f:
    data = pickle.load(f)
    sol = data[0]
    qs = data[1]
    qdots = data[2]
    qddots = data[3]
    time_parameters = data[4]
    q_reintegrated = data[5]
    qdot_reintegrated = data[6]
    time_vector = data[7]

sol.ocp = prepare_ocp(biorbd_model_path, n_shooting=n_shooting, num_twists=num_twists, n_threads=7)

biorbd_model_path = "models/SoMe_with_visual_criteria.bioMod"
b = bioviz.Viz(biorbd_model_path)
b.load_movement(qs)
b.exec()

sol.graphs(show_bounds=True)