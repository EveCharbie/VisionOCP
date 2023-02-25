"""
The goal of this program shows as animation of the solution of the optimal control problem.
"""

import numpy as np
import IPython
import time
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
import bioviz


biorbd_model_path = 'models/SoMe.bioMod'
n_shooting = (40, 100, 100, 100, 40)
num_twists = 1
name = "SoMe"
file_name = f"Solutions/{name}-{num_twists}-{str(n_shooting).replace(', ', '_')}-{timestamp}.pkl"

with open(file_name, "rb") as f:
    data = pickle.load(f)
    qs = data['qs']
    qdots = data['qdots']
    time_vector = data['time_vector']
    q_reintegrated = data['q_reintegrated']
    sol = data['sol']

b = bioviz.Viz(biorbd_model_path)
b.load_movement(qs)
b.exec()