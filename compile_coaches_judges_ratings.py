import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


colors = ['#103778', '#0593A2', '#FF7A48', '#E3371E', '#E5003F']
judges_colors = [cm.magma(0.3), cm.magma(0.6)]

### ------------- Choaches ratings -------------- ###

ratings_coaches = {
    "Stephen": {
        "SoMe_42_with_visual_criteria_without_mesh-(100_40)-1p0_CVG": [2, 3, 2, 2, 4, 2],  # A
        "a62d4691_0_0-45_796__42__1": [5, 5, 5, 5, 5, 5],  # B
        "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-1p0_CVG": [2, 2, 2, 2, 4, 2],  # C
        "a62d4691_0_0-45_796__42__0": [4, 4, 4, 2, 3, 2],  # D
        "87972c15_0_0-105_114__831__1": [2, 2, 3, 1, 3, 2],  # E
        "SoMe_42_with_visual_criteria_without_mesh-(100_40)-0p25_CVG": [5, 5, 5, 5, 5, 5],  # F
        "SoMe_42_with_visual_criteria_without_mesh-(100_40)-2p0_CVG": [2, 3, 3, 2, 3, 2],  # G
        "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-1p75_CVG": [4, 4, 4, 2, 4, 4],  # H
        "SoMe_42_without_mesh-(100_40)-0p0_CVG": [4, 4, 4, 4, 4, 4],  # I
        "a62d4691_0_0-45_796__42__3": [2, 4, 4, 2, 4, 2],  # J
        "SoMe_42_with_visual_criteria_without_mesh-(100_40)-1p75_CVG": [4, 2, 3, 2, 4, 2],  # K
        "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-2p0_CVG": [4, 4, 4, 2, 4, 3],  # L
        "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-0p5_CVG": [4, 4, 4, 4, 4, 4],  # M
        "87972c15_0_0-105_114__831__0": [2, 3, 3, 1, 3, 1],  # N
        "SoMe_without_mesh_831-(40_40_40_40_40_40)-2023-0p0_CVG": [5, 5, 5, 4, 5, 5],  # O
        "87972c15_0_0-105_114__831__2": [4, 4, 4, 2, 4, 2],  # P
        "a62d4691_0_0-45_796__42__2": [3, 4, 4, 2, 4, 2],  # Q
        "c2a19850_0_0-113_083__831__0": [2, 3, 3, 2, 3, 1],  # R
        "SoMe_42_with_visual_criteria_without_mesh-(100_40)-0p5_CVG": [3, 4, 4, 2, 4, 2],  # S
        "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-0p25_CVG": [4, 4, 4, 4, 4, 4],  # T
        "caccfb24_0_0-48_286__42__1": [4, 4, 4, 4, 4, 4],  # U
    },
    "Antoine": {
        "a62d4691_0_0-45_796__42__0": [5, 5, 5, 4, 5, 5],  # A
        "SoMe_42_with_visual_criteria_without_mesh-(100_40)-1p75_CVG": [5, 5, 4, 2, 5, 2],  # B
        "a62d4691_0_0-45_796__42__3": [5, 5, 5, 5, 5, 5],  # C
        "SoMe_42_with_visual_criteria_without_mesh-(100_40)-1p0_CVG": [4, 3, 2, 1, 5, 2],  # D
        "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-0p5_CVG": [5, 2, 1, 1, 4, 1],  # E
        "87972c15_0_0-105_114__831__2": [4, 5, 5, 3, 5, 4],  # F
        "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-0p25_CVG": [5, 3, 2, 1, 5, 1],  # G
        "caccfb24_0_0-48_286__42__1": [5, 5, 5, 5, 5, 5],  # H
        "a62d4691_0_0-45_796__42__1": [5, 5, 5, 5, 4, 5],  # I
        "87972c15_0_0-105_114__831__1": [1, 4, 4, 1, 5, 2],  # J
        "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-2p0_CVG": [5, 4, 2, 1, 4, 2],  # K
        "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-1p75_CVG": [5, 3, 1, 1, 4, 1],  # L
        "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-1p0_CVG": [5, 3, 2, 1, 4, 1],  # M
        "87972c15_0_0-105_114__831__0": [1, 5, 5, 1, 5, 1],  # N
        "a62d4691_0_0-45_796__42__2": [4, 5, 5, 5, 5, 4],  # O
        "SoMe_without_mesh_831-(40_40_40_40_40_40)-2023-0p0_CVG": [5, 5, 2, 1, 4, 1],  # P
        "SoMe_42_with_visual_criteria_without_mesh-(100_40)-2p0_CVG": [3, 4, 2, 1, 5, 1],  # Q
        "SoMe_42_with_visual_criteria_without_mesh-(100_40)-0p25_CVG": [4, 4, 2, 1, 5, 1],  # R
        "SoMe_42_with_visual_criteria_without_mesh-(100_40)-0p5_CVG": [4, 4, 3, 2, 5, 2],  # S
        "c2a19850_0_0-113_083__831__0": [2, 4, 4, 1, 5, 1],  # T
        "SoMe_42_without_mesh-(100_40)-0p0_CVG": [5, 4, 3, 2, 5, 2],  # U
    },
    "Marty": {
        "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-2p0_CVG": [4, 2, 2, 2, 4, 2],  # A
        "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-1p0_CVG": [4, 2, 2, 2, 4, 2],  # B
        "a62d4691_0_0-45_796__42__1": [4, 4, 4, 4, 4, 4],  # C
        "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-0p5_CVG": [4, 2, 2, 2, 4, 2],  # D
        "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-1p75_CVG": [4, 2, 2, 2, 4, 2],  # E
        "87972c15_0_0-105_114__831__1": [4, 4, 4, 2, 4, 3],  # F
        "SoMe_42_with_visual_criteria_without_mesh-(100_40)-1p75_CVG": [3, 3, 4, 2, 4, 2],  # G
        "SoMe_42_without_mesh-(100_40)-0p0_CVG": [3, 2, 4, 4, 2, 2],  # H
        "SoMe_42_with_visual_criteria_without_mesh-(100_40)-2p0_CVG": [3, 3, 4, 2, 4, 2],  # I
        "caccfb24_0_0-48_286__42__1": [4, 4, 4, 4, 4, 4],  # J
        "SoMe_42_with_visual_criteria_without_mesh-(100_40)-0p25_CVG": [3, 4, 3, 4, 4, 3],  # K
        "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-0p25_CVG": [4, 2, 2, 2, 4, 2],  # L
        "a62d4691_0_0-45_796__42__0": [4, 4, 4, 4, 4, 4],  # M
        "c2a19850_0_0-113_083__831__0": [4, 4, 4, 2, 4, 3],  # N
        "SoMe_42_with_visual_criteria_without_mesh-(100_40)-1p0_CVG": [3, 3, 4, 2, 4, 2],  # O
        "SoMe_without_mesh_831-(40_40_40_40_40_40)-2023-0p0_CVG": [3, 2, 2, 2, 2.5, 2],  # P
        "a62d4691_0_0-45_796__42__3": [4, 4, 4, 4, 4, 4],  # Q
        "a62d4691_0_0-45_796__42__2": [4, 4, 4, 4, 4, 4],  # R
        "87972c15_0_0-105_114__831__2": [4, 4, 4, 2, 4, 2],  # S
        "87972c15_0_0-105_114__831__0": [4, 4, 4, 2, 4, 2],  # T
        "SoMe_42_with_visual_criteria_without_mesh-(100_40)-0p5_CVG": [4, 4, 4, 4, 4, 4],  # U
    },
    "Karina": {
        "SoMe_42_without_mesh-(100_40)-0p0_CVG": [5, 5, 5, 5, 5, 5],  # A
        "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-2p0_CVG": [5, 4, 4, 2, 4, 2],  # B
        "a62d4691_0_0-45_796__42__0": [5, 5, 5, 4, 5, 4],  # C
        "SoMe_42_with_visual_criteria_without_mesh-(100_40)-1p0_CVG": [2, 3, 3, 2, 3, 2],  # D
        "SoMe_42_with_visual_criteria_without_mesh-(100_40)-1p75_CVG": [3, 2, 3, 3, 3, 2],  # E
        "87972c15_0_0-105_114__831__1": [2, 4, 4, 3, 3, 3],  # F
        "a62d4691_0_0-45_796__42__2": [4, 4, 4, 4, 4, 3],  # G
        "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-0p25_CVG": [4, 3, 4, 2, 4, 2],  # H
        "87972c15_0_0-105_114__831__0": [2, 3, 3, 2, 2, 2],  # I
        "87972c15_0_0-105_114__831__2": [3, 4, 4, 3, 4, 3],  # J
        "SoMe_42_with_visual_criteria_without_mesh-(100_40)-2p0_CVG": [4, 2, 3, 2, 2, 1],  # K
        "a62d4691_0_0-45_796__42__1": [4, 4, 4, 4, 4, 4],  # L
        "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-1p75_CVG": [4, 3, 3, 2, 4, 3],  # M
        "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-0p5_CVG": [3, 2, 2, 2, 3, 1],  # N
        "SoMe_without_mesh_831-(40_40_40_40_40_40)-2023-0p0_CVG": [3, 2, 3, 1, 3, 1],  # O
        "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-1p0_CVG": [2, 2, 2, 2, 3, 2],  # P
        "SoMe_42_with_visual_criteria_without_mesh-(100_40)-0p5_CVG": [2, 2, 2, 3, 2, 1],  # Q
        "a62d4691_0_0-45_796__42__3": [3, 4, 4, 3, 4, 3],  # R
        "SoMe_42_with_visual_criteria_without_mesh-(100_40)-0p25_CVG": [4, 4, 4, 4, 3, 2],  # S
        "c2a19850_0_0-113_083__831__0": [2, 3, 3, 2, 3, 2],  # T
        "caccfb24_0_0-48_286__42__1": [3, 4, 4, 4, 4, 4],  # U
    }
}

names_42_real = ["a62d4691_0_0-45_796__42__1",
                 "a62d4691_0_0-45_796__42__0",
                 "a62d4691_0_0-45_796__42__3",
                 "a62d4691_0_0-45_796__42__2",
                 "caccfb24_0_0-48_286__42__1",
                 ]
names_831_real = ["87972c15_0_0-105_114__831__1",
                  "87972c15_0_0-105_114__831__0",
                  "87972c15_0_0-105_114__831__2",
                  "c2a19850_0_0-113_083__831__0",
                  ]
names_42_simulations = ["SoMe_42_without_mesh-(100_40)-0p0_CVG",
                        "SoMe_42_with_visual_criteria_without_mesh-(100_40)-0p25_CVG",
                        "SoMe_42_with_visual_criteria_without_mesh-(100_40)-0p5_CVG",
                        "SoMe_42_with_visual_criteria_without_mesh-(100_40)-1p0_CVG",
                        "SoMe_42_with_visual_criteria_without_mesh-(100_40)-1p75_CVG",
                        "SoMe_42_with_visual_criteria_without_mesh-(100_40)-2p0_CVG",
                        ]
names_831_simulations = ["SoMe_without_mesh_831-(40_40_40_40_40_40)-2023-0p0_CVG",
                         "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-0p25_CVG",
                         "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-0p5_CVG",
                         "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-1p0_CVG",
                         "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-1p75_CVG",
                         "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-2p0_CVG",
                         ]

# Set up the bar width and positions
conditions_simulations = ['0.0', '0.25', '0.5', '1.0', '1.75', '2.0']
conditions_reel = ["", "", "", "", "", ""]

criteria = ["This techniques \nis efficient \nfor aerial \ntwist creation",
            "This technique \nis safe for \nan athlete \nto try",
            "Overall, \nthis technique \nseems realistic",
            "This technique \nis aesthetic",
            "This technique \nallows the athlete \nto get appropriate \nvisual information",
            "I would recommend \nmy athletes \nto use this \ntechnique"]


### ------------- Judges ratings -------------- ###

ratings_judges = {"Stephan": {"caccfb24_0_0-48_286__42__1": [-0.1],  # A
                       "a62d4691_0_0-45_796__42__0": [-0.2],  # B
                       "a62d4691_0_0-45_796__42__1": [-0.2],  # C
                       "SoMe_without_mesh_831-(40_40_40_40_40_40)-2023-0p0_CVG": [-0.1],  # D
                       "c2a19850_0_0-113_083__831__0": [-0.4],  # E
                       "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-2p0_CVG": [-0.1],  # F
                       "SoMe_42_with_visual_criteria_without_mesh-(100_40)-0p5_CVG": [-0.1],  # G
                       "87972c15_0_0-105_114__831__1": [-0.4],  # H
                       "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-1p0_CVG": [-0.1],  # I
                       "a62d4691_0_0-45_796__42__3": [-0.1],  # J
                       "SoMe_42_with_visual_criteria_without_mesh-(100_40)-1p75_CVG": [-0.2],  # K
                       "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-0p5_CVG": [-0.1],  # L
                       "a62d4691_0_0-45_796__42__2": [-0.2],  # M
                       "87972c15_0_0-105_114__831__2": [-0.4],  # N
                       "SoMe_42_with_visual_criteria_without_mesh-(100_40)-2p0_CVG": [-0.2],  # O
                       "SoMe_42_with_visual_criteria_without_mesh-(100_40)-1p0_CVG": [-0.1],  # P
                       "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-1p75_CVG": [-0.2],  # Q
                       "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-0p25_CVG": [-0.1],  # R
                       "SoMe_42_with_visual_criteria_without_mesh-(100_40)-0p25_CVG": [-0.1],  # S
                       "87972c15_0_0-105_114__831__0": [-0.4],  # T
                       "SoMe_42_without_mesh-(100_40)-0p0_CVG": [-0.3],  # U
                       },
            "Julie": {"SoMe_42_with_visual_criteria_without_mesh-(100_40)-0p25_CVG": [-0.1],  # A
                      "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-1p75_CVG": [-0.1],  # B
                      "87972c15_0_0-105_114__831__1": [-0.4],  # C
                      "SoMe_42_without_mesh-(100_40)-0p0_CVG": [-0.1],  # D
                      "87972c15_0_0-105_114__831__2": [-0.4],  # E
                      "a62d4691_0_0-45_796__42__2": [-0.2],  # F
                      "c2a19850_0_0-113_083__831__0": [-0.5],  # G
                      "SoMe_42_with_visual_criteria_without_mesh-(100_40)-2p0_CVG": [-0.1],  # H
                      "a62d4691_0_0-45_796__42__1": [-0.3],  # I
                      "a62d4691_0_0-45_796__42__0": [-0.3],  # J
                      "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-0p5_CVG": [-0.1],  # K
                      "SoMe_42_with_visual_criteria_without_mesh-(100_40)-1p0_CVG": [-0.1],  # L
                      "SoMe_without_mesh_831-(40_40_40_40_40_40)-2023-0p0_CVG": [-0.2],  # M
                      "SoMe_42_with_visual_criteria_without_mesh-(100_40)-0p5_CVG": [-0.1],  # N
                      "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-2p0_CVG": [-0.2],  # O
                      "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-0p25_CVG": [-0.1],  # P
                      "87972c15_0_0-105_114__831__0": [-0.5],  # Q
                      "caccfb24_0_0-48_286__42__1": [-0.3],  # R
                      "a62d4691_0_0-45_796__42__3": [-0.3],  # S
                      "SoMe_42_with_visual_criteria_without_mesh-(100_40)-1p75_CVG": [-0.1],  # T
                      "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-1p0_CVG": [-0.2],  # U
                },
           }

detailed_ratings_judges = {"Stephan": {"caccfb24_0_0-48_286__42__1": {"arm": -0.1, "legs": 0, "body": 0, "kickout": -0.1},  # A
                                      "a62d4691_0_0-45_796__42__0": {"arm": 0, "legs": 0, "body": 0, "kickout": -0.2},  # B
                                      "a62d4691_0_0-45_796__42__1": {"arm": -0.1, "legs": 0, "body": 0, "kickout": -0.1},  # C
                                      "SoMe_without_mesh_831-(40_40_40_40_40_40)-2023-0p0_CVG": {"arm": -0.1, "legs": 0, "body": 0, "kickout": -0.1},  # D
                                      "c2a19850_0_0-113_083__831__0": {"arm": 0, "legs": 0, "body": -0.2, "kickout": -0.2},  # E
                                      "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-2p0_CVG": {"arm": -0.1, "legs": -0.1, "body": 0, "kickout": 0},  # F
                                      "SoMe_42_with_visual_criteria_without_mesh-(100_40)-0p5_CVG": {"arm": -0.1, "legs": -0.1, "body": 0, "kickout": 0},  # G
                                      "87972c15_0_0-105_114__831__1": {"arm": 0, "legs": -0.1, "body": -0.2, "kickout": -0.2}, # H
                                      "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-1p0_CVG": {"arm": -0.1, "legs": -0.1, "body": 0, "kickout": 0},  # I
                                      "a62d4691_0_0-45_796__42__3": {"arm": 0, "legs": 0, "body": 0, "kickout": -0.1},  # J
                                      "SoMe_42_with_visual_criteria_without_mesh-(100_40)-1p75_CVG": {"arm": -0.1, "legs": -0.1, "body": 0, "kickout": 0},  # K
                                      "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-0p5_CVG": {"arm": -0.1, "legs": -0.1, "body": 0, "kickout": 0},  # L
                                      "a62d4691_0_0-45_796__42__2": {"arm": 0, "legs": 0, "body": 0, "kickout": -0.2},  # M
                                      "87972c15_0_0-105_114__831__2": {"arm": 0, "legs": 0, "body": -0.2, "kickout": -0.2},  # N
                                      "SoMe_42_with_visual_criteria_without_mesh-(100_40)-2p0_CVG": {"arm": -0.1, "legs": -0.1, "body": 0, "kickout": 0},  # O
                                      "SoMe_42_with_visual_criteria_without_mesh-(100_40)-1p0_CVG": {"arm": -0.1, "legs": 0, "body": 0, "kickout": 0},  # P
                                      "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-1p75_CVG": {"arm": -0.1, "legs": -0.1, "body": 0, "kickout": 0},  # Q
                                      "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-0p25_CVG": {"arm": -0.1, "legs": -0.1, "body": 0, "kickout": 0},  # R
                                      "SoMe_42_with_visual_criteria_without_mesh-(100_40)-0p25_CVG": {"arm": -0.1, "legs": -0.1, "body": 0, "kickout": 0},  # S
                                      "87972c15_0_0-105_114__831__0": {"arm": -0.1, "legs": 0, "body": -0.2, "kickout": -0.1},  # T
                                      "SoMe_42_without_mesh-(100_40)-0p0_CVG": {"arm": -0.1, "legs": -0.1, "body": 0, "kickout": 0},  # U
                       },
            "Julie": {"SoMe_42_with_visual_criteria_without_mesh-(100_40)-0p25_CVG": {"arm": 0, "legs": -0.1, "body": -0.1, "kickout": 0},  # A
                      "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-1p75_CVG": {"arm": 0, "legs": -0.1, "body": 0, "kickout": 0},  # B
                      "87972c15_0_0-105_114__831__1": {"arm": -0.1, "legs": -0.2, "body": -0.3, "kickout": -0.2},  # C
                      "SoMe_42_without_mesh-(100_40)-0p0_CVG": {"arm": 0, "legs": -0.1, "body": 0, "kickout": 0},  # D
                      "87972c15_0_0-105_114__831__2": {"arm": -0.1, "legs": -0.2, "body": -0.2, "kickout": -0.2},  # E
                      "a62d4691_0_0-45_796__42__2": {"arm": 0, "legs": -0.1, "body": -0.1, "kickout": -0.1},  # F
                      "c2a19850_0_0-113_083__831__0": {"arm": -0.1, "legs": -0.2, "body": -0.2, "kickout": -0.1},  # G
                      "SoMe_42_with_visual_criteria_without_mesh-(100_40)-2p0_CVG": {"arm": -0.1, "legs": -0.1, "body": -0.1, "kickout": -0.1},  # H
                      "a62d4691_0_0-45_796__42__1": {"arm": 0, "legs": -0.2, "body": -0.2, "kickout": -0.1},  # I
                      "a62d4691_0_0-45_796__42__0": {"arm": -0.1, "legs": -0.2, "body": -0.1, "kickout": -0.1},  # J
                      "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-0p5_CVG": {"arm": -0.1, "legs": -0.1, "body": -0.1, "kickout": 0},  # K
                      "SoMe_42_with_visual_criteria_without_mesh-(100_40)-1p0_CVG": {"arm": -0.1, "legs": -0.1, "body": 0, "kickout": 0},  # L
                      "SoMe_without_mesh_831-(40_40_40_40_40_40)-2023-0p0_CVG": {"arm": 0, "legs": -0.1, "body": 0, "kickout": 0},  # M
                      "SoMe_42_with_visual_criteria_without_mesh-(100_40)-0p5_CVG": {"arm": -0.1, "legs": -0.1, "body": -0.1, "kickout": 0},  # N
                      "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-2p0_CVG": {"arm": -0.1, "legs": -0.1, "body": 0, "kickout": -0.1},  # O
                      "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-0p25_CVG": {"arm": -0.1, "legs": -0.1, "body": -0.1, "kickout": 0},  # P
                      "87972c15_0_0-105_114__831__0": {"arm": -0.1, "legs": -0.2, "body": -0.2, "kickout": -0.2},  # Q
                      "caccfb24_0_0-48_286__42__1": {"arm": -0.1, "legs": -0.2, "body": -0.2, "kickout": -0.1},  # R
                      "a62d4691_0_0-45_796__42__3": {"arm": -0.1, "legs": -0.2, "body": -0.1, "kickout": -0.1},  # S
                      "SoMe_42_with_visual_criteria_without_mesh-(100_40)-1p75_CVG": {"arm": -0.1, "legs": -0.1, "body": 0, "kickout": 0},  # T
                      "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-1p0_CVG": {"arm": -0.1, "legs": -0.1, "body": 0, "kickout": 0},  # U
                },
           }

fig, axs = plt.subplots(7, 2, figsize=(8, 15))
bar_width = 0.15
x = [i * bar_width for i in range(6)]

# Create a bar plot for simulated 42/
total_per_criteria = np.zeros((len(names_42_simulations), len(criteria)))
for i_trial, trial in enumerate(names_42_simulations):
    for i_criteria in range(len(criteria)):
        total = 0
        for i_name, name in enumerate(ratings_coaches):
            axs[i_criteria, 0].bar(i_trial * bar_width, ratings_coaches[name][trial][i_criteria],
                    width=bar_width,
                    bottom=total,
                    color=colors[i_name],
                    alpha=0.5)
            total += ratings_coaches[name][trial][i_criteria]
        total_per_criteria[i_trial, i_criteria] = total
    total = 0
    for i_name, name in enumerate(ratings_judges):
        axs[6, 0].bar(i_trial * bar_width, ratings_judges[name][trial][0],
                               width=bar_width,
                               bottom=total,
                               color=judges_colors[i_name],
                               alpha=0.5)
        total += ratings_judges[name][trial][0]
print("42 total scores: ", np.mean(total_per_criteria, axis=0))

# Create a reference bar for real 42/
total_per_criteria = [[] for i in range(7)]
for i_trial, trial in enumerate(names_42_real):
    for i_criteria in range(len(criteria)):
        total = 0
        for i_name, name in enumerate(ratings_coaches):
            total += ratings_coaches[name][trial][i_criteria]
        total_per_criteria[i_criteria] += [total]
    total = 0
    for i_name, name in enumerate(ratings_judges):
        total += ratings_judges[name][trial][0]
    total_per_criteria[6] += [total]
mean_total = np.mean(np.asarray(total_per_criteria), axis=1)
min_total = np.min(np.asarray(total_per_criteria), axis=1)
max_total = np.max(np.asarray(total_per_criteria), axis=1)

for i_criteria in range(7):
    axs[i_criteria, 0].plot(7*bar_width-bar_width/2, mean_total[i_criteria], marker="s", color="k")
    axs[i_criteria, 0].plot(np.array([-bar_width/2, 7*bar_width]), np.array([mean_total[i_criteria], mean_total[i_criteria]]), color="k")
    axs[i_criteria, 0].plot(np.array([7*bar_width-bar_width/2, 7*bar_width-bar_width/2]), np.array([min_total[i_criteria], max_total[i_criteria]]), color="k", alpha=0.5)
    axs[i_criteria, 0].plot(np.array([7*bar_width-bar_width/2-0.05, 7*bar_width-bar_width/2+0.05]), np.array([min_total[i_criteria], min_total[i_criteria]]), color="k", alpha=0.5)
    axs[i_criteria, 0].plot(np.array([7*bar_width-bar_width/2-0.05, 7*bar_width-bar_width/2+0.05]), np.array([max_total[i_criteria], max_total[i_criteria]]), color="k", alpha=0.5)

# Create a bar plot for simulated 831<
total_per_criteria = np.zeros((len(names_831_simulations), len(criteria)))
for i_trial, trial in enumerate(names_831_simulations):
    for i_criteria in range(len(criteria)):
        total = 0
        for i_name, name in enumerate(ratings_coaches):
            axs[i_criteria, 1].bar(i_trial * bar_width, ratings_coaches[name][trial][i_criteria],
                    width=bar_width,
                    bottom=total,
                    color=colors[i_name],
                    alpha=0.5)
            total += ratings_coaches[name][trial][i_criteria]
        total_per_criteria[i_trial, i_criteria] = total
    total = 0
    for i_name, name in enumerate(ratings_judges):
        axs[6, 1].bar(i_trial * bar_width, ratings_judges[name][trial][0],
                               width=bar_width,
                               bottom=total,
                               color=judges_colors[i_name],
                               alpha=0.5)
        total += ratings_judges[name][trial][0]
print("831 total scores: ", np.mean(total_per_criteria, axis=0))

# Create a reference bar for real 831<
total_per_criteria = [[] for i in range(7)]
for i_trial, trial in enumerate(names_831_real):
    for i_criteria in range(len(criteria)):
        total = 0
        for i_name, name in enumerate(ratings_coaches):
            total += ratings_coaches[name][trial][i_criteria]
        total_per_criteria[i_criteria] += [total]
    total = 0
    for i_name, name in enumerate(ratings_judges):
        total += ratings_judges[name][trial][0]
    total_per_criteria[6] += [total]
mean_total = np.mean(np.asarray(total_per_criteria), axis=1)
min_total = np.min(np.asarray(total_per_criteria), axis=1)
max_total = np.max(np.asarray(total_per_criteria), axis=1)

for i_criteria in range(7):
    axs[i_criteria, 1].plot(7*bar_width-bar_width/2, mean_total[i_criteria], marker="s", color="k")
    axs[i_criteria, 1].plot(np.array([-bar_width/2, 7*bar_width]), np.array([mean_total[i_criteria], mean_total[i_criteria]]), color="k")
    axs[i_criteria, 1].plot(np.array([7*bar_width-bar_width/2, 7*bar_width-bar_width/2]), np.array([min_total[i_criteria], max_total[i_criteria]]), color="k", alpha=0.5)
    axs[i_criteria, 1].plot(np.array([7*bar_width-bar_width/2-0.05, 7*bar_width-bar_width/2+0.05]), np.array([min_total[i_criteria], min_total[i_criteria]]), color="k", alpha=0.5)
    axs[i_criteria, 1].plot(np.array([7*bar_width-bar_width/2-0.05, 7*bar_width-bar_width/2+0.05]), np.array([max_total[i_criteria], max_total[i_criteria]]), color="k", alpha=0.5)

axs[0, 0].set_title("42/", fontsize=16)
axs[0, 1].set_title("831<", fontsize=16)
for i_criteria in range(7):
    axs[i_criteria, 0].set_xticks(x)
    axs[i_criteria, 0].set_xticklabels(conditions_simulations)
    axs[i_criteria, 1].set_xticks(x)
    axs[i_criteria, 1].set_xticklabels(conditions_simulations)
    if i_criteria == 6:
        axs[i_criteria, 0].set_ylabel("Judges deductions", fontsize=12, labelpad=60)
        axs[i_criteria, 0].yaxis.label.set(rotation='horizontal', ha='center', va='center')
    else:
        axs[i_criteria, 0].set_ylabel(criteria[i_criteria], fontsize=12, labelpad=60)
        axs[i_criteria, 0].yaxis.label.set(rotation='horizontal', ha='center', va='center')

    if i_criteria == 6:
        axs[i_criteria, 0].set_ylim(-1, 0)
        axs[i_criteria, 1].set_ylim(-1, 0)
    else:
        axs[i_criteria, 0].set_ylim(0, 20)
        axs[i_criteria, 1].set_ylim(0, 20)
        axs[i_criteria, 0].set_xlabel("Global visual weight")
        axs[i_criteria, 1].set_xlabel("Global visual weight")

plt.tight_layout()
plt.savefig('Graphs/coaches_ratings.png', dpi=300)
plt.show()


# Simulated 42/ detailed ratings
criteria_colors = cm.get_cmap('magma')
spacing = 0.6
fig_detailed, axs_detailed = plt.subplots(2, 1, figsize=(6, 8))
for i_trial, trial in enumerate(names_42_simulations):
    for i_criteria, criteria in enumerate(detailed_ratings_judges["Julie"][trial]):
        total = 0
        for i_name, name in enumerate(detailed_ratings_judges):
                axs_detailed[0].bar(i_trial * spacing + i_criteria * bar_width, detailed_ratings_judges[name][trial][criteria],
                        width=bar_width*0.8,
                        bottom=total,
                        color=criteria_colors(i_criteria/4))
                total += detailed_ratings_judges[name][trial][criteria]


# Real 42/ detailed ratings
total_per_criteria = [[] for i in range(4)]
for i_trial, trial in enumerate(names_42_real):
    for i_criteria, criteria in enumerate(detailed_ratings_judges["Julie"][trial]):
        total = 0
        for i_name, name in enumerate(detailed_ratings_judges):
            total += detailed_ratings_judges[name][trial][criteria]
        total_per_criteria[i_criteria] += [total]
mean_total = np.mean(np.asarray(total_per_criteria), axis=1)

for i_criteria in range(4):
    axs_detailed[0].bar((len(names_42_simulations)+1)* spacing + i_criteria * bar_width, mean_total[i_criteria],
                        width=bar_width*0.8, color=criteria_colors(i_criteria/4), alpha=0.6)


# Simulated 831< detailed ratings
for i_trial, trial in enumerate(names_831_simulations):
    for i_criteria, criteria in enumerate(detailed_ratings_judges["Julie"][trial]):
        total = 0
        for i_name, name in enumerate(detailed_ratings_judges):
                axs_detailed[1].bar(i_trial * spacing + i_criteria * bar_width, detailed_ratings_judges[name][trial][criteria],
                        width=bar_width*0.8,
                        bottom=total,
                        color=criteria_colors(i_criteria/4))
                total += detailed_ratings_judges[name][trial][criteria]


# Real 831< detailed ratings
total_per_criteria = [[] for i in range(4)]
for i_trial, trial in enumerate(names_831_real):
    for i_criteria, criteria in enumerate(detailed_ratings_judges["Julie"][trial]):
        total = 0
        for i_name, name in enumerate(detailed_ratings_judges):
            total += detailed_ratings_judges[name][trial][criteria]
        total_per_criteria[i_criteria] += [total]
mean_total = np.mean(np.asarray(total_per_criteria), axis=1)

for i_criteria in range(4):
    axs_detailed[1].bar((len(names_831_simulations)+1) * spacing + i_criteria * bar_width, mean_total[i_criteria],
                        width=bar_width*0.8, color=criteria_colors(i_criteria/4), alpha=0.6)

for i_criteria, criteria in enumerate(["Arms", "Legs", "Body", "Kick-out"]):
    axs_detailed[0].bar(0, 0, width=0, color=criteria_colors(i_criteria/4), label=criteria)

axs_detailed[0].set_yticks([0.0, -0.1, -0.2])
axs_detailed[0].set_yticklabels(["0.0", "-0.1", "-0.2"])
axs_detailed[1].set_yticks([0.0, -0.1, -0.2, -0.3, -0.4])
axs_detailed[1].set_yticklabels(["0.0", "-0.1", "-0.2", "-0.3", "-0.4"])
axs_detailed[0].legend(bbox_to_anchor=(1.025, 0.5), loc='center left', frameon=False)
fig_detailed.subplots_adjust(right=0.8)

plt.savefig('Graphs/judges_detailed_ratings.png', dpi=300)
plt.show()