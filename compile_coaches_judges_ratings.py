import numpy as np
import matplotlib.pyplot as plt


colors = ['#103778', '#0593A2', '#FF7A48', '#E3371E', '#E5003F']  # '#151F31'

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
conditions_simulations = ['without vision', '0.25', '0.5', '1.0', '1.75', '2.0']
conditions_reel = ["", "", "", "", "", ""]

criteria = ["This techniques \nis efficient \nfor aerial \ntwist creation",
            "This technique \nis safe for \nan athlete \nto try",
            "Overall, \nthis technique \nseems realistic",
            "This technique \nis aesthetic",
            "This technique \nallow the athlete \nto get appropriate \nvisual information",
            "I would recommend \nmy athletes \nto use this \ntechnique"]


### ------------- Judges ratings -------------- ###

ratings_judges = {"Stephan": {"caccfb24_0_0-48_286__42__1": [0.1],  # A
                       "a62d4691_0_0-45_796__42__0": [0.2],  # B
                       "a62d4691_0_0-45_796__42__1": [0.2],  # C
                       "SoMe_without_mesh_831-(40_40_40_40_40_40)-2023-0p0_CVG": [0.1],  # D
                       "c2a19850_0_0-113_083__831__0": [0.4],  # E
                       "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-2p0_CVG": [0.1],  # F
                       "SoMe_42_with_visual_criteria_without_mesh-(100_40)-0p5_CVG": [0.1],  # G
                       "87972c15_0_0-105_114__831__1": [0.4],  # H
                       "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-1p0_CVG": [0.1],  # I
                       "a62d4691_0_0-45_796__42__3": [0.1],  # J
                       "SoMe_42_with_visual_criteria_without_mesh-(100_40)-1p75_CVG": [0.2],  # K
                       "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-0p5_CVG": [0.1],  # L
                       "a62d4691_0_0-45_796__42__2": [0.2],  # M
                       "87972c15_0_0-105_114__831__2": [0.4],  # N
                       "SoMe_42_with_visual_criteria_without_mesh-(100_40)-2p0_CVG": [0.2],  # O
                       "SoMe_42_with_visual_criteria_without_mesh-(100_40)-1p0_CVG": [0.1],  # P
                       "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-1p75_CVG": [0.2],  # Q
                       "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-0p25_CVG": [0.1],  # R
                       "SoMe_42_with_visual_criteria_without_mesh-(100_40)-0p25_CVG": [0.1],  # S
                       "87972c15_0_0-105_114__831__0": [0.4],  # T
                       "SoMe_42_without_mesh-(100_40)-0p0_CVG": [0.3],  # U
                       },
            "Julie": {"SoMe_42_with_visual_criteria_without_mesh-(100_40)-0p25_CVG": [0.1],  # A
                      "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-1p75_CVG": [0.1],  # B
                      "87972c15_0_0-105_114__831__1": [0.4],  # C
                      "SoMe_42_without_mesh-(100_40)-0p0_CVG": [0.1],  # D
                      "87972c15_0_0-105_114__831__2": [0.4],  # E
                      "a62d4691_0_0-45_796__42__2": [0.2],  # F
                      "c2a19850_0_0-113_083__831__0": [0.5],  # G
                      "SoMe_42_with_visual_criteria_without_mesh-(100_40)-2p0_CVG": [0.1],  # H
                      "a62d4691_0_0-45_796__42__1": [0.3],  # I
                      "a62d4691_0_0-45_796__42__0": [0.3],  # J
                      "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-0p5_CVG": [0.1],  # K
                      "SoMe_42_with_visual_criteria_without_mesh-(100_40)-1p0_CVG": [0.1],  # L
                      "SoMe_without_mesh_831-(40_40_40_40_40_40)-2023-0p0_CVG": [0.2],  # M
                      "SoMe_42_with_visual_criteria_without_mesh-(100_40)-0p5_CVG": [0.1],  # N
                      "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-2p0_CVG": [0.2],  # O
                      "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-0p25_CVG": [0.1],  # P
                      "87972c15_0_0-105_114__831__0": [0.5],  # Q
                      "caccfb24_0_0-48_286__42__1": [0.3],  # R
                      "a62d4691_0_0-45_796__42__3": [0.3],  # S
                      "SoMe_42_with_visual_criteria_without_mesh-(100_40)-1p75_CVG": [0.1],  # T
                      "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-1p0_CVG": [0.2],  # U
                },
           }

# ############################################################################################################ V1
#
# fig, axs = plt.subplots(4, 1, figsize=(15, 10))
# bar_width = 0.15
# x = [i * 7 * bar_width + 2.5 * bar_width for i in range(6)]
#
# # Create a bar plot for simulated 42/
# for i_trial, trial in enumerate(names_42_simulations):
#     for i_criteria in range(len(criteria)):
#         total = 0
#         for i_name, name in enumerate(ratings_coaches):
#             axs[0].bar(i_criteria * bar_width + i_trial * 7 * bar_width, ratings_coaches[name][trial][i_criteria],
#                     width=bar_width,
#                     bottom=total,
#                     color=colors[i_criteria],
#                     alpha=0.5)
#             axs[0].plot(np.array([i_criteria * bar_width + i_trial * 7 * bar_width - bar_width/2,
#                                i_criteria * bar_width + i_trial * 7 * bar_width + bar_width/2]),
#                      np.array([total+ratings_coaches[name][trial][i_criteria], total+ratings_coaches[name][trial][i_criteria]]),
#                     color=colors[i_criteria])
#             total += ratings_coaches[name][trial][i_criteria]
# axs[0].set_xticks(x)
# axs[0].set_xticklabels(conditions_simulations)
# axs[0].set_xlim(-bar_width, 7 * bar_width*6)
# axs[0].set_ylim(0, 20)
#
# # Create a bar plot for real 42/
# for i_trial, trial in enumerate(names_42_real):
#     for i_criteria in range(len(criteria)):
#         total = 0
#         for i_name, name in enumerate(ratings_coaches):
#             axs[1].bar(i_criteria * bar_width + i_trial * 7 * bar_width, ratings_coaches[name][trial][i_criteria],
#                     width=bar_width,
#                     bottom=total,
#                     color=colors[i_criteria],
#                     alpha=0.5)
#             axs[1].plot(np.array([i_criteria * bar_width + i_trial * 7 * bar_width - bar_width/2,
#                                i_criteria * bar_width + i_trial * 7 * bar_width + bar_width/2]),
#                      np.array([total+ratings_coaches[name][trial][i_criteria], total+ratings_coaches[name][trial][i_criteria]]),
#                     color=colors[i_criteria])
#             total += ratingratings_coachess[name][trial][i_criteria]
# axs[1].set_xticks(x)
# axs[1].set_xticklabels(conditions_reel)
# axs[1].set_xlim(-bar_width, 7 * bar_width*6)
# axs[1].set_ylim(0, 20)
#
# # Create a bar plot for simulated 831<
# for i_trial, trial in enumerate(names_831_simulations):
#     for i_criteria in range(len(criteria)):
#         total = 0
#         for i_name, name in enumerate(ratings_coaches):
#             axs[2].bar(i_criteria * bar_width + i_trial * 7 * bar_width, ratings_coaches[name][trial][i_criteria],
#                     width=bar_width,
#                     bottom=total,
#                     color=colors[i_criteria],
#                     alpha=0.5)
#             axs[2].plot(np.array([i_criteria * bar_width + i_trial * 7 * bar_width - bar_width/2,
#                                i_criteria * bar_width + i_trial * 7 * bar_width + bar_width/2]),
#                      np.array([total+ratings_coaches[name][trial][i_criteria], total+ratings_coaches[name][trial][i_criteria]]),
#                     color=colors[i_criteria])
#             total += ratings_coaches[name][trial][i_criteria]
# axs[2].set_xticks(x)
# axs[2].set_xticklabels(conditions_simulations)
# axs[2].set_xlim(-bar_width, 7 * bar_width*6)
# axs[2].set_ylim(0, 20)
#
# # Create a bar plot for real 831<
# for i_trial, trial in enumerate(names_831_real):
#     for i_criteria in range(len(criteria)):
#         total = 0
#         for i_name, name in enumerate(ratings_coaches):
#             axs[3].bar(i_criteria * bar_width + i_trial * 7 * bar_width, ratings_coaches[name][trial][i_criteria],
#                     width=bar_width,
#                     bottom=total,
#                     color=colors[i_criteria],
#                     alpha=0.5)
#             axs[3].plot(np.array([i_criteria * bar_width + i_trial * 7 * bar_width - bar_width/2,
#                                i_criteria * bar_width + i_trial * 7 * bar_width + bar_width/2]),
#                      np.array([total+ratings_coaches[name][trial][i_criteria], total+ratings_coaches[name][trial][i_criteria]]),
#                     color=colors[i_criteria])
#             total += ratings_coaches[name][trial][i_criteria]
# axs[3].set_xticks(x)
# axs[3].set_xticklabels(conditions_reel)
# axs[3].set_xlim(-bar_width, 7 * bar_width*6)
# axs[3].set_ylim(0, 20)
#
# for i_criteria in range(len(criteria)):
#     axs[0].plot(-1, -1, color=colors[i_criteria], label=criteria[i_criteria])
# axs[0].legend(bbox_to_anchor=(0.5, 2.25), loc='upper center')
#
# plt.subplots_adjust(top=0.80, bottom=0.05, left=0.05, right=0.95)
# plt.savefig('coaches_ratings.png', dpi=300)
# plt.show()
#
#
# fig, axs = plt.subplots(4, 1, figsize=(15, 10))
# bar_width = 0.15
# x = [i * bar_width + 2.5 * bar_width for i in range(6)]
#
# # Create a bar plot for simulated 42/
# for i_trial, trial in enumerate(names_42_simulations):
#     total = 0
#     for i_name, name in enumerate(ratings_judges):
#         axs[0].bar(i_trial * 7 * bar_width, ratings_judges[name][trial][0],
#                 width=bar_width,
#                 bottom=total,
#                 color=colors[2],
#                 alpha=0.5)
#         axs[0].plot(np.array([i_trial * 7 * bar_width - bar_width/2,
#                            i_trial * 7 * bar_width + bar_width/2]),
#                  np.array([total+ratings_judges[name][trial][0], total+ratings_judges[name][trial][0]]),
#                 color=colors[2])
#         total += ratings_judges[name][trial][0]
# axs[0].set_xticks(x)
# axs[0].set_xticklabels(conditions_simulations)
# axs[0].set_xlim(-bar_width, 7 * bar_width*6)
# axs[0].set_ylim(0, 1)
#
# # Create a bar plot for real 42/
# for i_trial, trial in enumerate(names_42_real):
#     total = 0
#     for i_name, name in enumerate(ratings_judges):
#         axs[1].bar(i_trial * 7 * bar_width, ratings_judges[name][trial][0],
#                 width=bar_width,
#                 bottom=total,
#                 color=colors[2],
#                 alpha=0.5)
#         axs[1].plot(np.array([i_trial * 7 * bar_width - bar_width/2,
#                            i_trial * 7 * bar_width + bar_width/2]),
#                  np.array([total+ratings_judges[name][trial][0], total+ratings_judges[name][trial][0]]),
#                 color=colors[2])
#         total += ratings_judgesname][trial][0]
# axs[1].set_xticks(x)
# axs[1].set_xticklabels(conditions_reel)
# axs[1].set_xlim(-bar_width, 7 * bar_width*6)
# axs[1].set_ylim(0, 1)
#
# # Create a bar plot for simulated 831<
# for i_trial, trial in enumerate(names_831_simulations):
#     total = 0
#     for i_name, name in enumerate(ratings_judges):
#         axs[2].bar(i_trial * 7 * bar_width, ratings_judges[name][trial][0],
#                 width=bar_width,
#                 bottom=total,
#                 color=colors[2],
#                 alpha=0.5)
#         axs[2].plot(np.array([i_trial * 7 * bar_width - bar_width/2,
#                            i_trial * 7 * bar_width + bar_width/2]),
#                  np.array([total+ratings_judges[name][trial][0], total+ratings_judges[name][trial][0]]),
#                 color=colors[2])
#         total += ratings_judges[name][trial][0]
# axs[2].set_xticks(x)
# axs[2].set_xticklabels(conditions_simulations)
# axs[2].set_xlim(-bar_width, 7 * bar_width*6)
# axs[2].set_ylim(0, 1)
#
# # Create a bar plot for real 831<
# for i_trial, trial in enumerate(names_831_real):
#     total = 0
#     for i_name, name in enumerate(ratings_judges):
#         axs[3].bar(i_trial * 7 * bar_width, ratings_judges[name][trial][0],
#                 width=bar_width,
#                 bottom=total,
#                 color=colors[2],
#                 alpha=0.5)
#         axs[3].plot(np.array([i_trial * 7 * bar_width - bar_width/2,
#                            i_trial * 7 * bar_width + bar_width/2]),
#                  np.array([total+ratings_judges[name][trial][0], total+ratings_judges[name][trial][0]]),
#                 color=colors[2])
#         total += ratings_judges[name][trial][0]
# axs[3].set_xticks(x)
# axs[3].set_xticklabels(conditions_reel)
# axs[3].set_xlim(-bar_width, 7 * bar_width*6)
# axs[3].set_ylim(0, 1)
#
# plt.savefig('judges_ratings.png', dpi=300)
# plt.show()

############################################################################################################ V2

fig, axs = plt.subplots(2, 7, figsize=(15, 8))
bar_width = 0.15
x = [i * bar_width for i in range(6)]

# Create a bar plot for simulated 42/
for i_trial, trial in enumerate(names_42_simulations):
    for i_criteria in range(len(criteria)):
        total = 0
        for i_name, name in enumerate(ratings_coaches):
            axs[0, i_criteria].bar(i_trial * bar_width, ratings_coaches[name][trial][i_criteria],
                    width=bar_width,
                    bottom=total,
                    color=colors[i_name],
                    alpha=0.5)
            # axs[0, i_criteria].plot(np.array([i_trial * bar_width - bar_width/2,
            #                    i_trial * bar_width + bar_width/2]),
            #          np.array([total+ratings_coaches[name][trial][i_criteria], total+ratings_coaches[name][trial][i_criteria]]),
            #         color=colors[i_name])
            total += ratings_coaches[name][trial][i_criteria]
    total = 0
    for i_name, name in enumerate(ratings_judges):
        axs[0, 6].bar(i_trial * bar_width, ratings_judges[name][trial][0],
                               width=bar_width,
                               bottom=total,
                               color=colors[i_name],
                               alpha=0.5)
        # axs[0, i_criteria].plot(np.array([i_trial * bar_width - bar_width/2,
        #                    i_trial * bar_width + bar_width/2]),
        #          np.array([total+ratings_coaches[name][trial][i_criteria], total+ratings_coaches[name][trial][i_criteria]]),
        #         color=colors[i_name])
        total += ratings_judges[name][trial][0]

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
    axs[0, i_criteria].plot(7*bar_width-bar_width/2, mean_total[i_criteria], marker="s", color="k")
    axs[0, i_criteria].plot(np.array([-bar_width/2, 7*bar_width]), np.array([mean_total[i_criteria], mean_total[i_criteria]]), color="k")
    axs[0, i_criteria].plot(np.array([7*bar_width-bar_width/2, 7*bar_width-bar_width/2]), np.array([min_total[i_criteria], max_total[i_criteria]]), color="k", alpha=0.5)
    axs[0, i_criteria].plot(np.array([7*bar_width-bar_width/2-0.05, 7*bar_width-bar_width/2+0.05]), np.array([min_total[i_criteria], min_total[i_criteria]]), color="k", alpha=0.5)
    axs[0, i_criteria].plot(np.array([7*bar_width-bar_width/2-0.05, 7*bar_width-bar_width/2+0.05]), np.array([max_total[i_criteria], max_total[i_criteria]]), color="k", alpha=0.5)

# Create a bar plot for simulated 831<
for i_trial, trial in enumerate(names_831_simulations):
    for i_criteria in range(len(criteria)):
        total = 0
        for i_name, name in enumerate(ratings_coaches):
            axs[1, i_criteria].bar(i_trial * bar_width, ratings_coaches[name][trial][i_criteria],
                    width=bar_width,
                    bottom=total,
                    color=colors[i_name],
                    alpha=0.5)
            # axs[1, i_criteria].plot(np.array([i_criteria * bar_width + i_trial * 7 * bar_width - bar_width/2,
            #                    i_criteria * bar_width + i_trial * 7 * bar_width + bar_width/2]),
            #          np.array([total+ratings_coaches[name][trial][i_criteria], total+ratings_coaches[name][trial][i_criteria]]),
            #         color=colors[i_criteria])
            total += ratings_coaches[name][trial][i_criteria]
    total = 0
    for i_name, name in enumerate(ratings_judges):
        axs[1, 6].bar(i_trial * bar_width, ratings_judges[name][trial][0],
                               width=bar_width,
                               bottom=total,
                               color=colors[i_name],
                               alpha=0.5)
        # axs[0, i_criteria].plot(np.array([i_trial * bar_width - bar_width/2,
        #                    i_trial * bar_width + bar_width/2]),
        #          np.array([total+ratings_coaches[name][trial][i_criteria], total+ratings_coaches[name][trial][i_criteria]]),
        #         color=colors[i_name])
        total += ratings_judges[name][trial][0]

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
    axs[1, i_criteria].plot(7*bar_width-bar_width/2, mean_total[i_criteria], marker="s", color="k")
    axs[1, i_criteria].plot(np.array([-bar_width/2, 7*bar_width]), np.array([mean_total[i_criteria], mean_total[i_criteria]]), color="k")
    axs[1, i_criteria].plot(np.array([7*bar_width-bar_width/2, 7*bar_width-bar_width/2]), np.array([min_total[i_criteria], max_total[i_criteria]]), color="k", alpha=0.5)
    axs[1, i_criteria].plot(np.array([7*bar_width-bar_width/2-0.05, 7*bar_width-bar_width/2+0.05]), np.array([min_total[i_criteria], min_total[i_criteria]]), color="k", alpha=0.5)
    axs[1, i_criteria].plot(np.array([7*bar_width-bar_width/2-0.05, 7*bar_width-bar_width/2+0.05]), np.array([max_total[i_criteria], max_total[i_criteria]]), color="k", alpha=0.5)

axs[0, 0].set_ylabel("42/", fontsize=20)
axs[1, 0].set_ylabel("831<", fontsize=20)
for i_criteria in range(len(criteria)):
    axs[0, i_criteria].set_title(criteria[i_criteria])
axs[0, 6].set_title("Judges ratings")
for i_trial in range(7):
    axs[0, i_trial].set_xticks(x)
    axs[0, i_trial].set_xticklabels(conditions_simulations, rotation=-90)
    axs[1, i_trial].set_xticks(x)
    axs[1, i_trial].set_xticklabels(conditions_simulations, rotation=-90)

plt.tight_layout()
plt.savefig('coaches_ratings.png', dpi=300)
plt.show()

