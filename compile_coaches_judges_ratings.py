import numpy as np
import matplotlib.pyplot as plt


colors = ['#151F31', '#103778', '#0593A2', '#FF7A48', '#E3371E', '#E5003F']


### ------------- Choaches ratings -------------- ###

ratings = {
    "Stephen": {
        "SoMe_42_with_visual_criteria_without_mesh-(100_40)-1p0_CVG": [2, 3, 2, 2, 4, 2],
        "a62d4691_0_0-45_796__42__1": [5, 5, 5, 5, 5, 5],
        "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-1p0_CVG": [2, 2, 2, 2, 4, 2],
        "a62d4691_0_0-45_796__42__0": [4, 4, 4, 2, 3, 2],
        "87972c15_0_0-105_114__831__1": [2, 2, 3, 1, 3, 2],
        "SoMe_42_with_visual_criteria_without_mesh-(100_40)-0p25_CVG": [5, 5, 5, 5, 5, 5],
        "SoMe_42_with_visual_criteria_without_mesh-(100_40)-2p0_CVG": [2, 3, 3, 2, 3, 2],
        "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-1p75_CVG": [4, 4, 4, 2, 4, 4],
        "SoMe_42_without_mesh-(100_40)-0p0_CVG": [4, 4, 4, 4, 4, 4],
        "a62d4691_0_0-45_796__42__3": [2, 4, 4, 2, 4, 2],
        "SoMe_42_with_visual_criteria_without_mesh-(100_40)-1p75_CVG": [4, 2, 3, 2, 4, 2],
        "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-2p0_CVG": [4, 4, 4, 2, 4, 3],
        "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-0p5_CVG": [4, 4, 4, 4, 4, 4],
        "87972c15_0_0-105_114__831__0": [2, 3, 3, 1, 3, 1],
        "SoMe_without_mesh_831-(40_40_40_40_40_40)-2023-0p0_CVG": [5, 5, 5, 4, 5, 5],
        "87972c15_0_0-105_114__831__2": [4, 4, 4, 2, 4, 2],
        "a62d4691_0_0-45_796__42__2": [3, 4, 4, 2, 4, 2],
        "c2a19850_0_0-113_083__831__0": [2, 3, 3, 2, 3, 1],
        "SoMe_42_with_visual_criteria_without_mesh-(100_40)-0p5_CVG": [3, 4, 4, 2, 4, 2],
        "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-0p25_CVG": [4, 4, 4, 4, 4, 4],
        "caccfb24_0_0-48_286__42__1": [4, 4, 4, 4, 4, 4],
    },
    "Eve": {  # fake data to build the graph
        "SoMe_42_with_visual_criteria_without_mesh-(100_40)-1p0_CVG": [2, 3, 2, 2, 4, 2],
        "a62d4691_0_0-45_796__42__1": [5, 5, 5, 5, 5, 5],
        "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-1p0_CVG": [2, 2, 2, 2, 4, 2],
        "a62d4691_0_0-45_796__42__0": [4, 4, 4, 2, 3, 2],
        "87972c15_0_0-105_114__831__1": [2, 2, 3, 1, 3, 2],
        "SoMe_42_with_visual_criteria_without_mesh-(100_40)-0p25_CVG": [5, 5, 5, 5, 5, 5],
        "SoMe_42_with_visual_criteria_without_mesh-(100_40)-2p0_CVG": [2, 3, 3, 2, 3, 2],
        "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-1p75_CVG": [4, 4, 4, 2, 4, 4],
        "SoMe_42_without_mesh-(100_40)-0p0_CVG": [4, 4, 4, 4, 4, 4],
        "a62d4691_0_0-45_796__42__3": [2, 4, 4, 2, 4, 2],
        "SoMe_42_with_visual_criteria_without_mesh-(100_40)-1p75_CVG": [4, 2, 3, 2, 4, 2],
        "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-2p0_CVG": [4, 4, 4, 2, 4, 3],
        "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-0p5_CVG": [4, 4, 4, 4, 4, 4],
        "87972c15_0_0-105_114__831__0": [2, 3, 3, 1, 3, 1],
        "SoMe_without_mesh_831-(40_40_40_40_40_40)-2023-0p0_CVG": [5, 5, 5, 4, 5, 5],
        "87972c15_0_0-105_114__831__2": [4, 4, 4, 2, 4, 2],
        "a62d4691_0_0-45_796__42__2": [3, 4, 4, 2, 4, 2],
        "c2a19850_0_0-113_083__831__0": [2, 3, 3, 2, 3, 1],
        "SoMe_42_with_visual_criteria_without_mesh-(100_40)-0p5_CVG": [3, 4, 4, 2, 4, 2],
        "SoMe_with_visual_criteria_without_mesh_831-(40_40_40_40_40_40)-0p25_CVG": [4, 4, 4, 4, 4, 4],
        "caccfb24_0_0-48_286__42__1": [4, 4, 4, 4, 4, 4],
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

criteria = ['This techniques is efficient for aerial twist creation',
            'This technique is safe for an athlete to try',
            'Overall, this technique seems realistic',
            'This technique is aesthetic',
            'This technique allow the athlete to get appropriate visual information',
            'I would recommend my athletes to use this technique']



fig, axs = plt.subplots(4, 1, figsize=(15, 10))
bar_width = 0.15
x = [i * 7 * bar_width + 2.5 * bar_width for i in range(6)]

# Create a bar plot for simulated 42/
for i_trial, trial in enumerate(names_42_simulations):
    for i_criteria in range(len(criteria)):
        total = 0
        for i_name, name in enumerate(ratings):
            axs[0].bar(i_criteria * bar_width + i_trial * 7 * bar_width, ratings[name][trial][i_criteria],
                    width=bar_width,
                    bottom=total,
                    color=colors[i_criteria],
                    alpha=0.5)
            axs[0].plot(np.array([i_criteria * bar_width + i_trial * 7 * bar_width - bar_width/2,
                               i_criteria * bar_width + i_trial * 7 * bar_width + bar_width/2]),
                     np.array([total+ratings[name][trial][i_criteria], total+ratings[name][trial][i_criteria]]),
                    color=colors[i_criteria])
            total += ratings[name][trial][i_criteria]
axs[0].set_xticks(x)
axs[0].set_xticklabels(conditions_simulations)
axs[0].set_xlim(-bar_width, 7 * bar_width*6)

# Create a bar plot for real 42/
for i_trial, trial in enumerate(names_42_real):
    for i_criteria in range(len(criteria)):
        total = 0
        for i_name, name in enumerate(ratings):
            axs[1].bar(i_criteria * bar_width + i_trial * 7 * bar_width, ratings[name][trial][i_criteria],
                    width=bar_width,
                    bottom=total,
                    color=colors[i_criteria],
                    alpha=0.5)
            axs[1].plot(np.array([i_criteria * bar_width + i_trial * 7 * bar_width - bar_width/2,
                               i_criteria * bar_width + i_trial * 7 * bar_width + bar_width/2]),
                     np.array([total+ratings[name][trial][i_criteria], total+ratings[name][trial][i_criteria]]),
                    color=colors[i_criteria])
            total += ratings[name][trial][i_criteria]
axs[1].set_xticks(x)
axs[1].set_xticklabels(conditions_reel)
axs[1].set_xlim(-bar_width, 7 * bar_width*6)

# Create a bar plot for simulated 831<
for i_trial, trial in enumerate(names_831_simulations):
    for i_criteria in range(len(criteria)):
        total = 0
        for i_name, name in enumerate(ratings):
            axs[2].bar(i_criteria * bar_width + i_trial * 7 * bar_width, ratings[name][trial][i_criteria],
                    width=bar_width,
                    bottom=total,
                    color=colors[i_criteria],
                    alpha=0.5)
            axs[2].plot(np.array([i_criteria * bar_width + i_trial * 7 * bar_width - bar_width/2,
                               i_criteria * bar_width + i_trial * 7 * bar_width + bar_width/2]),
                     np.array([total+ratings[name][trial][i_criteria], total+ratings[name][trial][i_criteria]]),
                    color=colors[i_criteria])
            total += ratings[name][trial][i_criteria]
axs[2].set_xticks(x)
axs[2].set_xticklabels(conditions_simulations)
axs[2].set_xlim(-bar_width, 7 * bar_width*6)

# Create a bar plot for real 831<
for i_trial, trial in enumerate(names_831_real):
    for i_criteria in range(len(criteria)):
        total = 0
        for i_name, name in enumerate(ratings):
            axs[3].bar(i_criteria * bar_width + i_trial * 7 * bar_width, ratings[name][trial][i_criteria],
                    width=bar_width,
                    bottom=total,
                    color=colors[i_criteria],
                    alpha=0.5)
            axs[3].plot(np.array([i_criteria * bar_width + i_trial * 7 * bar_width - bar_width/2,
                               i_criteria * bar_width + i_trial * 7 * bar_width + bar_width/2]),
                     np.array([total+ratings[name][trial][i_criteria], total+ratings[name][trial][i_criteria]]),
                    color=colors[i_criteria])
            total += ratings[name][trial][i_criteria]
axs[3].set_xticks(x)
axs[3].set_xticklabels(conditions_reel)
axs[3].set_xlim(-bar_width, 7 * bar_width*6)

for i_criteria in range(len(criteria)):
    axs[0].plot(-1, -1, color=colors[i_criteria], label=criteria[i_criteria])
axs[0].legend(bbox_to_anchor=(0.5, 2.25), loc='upper center')
axs[0].set_ylim(0, 10)

plt.subplots_adjust(top=0.80, bottom=0.05, left=0.05, right=0.95)
plt.savefig('coaches_ratings', format='eps', dpi=300)
plt.show()


### ------------- Judges ratings -------------- ###

ratings = {"Stephan": {"A": [1],
                       "B": [1],
                       "C": [1],
                       "D": [1],
                       "E": [1],
                       "F": [1],
                       "G": [1],
                       "H": [1],
                       "I": [1],
                       "J": [1],
                       "K": [1],
                       "L": [1],
                       "M": [1],
                       "N": [1],
                       "O": [1],
                       "P": [1],
                       "Q": [1],
                       "R": [1],
                       "S": [1],
                       "T": [1],
                       "U": [1],
                       },
            "Julie": {"A": [1],
                      "B": [1],
                      "C": [1],
                      "D": [1],
                      "E": [1],
                      "F": [1],
                      "G": [1],
                      "H": [1],
                      "I": [1],
                      "J": [1],
                      "K": [1],
                      "L": [1],
                      "M": [1],
                      "N": [1],
                      "O": [1],
                      "P": [1],
                      "Q": [1],
                      "R": [1],
                      "S": [1],
                      "T": [1],
                      "U": [1],
                },
           }


fig, axs = plt.subplots(4, 1, figsize=(15, 10))
bar_width = 0.15
x = [i * bar_width + 2.5 * bar_width for i in range(6)]

# Create a bar plot for simulated 42/
for i_trial, trial in enumerate(names_42_simulations):
    total = 0
    for i_name, name in enumerate(ratings):
        axs[0].bar(i_trial * 7 * bar_width, ratings[name][trial][0],
                width=bar_width,
                bottom=total,
                color=colors[2],
                alpha=0.5)
        axs[0].plot(np.array([i_trial * 7 * bar_width - bar_width/2,
                           i_trial * 7 * bar_width + bar_width/2]),
                 np.array([total+ratings[name][trial][0], total+ratings[name][trial][0]]),
                color=colors[2])
        total += ratings[name][trial][0]
axs[0].set_xticks(x)
axs[0].set_xticklabels(conditions_simulations)
axs[0].set_xlim(-bar_width, 7 * bar_width*6)

# Create a bar plot for real 42/
for i_trial, trial in enumerate(names_42_real):
    total = 0
    for i_name, name in enumerate(ratings):
        axs[1].bar(i_trial * 7 * bar_width, ratings[name][trial][0],
                width=bar_width,
                bottom=total,
                color=colors[2],
                alpha=0.5)
        axs[1].plot(np.array([i_trial * 7 * bar_width - bar_width/2,
                           i_trial * 7 * bar_width + bar_width/2]),
                 np.array([total+ratings[name][trial][0], total+ratings[name][trial][0]]),
                color=colors[2])
        total += ratings[name][trial][0]
axs[1].set_xticks(x)
axs[1].set_xticklabels(conditions_reel)
axs[1].set_xlim(-bar_width, 7 * bar_width*6)

# Create a bar plot for simulated 831<
for i_trial, trial in enumerate(names_831_simulations):
    total = 0
    for i_name, name in enumerate(ratings):
        axs[2].bar(i_trial * 7 * bar_width, ratings[name][trial][0],
                width=bar_width,
                bottom=total,
                color=colors[2],
                alpha=0.5)
        axs[2].plot(np.array([i_trial * 7 * bar_width - bar_width/2,
                           i_trial * 7 * bar_width + bar_width/2]),
                 np.array([total+ratings[name][trial][0], total+ratings[name][trial][0]]),
                color=colors[2])
        total += ratings[name][trial][0]
axs[2].set_xticks(x)
axs[2].set_xticklabels(conditions_simulations)
axs[2].set_xlim(-bar_width, 7 * bar_width*6)

# Create a bar plot for real 831<
for i_trial, trial in enumerate(names_831_real):
    total = 0
    for i_name, name in enumerate(ratings):
        axs[3].bar(i_trial * 7 * bar_width, ratings[name][trial][0],
                width=bar_width,
                bottom=total,
                color=colors[2],
                alpha=0.5)
        axs[3].plot(np.array([i_trial * 7 * bar_width - bar_width/2,
                           i_trial * 7 * bar_width + bar_width/2]),
                 np.array([total+ratings[name][trial][0], total+ratings[name][trial][0]]),
                color=colors[2])
        total += ratings[name][trial][0]
axs[3].set_xticks(x)
axs[3].set_xticklabels(conditions_reel)
axs[3].set_xlim(-bar_width, 7 * bar_width*6)
axs[0].set_ylim(0, 10)

plt.savefig('judges_ratings', format='eps', dpi=300)
plt.show()