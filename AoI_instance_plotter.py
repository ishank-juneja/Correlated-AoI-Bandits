import numpy as np
import argparse
import pathlib
from matplotlib import pyplot as plt
from matplotlib import rc
# For Type 3 free fonts in the figures
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
# for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


# Command line input
parser = argparse.ArgumentParser()
parser.add_argument("-i", action="store", dest="file")
args = parser.parse_args()
in_file = args.file
# Extract out just file name
in_name = in_file.split('/')[-1].split('.')[0]
# Create folder to save results if it doesn't already exist
pathlib.Path('results/' + in_name).mkdir(parents=False, exist_ok=True)
# Open bandit instance file to be read
fin = open(in_file, 'r')
# Plot the distribution
dist_str = fin.readline()
# Convert read in string of values to float array
dist = [float(x) for x in dist_str.split(' ')]
# Read in lines corresponding to arm functions
functions_str = fin.readlines()
# Count number of arms
n_arms = len(functions_str)
# Convert strings to lists
arm_list = []
for i in range(n_arms):
    # Read in functions as 0-1 integers
    arm_list.append([float(s) for s in functions_str[i].split(' ')])
# Covert to numpy array for sliced indexing
arm_list = np.array(arm_list)
# Get largest reward in bandit instance
B = np.max(arm_list)
# Get list of data support points as
# x1, x2, ...
my_xticks = []
for i in range(len(dist)):
    my_xticks.append('$x_{0}$'.format(i + 1))
# Plot the distribution
x = np.cumsum(dist)
numerical_xticks = []
for i in range(len(dist)):
    numerical_xticks.append('${0}$'.format(x[i]))

# For line starts and ends
x_begin = np.insert(x[:-1], 0, 0.0)
# Get a transition array indicating points where transition happens
arm_transitions = arm_list[:, :-1] != arm_list[:, 1:]
# plt.figure(figsize=(6, 6))
# plt.xticks(x, my_xticks, fontsize=24)
# plt.yticks(fontsize=20)
# # Make plot using stem impulses
# plt.scatter(x, dist, s=160, color='#FA8C12')
# markerline, stemlines, baseline = \
#     plt.stem(x, dist, basefmt='grey', linefmt='C7--', markerfmt='C1o',
#              bottom=0.0, use_line_collection=True)
# # plt.xlabel("Support Points", fontweight='bold')
# # plt.ylabel("pmf", fontsize=20)
# plt.title("Distribution $p_{X}$", fontweight='bold', fontsize=20)
# plt.grid(which='both', axis='both', color='k', alpha=0.1)
# # plt.savefig(in_file + "_plot" + ".svg", bbox_inches="tight")
# # plt.savefig('results/' + in_name + '/' + in_name + "-dist" + ".png")
# plt.show()
# plt.close()

# Plot the arm/channel functions on separate plots
for i in range(n_arms):
    plt.figure(figsize=(6, 2))
    plt.xticks(x, my_xticks, fontsize=24)
    plt.yticks([0, 1], fontsize=20)
    plt.hlines(arm_list[i, :], x_begin, x, linestyles='solid', alpha=1.0, zorder=-1)
    # list for vertical transitions
    transitions = []
    for j in range(arm_list.shape[1] - 1):
        if arm_transitions[i, j]:
            transitions.append(x[j])
    plt.vlines(transitions, 0, 1, linestyles='dashed', alpha=0.2, zorder=-1)
    plt.scatter(x, arm_list[i, :], s=200, zorder=4, facecolors='k', edgecolors='k')
    plt.scatter(x[:-1], arm_list[i, 1:], s=200, zorder=3, facecolors='w', edgecolors='k')
    # plt.xlabel("Support Points", fontweight='bold')
    plt.ylabel("Reward", fontweight='bold', fontsize=20)
    plt.title("$Y_{0}(X)$".format(i + 1), fontweight='bold', fontsize=20)
    plt.grid(which='both', axis='both', color='k', alpha=0.1, zorder=0)
    plt.ylim(-0.1, B + 0.1)
    plt.xlim(0, 1.05)
    plt.savefig('results/' + in_name + "/arm{0}_plot.png".format(i + 1), bbox_inches="tight")
    # plt.show()
    plt.close()

# Combined plot for all arms
fig = plt.figure(figsize=(8, 6))
# Populate the subplots
for i in range(n_arms):
    plt.subplot(2, 2, i+1)
    # plt.xticks(x, my_xticks, fontsize=24)
    # plt.yticks([0, 1], fontsize=12)
    plt.grid(which='both', axis='both', color='k', alpha=0.2, zorder=0)
    # Create dark and light bands using polygon fill color
    x_corners = [0, x[0], x[0], 0]
    y_corners = [0, 0, 1, 1]
    plt.fill(x_corners, y_corners, color='k', alpha=0.07, zorder=-4)
    x_corners = [x[1], x[2], x[2], x[1]]
    y_corners = [0, 0, 1, 1]
    plt.fill(x_corners, y_corners, color='k', alpha=0.07, zorder=-4)
    # fill_x = np.array([0.0, x[0], x[]])
    # ax.fill_between(x, 0, 1, where=y > theta,
    #                 facecolor='k', alpha=0.2, transform=trans)
    # Mark xticks only for the bottom plots
    if i > 1:
        plt.xticks(x, my_xticks, fontsize=18)
    # For the top plots, make sure there are no xticks neither alphabetical nor numerical
    else:
        plt.xticks(x, numerical_xticks, fontsize=14)
        plt.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=True,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=True)  # True just as a hack to get desired gridlines
    plt.hlines(arm_list[i, :], x_begin, x, linestyles='solid', alpha=1.0, zorder=-1)
    # list for vertical transitions
    transitions = []
    for j in range(arm_list.shape[1] - 1):
        if arm_transitions[i, j]:
            transitions.append(x[j])
    plt.vlines(transitions, 0, 1, linestyles='dashed', alpha=1.0, zorder=-1)
    plt.scatter(x, arm_list[i, :], s=100, zorder=4, facecolors='k', edgecolors='k')
    plt.scatter(x[:-1], arm_list[i, 1:], s=100, zorder=3, facecolors='w', edgecolors='k')
    # plt.xlabel("Support Points", fontweight='bold')
    # Y-Label only for leftmost plots
    # if i % 2 == 0:
    plt.ylabel("$Y_{0}(X)$".format(i + 1), fontweight='bold', fontsize=14, labelpad=-12)
    # plt.ylabel("Reward", fontweight='bold', fontsize=20)
    plt.yticks([0, 1], ["$0$", "$1$"], fontsize=16)
    # Remove y axis for right most plots
    # else:
    #     plt.tick_params(
    #         axis='y',  # changes apply to the x-axis
    #         which='both',  # both major and minor ticks are affected
    #         left=False,  # ticks along the bottom edge are off
    #         right=False,  # ticks along the top edge are off
    #         labelleft=False)  # labels along the bottom edge are off
    # plt.text(0.55, 1.02, "$Y_{0}(X)$".format(i + 1), fontweight='bold', fontsize=20)
    plt.ylim(-0.1, B + 0.1)
    plt.xlim(0, 1.05)
plt.subplots_adjust(wspace=0.10, hspace=0.15)
plt.savefig('results/' + in_name + "/all_arms_plot_AoI.pdf".format(i + 1), bbox_inches="tight")
# plt.savefig('results/' + in_name + "/all_arms_plot_AoI.svg".format(i + 1), bbox_inches="tight")
# plt.show()
plt.close()
