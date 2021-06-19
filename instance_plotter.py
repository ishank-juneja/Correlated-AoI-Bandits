from matplotlib import pyplot as plt
import numpy as np
import argparse
import pathlib


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
x = np.arange(1, len(dist) + 1)
plt.figure(figsize=(6, 6))
plt.xticks(x, my_xticks, fontsize=24)
plt.yticks(fontsize=20)
# Make plot using stem impulses
plt.scatter(x, dist, s=160, color='#FA8C12')
markerline, stemlines, baseline = \
    plt.stem(x, dist, basefmt='grey', linefmt='C7--', markerfmt='C1o',
             bottom=0.0, use_line_collection=True)
# plt.xlabel("Support Points", fontweight='bold')
# plt.ylabel("pmf", fontsize=20)
plt.title("Distribution $p_{X}$", fontweight='bold', fontsize=20)
plt.grid(which='both', axis='both', color='k', alpha=0.1)
# plt.savefig(in_file + "_plot" + ".svg", bbox_inches="tight")
plt.savefig('results/' + in_name + '/' + in_name + "-dist" + ".png")
plt.close()

# Use matlab C{i} type notation for colors
# Plot the arm/channel functions on separate plots
for i in range(n_arms):
    plt.figure(figsize=(6, 6))
    cur_color = "C{0}".format(i + 1)
    plt.xticks(x, my_xticks, fontsize=24)
    plt.yticks(fontsize=20)
    plt.plot(x, arm_list[i, :], linewidth=2.5, color=cur_color)
    plt.scatter(x, arm_list[i, :], s=120, color=cur_color)
    # plt.xlabel("Support Points", fontweight='bold')
    plt.ylabel("Reward", fontweight='bold', fontsize=20)
    plt.title("$g_{0}(x)$".format(i + 1), fontweight='bold', fontsize=20)
    plt.grid(which='both', axis='both', color='k', alpha=0.1)
    plt.ylim(-0.1, B + 0.1)
    plt.savefig('results/' + in_name + "/arm{0}_plot_line.png".format(i + 1), bbox_inches="tight")
    plt.close()

# # Combined plot of all arms
# plt.figure(figsize=(6, 6))
# for i in range(n_arms):
#     cur_color = "C{0}".format(i + 1)
#     plt.xticks(x, my_xticks, fontsize=24)
#     plt.yticks(fontsize=20)
#     plt.plot(x, arm_list[i, :], linewidth=2.5, color=cur_color)
#     plt.scatter(x, arm_list[i, :], s=120, color=cur_color)
# plt.ylabel("Reward", fontweight='bold', fontsize=20)
# plt.title("All Reward Arms", fontweight='bold', fontsize=20)
# plt.grid(which='both', axis='both', color='k', alpha=0.1)
# # plt.ylim(bottom=0, top=5)
# plt.savefig('results/' + in_name + "/all_arms_plot.png", bbox_inches="tight")
# plt.close()
