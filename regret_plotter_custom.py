import pandas as pd
from matplotlib import pyplot as plt
# For Type 3 free fonts in the figures
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
import numpy as np
import matplotlib.ticker as mticker
import argparse
import pathlib

# Command line input
parser = argparse.ArgumentParser()
parser.add_argument("-i", action="store", dest="file")
parser.add_argument("-STEP", action="store", dest="STEP", type=int)
parser.add_argument("-horizon", action="store", dest="horizon", type=float)
args = parser.parse_args()
in_file = args.file
# Extract out just file name
in_name = in_file.split('/')[-1].split('.')[0]
# Ignore last 4 characters -out
in_name = in_name[:-4]
# Create folder to save results if it doesn't already exist
pathlib.Path('results/' + in_name).mkdir(parents=False, exist_ok=True)
x_points = np.arange(0, int(args.horizon) + 1, args.STEP)
# For this collection of x_points identify the indices that give
# the markers as [1000, 2000, 3000, ..., 9000]
if 1000 % args.STEP != 0:
    print("-STEP is not a factor of 1000")
    exit(-1)
# We need the indices of the points where we want the markers
markers_on = list(range(1000//args.STEP, 10000//args.STEP, 1000//args.STEP))
y_points = np.zeros_like(x_points)
# These are all the possible labels, of these atmost 8 can be supported with below colors
# LABELS = ['ucb', 'ts', 'qucb', 'qts', 'cucb', 'cts', 'u-cucb', 'new', 'cts-old', 'cucb-old']
# selected must be a subset of algos selected in simulate_policies.py
# All the algorithms selected must have already been simulated in simulate_policies.py
selected = ['ucb', 'aoi-aware-ucb', 'cucb', 'aoi-aware-cucb', 'ts', 'aoi-aware-ts', 'cts', 'aoi-aware-cts']
bandit_data = pd.read_csv(in_file, sep=",", header=None)
bandit_data.columns = ["algo", "rs", "eps", "horizon", "reg", "exp_reg", "AoI_reg", "exp_AoI_reg"]
# List of dependent variables to be plotted, from above list
dependent = "exp_AoI_reg"


# Plot and average the results for each label onto a single plot,
# doesn't make a lot of sense, just there
# COLORS = ['green', 'blue', 'darkorange', 'red']
# marker_styles = ['o', 's', '^', '*']
fig = plt.figure(figsize=(8, 10))
ax1 = plt.subplot2grid((10, 7), (0, 0), colspan=7, rowspan=7)
ax2 = plt.subplot2grid((20, 7), (15, 0), colspan=7, rowspan=7)
# Assign variables to data
# Get data points for each algorithm
y_points[0] = 0
# UCB
cur_data = bandit_data.loc[bandit_data["algo"] == 'ucb']
for i in range(1, len(y_points)):
    y_points[i] = cur_data.loc[cur_data["horizon"] == x_points[i]][dependent].mean()
ax1.plot(x_points, y_points, color='green', linewidth=2, linestyle='-', marker='o', markevery=markers_on, markersize=10)
# AA UCB
cur_data = bandit_data.loc[bandit_data["algo"] == 'aoi-aware-ucb']
for i in range(1, len(y_points)):
    y_points[i] = cur_data.loc[cur_data["horizon"] == x_points[i]][dependent].mean()
ax1.plot(x_points, y_points, color='green', linewidth=2, linestyle='--', marker='o', markevery=markers_on, markersize=10)
# CUCB
cur_data = bandit_data.loc[bandit_data["algo"] == 'cucb']
for i in range(1, len(y_points)):
    y_points[i] = cur_data.loc[cur_data["horizon"] == x_points[i]][dependent].mean()
ax1.plot(x_points, y_points, color='blue', linewidth=2, linestyle='-', marker='s', markevery=markers_on, markersize=10)
# AA - CUCB
cur_data = bandit_data.loc[bandit_data["algo"] == 'aoi-aware-cucb']
for i in range(1, len(y_points)):
    y_points[i] = cur_data.loc[cur_data["horizon"] == x_points[i]][dependent].mean()
ax1.plot(x_points, y_points, color='blue', linewidth=2, linestyle='--', marker='s', markevery=markers_on, markersize=10)
## TS
cur_data = bandit_data.loc[bandit_data["algo"] == 'ts']
for i in range(1, len(y_points)):
    y_points[i] = cur_data.loc[cur_data["horizon"] == x_points[i]][dependent].mean()
ax1.plot(x_points, y_points, color='darkorange', linewidth=2, linestyle='-', marker='^', markevery=markers_on, markersize=10)
ax2.plot(x_points, y_points, color='darkorange', linewidth=2, linestyle='-', marker='^', markevery=markers_on, markersize=10)
## AA TS
cur_data = bandit_data.loc[bandit_data["algo"] == 'aoi-aware-ts']
for i in range(1, len(y_points)):
    y_points[i] = cur_data.loc[cur_data["horizon"] == x_points[i]][dependent].mean()
ax1.plot(x_points, y_points, color='darkorange', linewidth=2, linestyle='--', marker='^', markevery=markers_on, markersize=10)
ax2.plot(x_points, y_points, color='darkorange', linewidth=2, linestyle='--', marker='^', markevery=markers_on, markersize=10)
## CTS
cur_data = bandit_data.loc[bandit_data["algo"] == 'cts']
for i in range(1, len(y_points)):
    y_points[i] = cur_data.loc[cur_data["horizon"] == x_points[i]][dependent].mean()
ax1.plot(x_points, y_points, color='red', linewidth=2, linestyle='-', marker='*', markevery=markers_on, markersize=10)
ax2.plot(x_points, y_points, color='red', linewidth=2, linestyle='-', marker='*', markevery=markers_on, markersize=10)
# AA-CTS
cur_data = bandit_data.loc[bandit_data["algo"] == 'aoi-aware-cts']
for i in range(1, len(y_points)):
    y_points[i] = cur_data.loc[cur_data["horizon"] == x_points[i]][dependent].mean()
ax1.plot(x_points, y_points, color='red', linewidth=2, linestyle='--', marker='*', markevery=markers_on, markersize=10)
ax2.plot(x_points, y_points, color='red', linewidth=2, linestyle='--', marker='*', markevery=markers_on, markersize=10)

# Top Plot
ax1.set_xticks(np.arange(0, 12000, 2000))
ax1.grid(which='both', axis='both', color='k')	# , alpha=0.5
ax1.set_ylabel('$R(T)$', fontsize=16)
ax1.set_xlim(xmin=0.0, xmax=args.horizon)
ax1.set_ylim(ymin=0.0, ymax=250)
# ax1.set_ylim(ymin=0.0)
ax1.grid(True, alpha=0.2)
ax1.legend(['UCB', 'AA UCB', 'CUCB', 'AA CUCB', 'TS', 'AA TS', 'CTS', 'AA CTS'],
           fontsize=11, handlelength=3, framealpha=1.0, ncol=2, loc="upper left")
ax1.tick_params(labelsize=12)
# Bottom Plot
ax2.set_xticks(np.arange(0, 12000, 2000))
ax2.grid(which='both', axis='both', color='k')  # , alpha=0.5
ax2.set_ylabel('$R(T)$', fontsize=16)
ax2.set_xlabel('Time ($t$)', fontsize=16)
# ax2.set_xlim(xmin=0.0, xmax=args.horizon)
ax2.set_ylim(ymin=0.0, ymax=57)
ax2.set_ylim(ymin=0.0)
ax2.grid(True, alpha=0.2)
# ax2.legend(['TS', 'AA TS', 'CTS', 'AA CTS'], fontsize=11, handlelength=3, framealpha=1.0,
#            ncol=2, loc="lower right")
ax2.legend(['TS', 'AA TS', 'CTS', 'AA CTS'], fontsize=11, handlelength=3, framealpha=1.0,
           ncol=2, loc="upper left")
ax2.tick_params(labelsize=12)
plt.savefig('results/' + in_name + "/{0}_{1}_complete_plot_step_{2}".format(in_name, dependent, args.STEP) + ".pdf", bbox_inches="tight")
plt.savefig('results/' + in_name + "/{0}_{1}_complete_plot_step_{2}".format(in_name, dependent, args.STEP) + ".svg", bbox_inches="tight")
plt.close()

# # Plot and average the results for each label onto a single plot,
# # doesn't make a lot of sense, just there
# # COLORS = ['green', 'blue', 'darkorange', 'red']
# # marker_styles = ['o', 's', '^', '*']
# fig = plt.figure(figsize=(8, 10))
# ax1 = plt.subplot2grid((10, 7), (0, 0), colspan=7, rowspan=7)
# ax2 = plt.subplot2grid((20, 7), (15, 0), colspan=7, rowspan=7)
# # Assign variables to data
# # Get data points for each algorithm
# y_points[0] = 0
# # UCB
# cur_data = bandit_data.loc[bandit_data["algo"] == 'ucb']
# for i in range(1, len(y_points)):
#     y_points[i] = cur_data.loc[cur_data["horizon"] == x_points[i]][dependent].mean()
# ax1.plot(x_points, y_points, color='green', linewidth=2, linestyle='-', marker='o', markevery=markers_on, markersize=10)
# # AA UCB
# cur_data = bandit_data.loc[bandit_data["algo"] == 'aoi-aware-ucb']
# for i in range(1, len(y_points)):
#     y_points[i] = cur_data.loc[cur_data["horizon"] == x_points[i]][dependent].mean()
# ax1.plot(x_points, y_points, color='green', linewidth=2, linestyle='--', marker='o', markevery=markers_on, markersize=10)
# # CUCB
# cur_data = bandit_data.loc[bandit_data["algo"] == 'cucb']
# for i in range(1, len(y_points)):
#     y_points[i] = cur_data.loc[cur_data["horizon"] == x_points[i]][dependent].mean()
# ax1.plot(x_points, y_points, color='blue', linewidth=2, linestyle='-', marker='s', markevery=markers_on, markersize=10)
# # AA - CUCB
# cur_data = bandit_data.loc[bandit_data["algo"] == 'aoi-aware-cucb']
# for i in range(1, len(y_points)):
#     y_points[i] = cur_data.loc[cur_data["horizon"] == x_points[i]][dependent].mean()
# ax1.plot(x_points, y_points, color='blue', linewidth=2, linestyle='--', marker='s', markevery=markers_on, markersize=10)
# ## TS
# cur_data = bandit_data.loc[bandit_data["algo"] == 'ts']
# for i in range(1, len(y_points)):
#     y_points[i] = cur_data.loc[cur_data["horizon"] == x_points[i]][dependent].mean()
# ax1.plot(x_points, y_points, color='darkorange', linewidth=2, linestyle='-', marker='^', markevery=markers_on, markersize=10)
# ax2.plot(x_points, y_points, color='darkorange', linewidth=2, linestyle='-', marker='^', markevery=markers_on, markersize=10)
# ## AA TS
# cur_data = bandit_data.loc[bandit_data["algo"] == 'aoi-aware-ts']
# for i in range(1, len(y_points)):
#     y_points[i] = cur_data.loc[cur_data["horizon"] == x_points[i]][dependent].mean()
# ax1.plot(x_points, y_points, color='darkorange', linewidth=2, linestyle='--', marker='^', markevery=markers_on, markersize=10)
# ax2.plot(x_points, y_points, color='darkorange', linewidth=2, linestyle='--', marker='^', markevery=markers_on, markersize=10)
# ## CTS
# cur_data = bandit_data.loc[bandit_data["algo"] == 'cts']
# for i in range(1, len(y_points)):
#     y_points[i] = cur_data.loc[cur_data["horizon"] == x_points[i]][dependent].mean()
# ax1.plot(x_points, y_points, color='red', linewidth=2, linestyle='-', marker='*', markevery=markers_on, markersize=10)
# ax2.plot(x_points, y_points, color='red', linewidth=2, linestyle='-', marker='*', markevery=markers_on, markersize=10)
# # AA-CTS
# cur_data = bandit_data.loc[bandit_data["algo"] == 'aoi-aware-cts']
# for i in range(1, len(y_points)):
#     y_points[i] = cur_data.loc[cur_data["horizon"] == x_points[i]][dependent].mean()
# ax1.plot(x_points, y_points, color='red', linewidth=2, linestyle='--', marker='*', markevery=markers_on, markersize=10)
# ax2.plot(x_points, y_points, color='red', linewidth=2, linestyle='--', marker='*', markevery=markers_on, markersize=10)
#
# # Top Plot
# ax1.set_xticks(np.arange(0, 12000, 2000))
# ax1.grid(which='both', axis='both', color='k')	# , alpha=0.5
# ax1.set_ylabel('$R(T)$', fontsize=16)
# ax1.set_xlim(xmin=0.0, xmax=args.horizon)
# ax1.set_ylim(ymin=0.0)
# ax1.grid(True, alpha=0.2)
# ax1.legend(['UCB', 'AA UCB', 'CUCB', 'AA CUCB', 'TS', 'AA TS', 'CTS', 'AA CTS'], fontsize=11, handlelength=3, framealpha=1.0, ncol=2, loc="upper left")
# ax1.tick_params(labelsize=12)
# # Bottom Plot
# ax2.set_xticks(np.arange(0, 12000, 2000))
# ax2.grid(which='both', axis='both', color='k')  # , alpha=0.5
# ax2.set_ylabel('$R(T)$', fontsize=16)
# ax2.set_xlabel('Time ($t$)', fontsize=16)
# ax2.set_xlim(xmin=0.0, xmax=args.horizon)
# ax2.set_ylim(ymin=0.0)
# ax2.grid(True, alpha=0.2)
# ax2.legend(['TS', 'AA TS', 'CTS', 'AA CTS'], fontsize=11, handlelength=3, framealpha=1.0, ncol=2, loc="lower right")
# ax2.tick_params(labelsize=12)
# plt.savefig('results/' + in_name + "/{0}_{1}_complete_plot_step_{2}".format(in_name, dependent, args.STEP) + ".pdf", bbox_inches="tight")
# plt.savefig('results/' + in_name + "/{0}_{1}_complete_plot_step_{2}".format(in_name, dependent, args.STEP) + ".svg", bbox_inches="tight")
# plt.close()
