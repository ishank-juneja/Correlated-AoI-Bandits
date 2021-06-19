import pandas as pd
from matplotlib import pyplot as plt
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
x_points = np.arange(args.STEP, int(args.horizon) + 1, args.STEP)
y_points = np.zeros_like(x_points)
# These are all the possible labels, of these atmost 8 can be supported with below colors
# LABELS = ['ucb', 'ts', 'qucb', 'qts', 'cucb', 'cts', 'u-cucb', 'new', 'cts-old', 'cucb-old']
# selected must be a subset of algos selected in simulate_policies.py
# All the algorithms selected must have already been simulated in simulate_policies.py
selected = ['ucb', 'ts']
# Number of distinct algorithms used
nalgos = len(selected)
# Number of colors should be at least as many as number of LABELS
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
# Choose as many colors as there are algorithms selected
if len(COLORS) < len(selected):
	print("Error: Not enough colors present for plotting\nAdd colors or reduce number of algorithms")
	exit(-1)
else:
	COLORS = COLORS[:len(selected)]
# placeholder for the aoi versions of these algos
aoi_labels = []
for label in selected:
	aoi_labels.append("aoi-aware-" + label)
# Merge the 2 lists into single complete list of policies
selected = selected + aoi_labels
bandit_data = pd.read_csv(in_file, sep=",", header=None)
bandit_data.columns = ["algo", "rs", "eps", "horizon", "reg", "exp_reg", "AoI_reg", "exp_AoI_reg"]
# List of dependent variables to be plotted, from above list
dependents = bandit_data.columns[4:]


# Plot and average the results for each label onto a single plot,
# doesn't make a lot of sense, just there
for dependent in dependents:
	plt.figure(figsize=(10, 10))
	for index, label in enumerate(selected):
		cur_data = bandit_data.loc[bandit_data["algo"] == label]
		# print(label)
		# Get data points for each algorithm
		for i in range(len(y_points)):
			y_points[i] = cur_data.loc[cur_data["horizon"] == x_points[i]][dependent].mean()
		# print(y_points)
		if label[:9] == 'aoi-aware':
			plt.plot(x_points, y_points, color=COLORS[index % nalgos], linewidth=3, linestyle='--')
			# plt.plot(np.log10(x_points), y_points, color='b')
		else:
			plt.plot(x_points, y_points, color=COLORS[index], linewidth=3)

	plt.legend(selected)
	# f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
	# g = lambda x,pos : "${}$".format(f._formatSciNotation('%1.10e' % x))
	# plt.gca().xaxis.set_major_formatter(mticker.FuncFormatter(g))
	# plt.xticks(np.arange(0, max(x_points)+1, 50000))
	plt.xlabel("Number of rounds T", fontweight="bold")
	plt.ylabel("Cumulative Regret", fontweight="bold")
	plt.title("AoI Policy Comparisons", fontweight="bold")
	plt.yticks()
	plt.savefig('results/' + in_name + "/{0}_{1}_complete_plot".format(in_name, dependent) + ".png", bbox_inches="tight")
	plt.close()

for dependent in dependents:
	plt.figure(figsize=(10, 10))
	for index, label in enumerate(selected):
		cur_data = bandit_data.loc[bandit_data["algo"] == label]
		# print(cur_data)
		# print(label)
		# Get data points for each algorithm
		for i in range(len(y_points)):
			y_points[i] = cur_data.loc[cur_data["horizon"] == x_points[i]][dependent].mean()
		# print(y_points)
		if label[:9] == 'aoi-aware':
			plt.plot(x_points, y_points, color=COLORS[index % nalgos], linewidth=3, linestyle='--')
			# plt.plot(np.log10(x_points), y_points, color='b')
		else:
			plt.plot(x_points, y_points, color=COLORS[index], linewidth=3)

	plt.legend(selected)
	# f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
	# g = lambda x,pos : "${}$".format(f._formatSciNotation('%1.10e' % x))
	# plt.gca().xaxis.set_major_formatter(mticker.FuncFormatter(g))
	# plt.xticks(np.arange(0, max(x_points)+1, 50000))
	plt.xlabel("Number of rounds T", fontweight="bold")
	plt.ylabel("Cumulative Regret", fontweight="bold")
	plt.title("AoI Policy Comparisons", fontweight="bold")
	plt.yticks()
	plt.savefig('results/' + in_name + "/{0}_{1}_plot".format(in_name, dependent) + ".png", bbox_inches="tight")
	plt.close()