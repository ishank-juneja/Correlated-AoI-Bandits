#!/bin/bash
for instance in instances/*.txt; do
	# Checks if a file like instance
	# actually exists or is just a glob	
	[ -e "$instance" ] || continue
	# Plot arm functions
	echo $"Visualizing instance functions $instance"
	python3 arm_plotter.py -i $instance
	# Slow step so check if output file already exists
	out_name="results/$(basename "$instance" .txt)-out.txt"
	# echo $out_name	
	if [ -f $out_name ]; then
		echo "Results for instance $instance exist, ... skipping simulation"
	else
		echo $"Currently Simulating Policies on $instance"
		python3 simulate_policies.py -i $instance > $out_name
	fi	
	echo $"Plotting Simulation results for $instance"
	python3 simulate_policies.py -i $out_name
done

