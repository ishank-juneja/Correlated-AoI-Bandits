from matplotlib import pyplot as plt
import numpy as np

in_file = 'mid_range.txt'
fin = open(in_file, 'r')
y1 = [float(x) for x in fin.readline().split('\t')]
plt.figure(figsize=(2, 1.5))
my_xticks = ['$x_1$','$x_2$','$x_3$','$x_4$', '$x_5$']
x = np.arange(1, 6)
plt.xticks(x, my_xticks)
plt.plot(x, y1, linewidth=2.5, color='red')
plt.scatter(x, y1, s=40, color='red')
plt.xlabel("Support Points", fontweight='bold')
plt.ylabel("Purchase Prob*14", fontweight='bold')
plt.title("$g_2(x)$", fontweight='bold')
plt.grid(which='both', axis='both', color='k', alpha = 0.1)
plt.ylim(bottom=0, top=5)
plt.savefig(in_file + "_plot" + ".svg", bbox_inches="tight")
plt.close()

