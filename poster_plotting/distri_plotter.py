from matplotlib import pyplot as plt
import numpy as np

in_file = 'uni_better_arms.txt'
fin = open(in_file, 'r')
y1 = [float(x) for x in fin.readline().split('\t')]
y2 = [float(x) for x in fin.readline().split('\t')]
plt.figure(figsize=(3, 3*3/4))
my_xticks = ['$x_1$','$x_2$','$x_3$','$x_4$', '$x_5$']
x = np.arange(1, 6)
plt.xticks(x, my_xticks)
plt.plot(x, y1, linewidth=2.5)
plt.plot(x, y2, linewidth=2.5)
plt.scatter(x, y1, s=40)
plt.scatter(x, y2, s=40)
plt.xlabel("Support Points", fontweight="bold")
plt.ylabel("Reward Obtained", fontweight="bold")
plt.title("Reward Functions", fontweight="bold")
plt.legend(["$g_1(x)$", "$g_2(x)$"])
plt.grid(which='both', axis='both', color='k', alpha = 0.1)
plt.ylim(bottom=0, top=3.1)
#plt.show()
plt.savefig(in_file + "_plot" + ".svg", bbox_inches="tight")
plt.close()

