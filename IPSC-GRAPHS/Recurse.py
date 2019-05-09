import math
import numpy as np
import matplotlib.pyplot as plt

def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.02*height,
                '%d' % int(height),
                ha='center', va='bottom')


cx = (227/255, 222/255, 229/255)
framework = {}

RH =  [5, 2, 0.8,0.5, 0.2,0.3,0.2]
RH2 = [4, 1.5, 0.5,0.2,0.1,0.1,0.1]
fig, ax = plt.subplots()
bars = ('1','2','4','8','16','32','64')
plt.subplot(1, 3, 1)
y_pos1 = [1,2,3,4,5,6,7]
plt.xlabel("Threads")
plt.xticks(y_pos1,bars)
ax = plt.gca()
plt.title("Scalability")
plt.ylim(0,6)
plt.plot(y_pos1, RH, marker='x', markerfacecolor='orange', markersize=8, color='orange', linewidth=2, label="Recursive")
plt.plot(y_pos1, RH2, marker='o', markerfacecolor='green', markersize=8,  color='green', linewidth=2, label="Recursive Hybrid")
plt.ylabel('Time in ms')
plt.legend()
plt.tight_layout()


MatrixSize = [64,256,1024,4096,16384]
LUt =  [0.06, 0.05, 0.03,0.02,0.01,0.02]
LU2t = [0.1,0.08,0.07,0.06,0.06]

LU = []
LU2 = []

for i in range(0,5):
    LU.append(math.log((((math.pow(MatrixSize[i],3)/LUt[i])*1000)/1000000000),2))
    LU2.append(math.log((((math.pow(MatrixSize[i], 3) / LU2t[i]) * 1000) / 1000000000),2))

bars = ('64','256','1024','4096','16384')
plt.subplot(1, 3, 2)
y_pos1 = [1,2,3,4,5]
plt.xlabel("Mat Size")
plt.xticks(y_pos1,bars)
ax = plt.gca()
plt.title("Throughput")
plt.ylim(0,50)
plt.plot(y_pos1, LU, marker='o', markerfacecolor='green', markersize=8, color='green', linewidth=2, label="Recursive Hybrid")
plt.plot(y_pos1, LU2, marker='x', markerfacecolor='orange', markersize=8,  color='orange', linewidth=2, label="Recursive")
plt.ylabel('Throghput in GOPS/sec log2 scale')
plt.legend()
#plt.tight_layout()



RH2 = [6.8,5,4,2,3,4]
bars = ('2','8','16','32','64','128')
plt.subplot(1, 3, 3)
y_pos1 = [1,2,3,4,5,6]
plt.xlabel("Block Size b")
plt.xticks(y_pos1,bars)
ax = plt.gca()
plt.title("Scalability Matrix = 512x512")
plt.ylim(0,10)
plt.plot(y_pos1, RH2, marker='o', markerfacecolor='green', markersize=8,  color='green', linewidth=2, label="Recursive Hybrid")
plt.ylabel('Time in ms')
plt.legend()
#plt.tight_layout()


plt.tight_layout()
#plt.subplots_adjust(wspace=0, hspace=5)
handles, labels = ax.get_legend_handles_labels()
#fig.legend(handles, labels, loc='lower center',ncol=2,bbox_to_anchor=(0.5, 0))
fig.set_size_inches(12, 4)
fig.savefig("recurse.eps")
fig.savefig("recurse.png")