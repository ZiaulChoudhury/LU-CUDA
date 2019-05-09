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

LU =  [5, 3, 2,1,1,0.9,0.8,0.8]
LU2 = [4.7, 2.5, 1.5,0.8,0.7,0.5,0.6,0.8]
fig, ax = plt.subplots()
bars = ('1','2','4','8','16','32','64','128')
plt.subplot(1, 3, 1)
y_pos1 = [1,2,3,4,5,6,7,8]
plt.xlabel("Blocks")
plt.xticks(y_pos1,bars)
ax = plt.gca()
plt.title("Scalability")
plt.ylim(0,6)
plt.plot(y_pos1, LU, marker='o', markerfacecolor='blue', markersize=8, color='blue', linewidth=1, label=" Multi-Kernel Approach")
plt.plot(y_pos1, LU2, marker='d', markerfacecolor='red', markersize=8,  color='red', linewidth=1, label="Single-kernel Approach")
plt.ylabel('Time in ms')
plt.legend()
#plt.tight_layout()


MatrixSize = [64,256,1024,4096,16384]
LUt =  [0.06, 0.1, 0.2,1,0.4,0.9]
LU2t = [0.24,0.26,0.32,0.44,0.57]

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
plt.plot(y_pos1, LU, marker='o', markerfacecolor='blue', markersize=8, color='blue', linewidth=2, label=" Multi-Kernel Approach")
plt.plot(y_pos1, LU2, marker='d', markerfacecolor='red', markersize=8,  color='red', linewidth=2, label="Single-kernel Approach")
plt.ylabel('Throghput in GOPS/sec log2 scale')
plt.legend()
#plt.tight_layout()


LU2 = [27,14,15,20,25]
bars = ('2','20','80','100','200')
plt.subplot(1, 3, 3)
y_pos1 = [1,2,3,4,5]
plt.xlabel("Blocks")
plt.xticks(y_pos1,bars)
ax = plt.gca()
plt.title("Scalability MatSize(1M)")
plt.ylim(0,50)
plt.plot(y_pos1, LU2, marker='d', markerfacecolor='red', markersize=8,  color='red', linewidth=2, label="Single-kernel Approach")
plt.ylabel('Time in ms')
plt.legend()





#plt.tight_layout()
#plt.subplots_adjust(wspace=0, hspace=5)
handles, labels = ax.get_legend_handles_labels()
#fig.legend(handles, labels, loc='lower center',ncol=2,bbox_to_anchor=(0.5, 0))
fig.set_size_inches(12, 4)
fig.savefig("piplot.eps")
fig.savefig("piplot.png")