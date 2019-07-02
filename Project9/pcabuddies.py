import numpy as np
from sklearn import decomposition
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import sys
import csv
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import data
import matplotlib.pyplot as plt
import mplcursors

dobj = data.Data("premierleague.csv")
headers = dobj.get_headers()[1:]
position = dobj.limit_columns(['position']).T
d = dobj.limit_columns(headers).T

pearsons = []
spearmans = []
for a in d:
	pearson, _ = pearsonr(np.squeeze(np.asarray(position)),np.squeeze(np.asarray(a)))
	pearsons.append(pearson)
	spearman, _ = spearmanr(np.squeeze(np.asarray(position)),np.squeeze(np.asarray(a)))
	spearmans.append(spearman)

print('pearsons\n',pearsons)
print('\n\nspearmans\n', spearmans)


#https://stackoverflow.com/questions/50560525/how-to-annotate-the-values-of-x-and-y-while-hovering-mouse-over-the-bar-graph
fig = plt.figure()
ax = plt.subplot()
bars = plt.bar([i for i in range(len(headers))], spearmans)
annot = ax.annotate("", xy=(0,0), xytext=(-20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="black", ec="b", lw=2),
                    arrowprops=dict(arrowstyle="->"))

def update_annot(bar):
    x = bar.get_x()+bar.get_width()/2.
    y = bar.get_y()+bar.get_height()
    annot.xy = (x,y)
    text = headers[int(x)]
    annot.set_text(text)
    annot.get_bbox_patch().set_alpha(0.4)


def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        for bar in bars:
            cont, ind = bar.contains(event)
            if cont:
                update_annot(bar)
                annot.set_visible(True)
                fig.canvas.draw_idle()
                return
    if vis:
        annot.set_visible(False)
        fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", hover)

plt.show()	

# if __name__ == "__main__":
	# main(sys.argv)