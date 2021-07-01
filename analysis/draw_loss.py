import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


matplotlib.use("Agg")
plt.style.use(["science", "ieee"])

fig, (ax, ax2) = plt.subplots(2, 1, sharex=True, facecolor="w")

df = pd.read_csv("./analysis/data/large_net.csv")

ax.plot(df["Step"][:], df["Value"][:])
ax2.plot(df["Step"][:], df["Value"][:])
ax.set_ylim(12, 13)
ax2.set_ylim(0, 2)

ax.spines["bottom"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax.xaxis.tick_top()
ax.tick_params(labeltop=False)  # don't put tick labels at the top
ax2.xaxis.tick_bottom()

# this looks pretty good, and was fairly painless, but you can get that
# cut-out diagonal lines look with just a bit more work. the important
# thing to know here is that in axes coordinates, which are always
# between 0-1, spine endpoints are at these locations (0,0), (0,1),
# (1,0), and (1,1).  thus, we just need to put the diagonals in the
# appropriate corners of each of our axes, and so long as we use the
# right transform and disable clipping.

d = 0.02  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax.transAxes, color="k", clip_on=False, linewidth=0.5)
ax.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

# what's cool about this is that now if we vary the distance between
# ax and ax2 via f.subplots_adjust(hspace=...) or plt.subplot_tool(),
# the diagonal lines will move accordingly, and stay right at the tips
# of the spines they are 'breaking'

ax.set_title("large scale network loss")
ax2.set_xlabel("times")
fig.text(0.04, 0.5, "convergence index", va="center", rotation="vertical")

plt.show()
plt.savefig("large-net.eps", format="eps")
