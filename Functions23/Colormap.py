# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
def plot_examples(cms):
    """
    helper function to plot two colormaps
    """
    np.random.seed(19680801)
    data = np.random.randn(30, 30)

    fig, axs = plt.subplots(1, 2, figsize=(6, 3), constrained_layout=True)
    for [ax, cmap] in zip(axs, cms):
        psm = ax.pcolormesh(data, cmap=cmap, rasterized=True, vmin=-4, vmax=4)
        fig.colorbar(psm, ax=ax)
    plt.show()
    
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
# viridis = cm.get_cmap('viridis', 256)

# top = cm.get_cmap('Oranges_r', 128)
# bottom = cm.get_cmap('Blues', 128)


# newcolors = np.vstack((top(np.linspace(0, 1, 128)),
#                        bottom(np.linspace(0, 1, 128))))
# pink = np.array([248/256, 24/256, 148/256, 1])
# newcolors[:25, :] = pink
# newcmp = ListedColormap(newcolors, name='OrangeBluePink')
# plot_examples([viridis, newcmp])

colors = ["white", "peachpuff", "sandybrown", "chocolate"]
cmap1 = LinearSegmentedColormap.from_list("mycmap", colors)

nodes = [0.0, 0.2, 0.8,  1.0]
cmap2 = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colors)))

plot_examples([cmap1, cmap2])

#mpl.rc('image', cmap=cmap2)
