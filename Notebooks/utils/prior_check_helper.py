import matplotlib.pyplot as plt
import seaborn as sns
import os

class HistogramPlot:
    def __init__(self, data, title, xlabel, color, bin_range=None, bins=None):
        self.data = data
        self.title = title
        self.xlabel = xlabel
        self.color = color
        self.bin_range = bin_range
        self.bins = bins

    def plot(self, ax=None):
        # Accept an existing Axes object to plot on
        if ax is None:
            ax = plt.gca()  # Get the current Axes to plot on if none provided
        kwargs = {'color': self.color}
        if self.bins is not None:
            kwargs['bins'] = self.bins
        if self.bin_range is not None:
            kwargs['binrange'] = self.bin_range
        sns.histplot(self.data, ax=ax, linewidth=0, **kwargs)  # Use the provided or current Axes
        ax.set_title(self.title)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel('Frequency')

        


def plot_histograms(plots: list[HistogramPlot], nrows=5, ncols=2, figsize=(10, 20), save_img=False, prefix_name="",output_folder='priors/'):
    """
    Plot a list of histograms based on the provided HistogramPlot objects.
    
    Parameters:
    plots (list of HistogramPlot): List of HistogramPlot objects to plot.
    nrows (int): Number of rows in the figure grid.
    ncols (int): Number of columns in the figure grid.
    figsize (tuple): Figure dimension.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten()
    
    for i, (plot, ax) in enumerate(zip(plots, axes)):
        plot.plot(ax)

        # Optionally save each plot as a separate PDF
        if save_img:
            fig_ind, ax_ind = plt.subplots(figsize=(6,4))
            plot.plot(ax_ind)
            ax_ind.set_title("")
            fig_ind.tight_layout()
            fig_ind.savefig(f"{output_folder}/{prefix_name}_{plot.xlabel}.pdf")
            plt.close(fig_ind)  # Close the individual figure to free up memory

    plt.tight_layout()
    plt.show()
