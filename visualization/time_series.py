import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_time_series(x: np.ndarray, title=None):
    sns.set(font_scale=1.5)
    sns.set_style("white")
    t = np.arange(start=0, stop=x.shape[0])
    plt.plot(t, x, linestyle='-', marker='o')
    plt.title(title)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$x_t$')
    plt.show()
