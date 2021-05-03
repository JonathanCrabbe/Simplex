import matplotlib.pyplot as plt


def plot_mnist(data, title=''):
    fig = plt.figure()
    plt.imshow(data, cmap='gray', interpolation='none')
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    return fig
