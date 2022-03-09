import numpy as np
import matplotlib.pyplot as plt


def plot(model, x: np.array, y: np.array, data_labels: list, title: str, x_label: str, y_label: str = 'ln(lag-time)'):
    """
    Generate and display 2D plot of the given data points, as well as the line of the model-predicted values.
    :param model: Trained model, expected to be that of linear regression.
    :param x: Independent variables.
    :param y: Dependent variables.
    :param data_labels: Labels for each data point, expected to be synuclein names.
    :param title: Title of plot.
    :param x_label: Label of independent variables.
    :param y_label: Label of dependent variables.
    """
    plt.scatter(x, y, color='g')
    plt.plot(x, model.predict(x), color='k')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    for (dl, x, y) in zip(data_labels, x, y):
        plt.annotate(dl, (x, y + 0.1), size=6)
    plt.show()

