import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def normalize_data(array: np.ndarray, delta: float) -> np.ndarray:
    """
    This function takes an n-dimensional numpy array as an input
    and normalizes it using the min-max normalization formula :

    x_norm = (x - x_min) / (x_max - x_min)

    (x_max - x_min) = delta

    This ensures our values are between 0 and 1,
    1 being the max value and 0 being the min value
    of the original array.

    :param array: the array we want to normalize
    :param delta: the difference between max and min
    :return: the normalized version of the array
    """
    return (array - min(array)) / delta


def estimate_price(mileage, theta0: float, theta1: float) -> np.ndarray:
    """
    This function estimates the price of our vehicle using this given formula :

    estimatePrice(mileage) = theta0 + (theta1 * mileage)

    :param mileage: the given mileage
    :param theta0: our theta 0
    :param theta1: our theta 1
    :return: the estimated price for a given mileage
    """
    return theta0 + (theta1 * mileage)


def plot_line(dataFrame: pd.DataFrame, theta0, theta1) -> None:
    """
    This function plots the price of the car depending on the mileage.

    In addition to that, we plot a line representing our model,
    using the estimated price after the training is done.

    :param dataFrame: the dataFrame
    :param theta0: the unnormalized theta0
    :param theta1: the unnormalized theta1
    :return: Nothing
    """
    mileage = dataFrame["km"]
    price = dataFrame["price"]

    plt.scatter(x=mileage, y=price, color="blue")

    plt.xlabel("Mileage")
    plt.ylabel("Price",
               rotation=0,
               labelpad=15,
               ha='right',
               va='center')

    plt.plot(mileage,
             estimate_price(mileage, theta0, theta1),
             color="red")

    plt.show()


def plot_learning_curve(cost_history) -> None:
    """
    This function plots the learning curve of our model.
    It represents the loss over the epochs.

    :param cost_history: the history of loss over all epochs during training
    :return: Nothing
    """
    plt.figure()
    plt.plot(cost_history, color="blue")

    plt.xlabel("Epochs")
    plt.ylabel("Loss",
               rotation=0,
               labelpad=15,
               ha='right',
               va='center')

    plt.show()


def compute_rmse(dataFrame, theta0, theta1):
    """
    This function computes the RMSE (root-mean-square error)

    :param dataFrame: the original dataFrame
    :param theta0: the final theta 0
    :param theta1: the final theta 1
    :return: the computed root-mean-square error
    """
    mileage = dataFrame["km"].to_numpy()
    price = dataFrame["price"].to_numpy()
    data_count = mileage.shape[0]

    estimated_price = estimate_price(mileage, theta0, theta1)

    return np.sqrt(np.sum((estimated_price - price) ** 2 / data_count))


def compute_loss(estimated_price: np.ndarray,
                 normalized_price: np.ndarray) -> float:
    """
    This function computes the loss of our model using the loss function.

    :param estimated_price: the estimated price made by the model (normalized)
    :param normalized_price: the original price (normalized)
    :return: the loss of the model
    """
    m = estimated_price.shape[0]

    return ((1 / (2 * m))
            * (np.sum((estimated_price - normalized_price) ** 2)))


def compute_coefficient_of_determination(dataFrame: pd.DataFrame,
                                         theta0, theta1) -> float:
    """
    This function computes the coefficient of determination of the model.

    :param dataFrame: the original dataFrame
    :param theta0: the unnormalized theta 0
    :param theta1: the unnormalized theta 1
    :return: the coefficient R²
    """

    mileage = dataFrame["km"].to_numpy()
    price = dataFrame["price"].to_numpy()

    estimated_price = estimate_price(mileage, theta0, theta1)

    # Compute the error of my model
    RSS = np.sum((estimated_price - price) ** 2)

    # Compute the error of an "idiot" model
    # (a model using the mean price to make its predictions)
    TSS = np.sum((np.mean(price) - price) ** 2)

    # Compute and return the coefficient R²
    return 1 - (float(RSS) / float(TSS))
