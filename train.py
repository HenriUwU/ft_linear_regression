import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from colorama import Fore, Style

max_epochs = 150000000000
learning_rate = 0.01


def normalize_data(array: np.ndarray, delta: float) -> np.ndarray:
    """
    This function takes an n-dimensional numpy array as an input
    and normalizes it using the min-max normalization formula :

    x_norm = (x - x_min) / (x_max - x_min)

    (x_max - x_min) = delta

    This ensures our values are between 0 and 1,
    1 being the max value and 0 being the min value
    of the original array

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
    This functions computes the RMSE (root-mean-square error)

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
    This functions computes the loss of our model using the loss function.

    :param estimated_price: the estimated price made by the model (normalized)
    :param normalized_price: the original price (normalized)
    :return: the loss of the model
    """
    m = estimated_price.shape[0]

    return ((1 / (2 * m))
            * (np.sum((estimated_price - normalized_price) ** 2)))


def main(path: str, line, learning) -> None:
    try:
        df = pd.read_csv(path)

        mileage = df["km"].to_numpy()
        price = df["price"].to_numpy()
        data_count = mileage.shape[0]

        delta_x = float(np.max(mileage)) - float(np.min(mileage))
        delta_y = float(np.max(price)) - float(np.min(price))

        normalized_mileage = normalize_data(mileage, delta_x)
        normalized_price = normalize_data(price, delta_y)

        theta0 = 0.0
        theta1 = 0.0
        tolerance = 1e-5
        cost_history = []

        for epoch in range(max_epochs):
            # Estimate the price using current theta values
            estimated_price = estimate_price(normalized_mileage,
                                             theta0,
                                             theta1)

            # Save the loss in the history
            cost_history.append(
                compute_loss(estimated_price, normalized_price)
            )

            # Compute tmp_theta0 using the given formula
            tmp_theta0 = learning_rate * (1 / data_count) * np.sum(
                estimated_price - normalized_price
            )
            # Compute tmp_theta1 using the given formula
            tmp_theta1 = learning_rate * (1 / data_count) * np.sum(
                (estimated_price - normalized_price) * normalized_mileage
            )

            # Perform gradient descent
            new_theta0 = theta0 - tmp_theta0
            new_theta1 = theta1 - tmp_theta1

            # Verify if we learned enough, if not -> early stop
            if (epoch > 0
                    and np.abs(theta0 - new_theta0) < tolerance
                    and np.abs(theta1 - new_theta1) < tolerance):
                print(f"{Fore.CYAN}"
                      f"Early stop at epoch {epoch}"
                      f"{Style.RESET_ALL}")
                break

            # Update both theta values
            theta0 = new_theta0
            theta1 = new_theta1

        # Unnormalize the thetas before saving them
        theta1 = theta1 * (delta_y / delta_x)
        theta0 = (float(np.min(price))
                  + (delta_y * theta0)
                  - (theta1 * float(np.min(mileage))))

        # Save the thetas
        np.savez("thetas.npz", theta0=theta0, theta1=theta1)

        # Plot the line of our model
        if line:
            plot_line(df, theta0, theta1)

        # Plot the learning curve of our model
        if learning:
            plot_learning_curve(cost_history)

        # Compute the RMSE for precision
        print(f"{Fore.YELLOW}RMSE :"
              f"{compute_rmse(df, theta0, theta1)}{Style.RESET_ALL}")

    except Exception as e:
        print(f"{Fore.RED}An error occurred: {e}{Style.RESET_ALL}")


if __name__ == "__main__":
    # Create the arguments parser
    parser = argparse.ArgumentParser()

    # Mandatory CSV file argument
    parser.add_argument("path",
                        type=str,
                        help="Path to the CSV file (dataset)")

    # Optional "plot" argument to display the plot
    parser.add_argument("--line",
                        action="store_true",
                        help="Display the line of the model")

    # Optional "learning" argument to display a plot of the learning curve
    parser.add_argument("--learning",
                        action="store_true",
                        help="Display the learning curve of the model")

    # Parse the arguments
    args = parser.parse_args()

    # Call main function with given arguments
    main(args.path, args.line, args.learning)
