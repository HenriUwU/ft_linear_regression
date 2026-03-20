import argparse

import numpy as np
import pandas as pd
from colorama import Fore, Style

max_epochs = 50000
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


def estimate_price(mileage: np.ndarray, theta0: float, theta1: float) -> np.ndarray:
    """
    This function estimates the price of our vehicle using this given formula :

    estimatePrice(mileage) = theta0 + (theta1 * mileage)

    :param mileage: the given mileage
    :param theta0:
    :param theta1:
    :return: the estimated price
    """
    return theta0 + (theta1 * mileage)


def main(path: str) -> None:
    try:
        df = pd.read_csv(path)

        mileage = df["km"].to_numpy()
        price = df["price"].to_numpy()
        data_count = mileage.shape[0]

        delta_x = max(mileage) - min(mileage)
        delta_y = max(price) - min(price)

        normalized_mileage = normalize_data(mileage, delta_x)
        normalized_price = normalize_data(price, delta_y)

        theta0 = 0.0
        theta1 = 0.0

        for epoch in range(max_epochs):
            estimated_price = estimate_price(normalized_mileage, theta0, theta1)

            tmp_theta0 = learning_rate * (1 / data_count) * np.sum(
                estimated_price - normalized_price
            )
            tmp_theta1 = learning_rate * (1 / data_count) * np.sum(
                (estimated_price - normalized_price) * normalized_mileage
            )

            theta0 = theta0 - tmp_theta0
            theta1 = theta1 - tmp_theta1

        print('final Theta 0 : ', theta0)
        print('final Theta 1 : ', theta1)

        theta1 = theta1 * (delta_y / delta_x)
        theta0 = min(price) + (delta_y * theta0) - (theta1 * min(mileage))

        print('final Theta 0 unnormalized : ', theta0)
        print('final Theta 1 unnormalized : ', theta1)
    except Exception as e:
        print(f"{Fore.RED}An error occurred: {e}{Style.RESET_ALL}")


if __name__ == "__main__":
    # Create the arguments parser
    parser = argparse.ArgumentParser()

    # Mandatory CSV file argument
    parser.add_argument("path", type=str, help="Path to the CSV file (dataset)")

    # Parse the arguments
    args = parser.parse_args()

    # Call main function with given arguments
    main(args.path)
