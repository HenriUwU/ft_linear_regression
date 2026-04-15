import argparse

import numpy as np
import pandas as pd
from colorama import Fore, Style

from utils import (normalize_data,
                   estimate_price,
                   compute_loss,
                   plot_line,
                   plot_learning_curve,
                   compute_rmse)

max_epochs = 9999999
learning_rate = 0.01


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
