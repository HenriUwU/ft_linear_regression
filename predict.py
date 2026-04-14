import numpy as np
from colorama import Fore, Style

from train import estimate_price


def main() -> None:
    try:
        thetas = np.load("thetas.npz")

        theta0 = thetas['theta0']
        theta1 = thetas['theta1']

        mileage = input(f"{Fore.LIGHTBLUE_EX}Your vehicle's Mileage : ")

        estimated_price = estimate_price(int(mileage), theta0, theta1)

        if estimated_price > 0:
            print(f"{Fore.GREEN}Estimated Price :"
                  f"{estimated_price}{Style.RESET_ALL}")
        else:
            print(f"{Fore.GREEN}Unfortunately it seems like"
                  f"you drove way too far,"
                  f"the estimated price is :"
                  f"{Fore.RED} {estimated_price} {Fore.GREEN},"
                  f"but realistically your vehicle"
                  f"actually is worth nothing. {Style.RESET_ALL}")

    except Exception as e:
        print(f"{Fore.RED}An error occurred: {e}{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
