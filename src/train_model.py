import os
import csv
import numpy as np

def load_dataset(dataset_path):
	mileage = []
	price = []
	try:
		with open(dataset_path, 'r') as file:
			reader = csv.DictReader(file)
			for row in reader:
				mileage.append(float(row['km']))
				price.append(float(row['price']))
	except Exception as e:
		print(f"Error while loading dataset: {e}")
		exit(1)
	return np.array(mileage), np.array(price)

def estimate_price(mileage, theta0, theta1):
	return theta0 + mileage * theta1

def gradient_descent(mileage, price, learning_rate, num_iterations):
	m = len(mileage)
	theta0 = 0
	theta1 = 0

	for i in range(num_iterations):
		error = estimate_price(mileage, theta0, theta1) - price

		tmp_theta0 = theta0 - learning_rate * (1 / m) * np.sum(error)
		tmp_theta1 = theta1 - learning_rate * (1 / m) * np.sum(error * mileage)

		theta0 = tmp_theta0
		theta1 = tmp_theta1

		if i % 100 == 0:
			cost = (1 / (2 * m)) * np.sum(error ** 2)
			print(f"Iteration {i}: Cost = {cost:.4f}, theta0 = {theta0:.4f}, theta1 = {theta1:.4f}")
	
	return theta0, theta1

def save_model(theta0, theta1, file_path):
	try:
		with open(file_path, 'w') as file:
			file.write(f"{theta0}\n{theta1}")
		print("Model saved to {file_path}")
	except Exception as e:
		print(f"Error while saving model: {e}")

def standardize(data):
	return (data - np.mean(data)) / np.std(data)

def main():
	dataset_path = "./data/data.csv"
	mileage, price = load_dataset(dataset_path)
	mileage = standardize(mileage)
	learning_rate = 0.0001
	num_iterations = 10000

	print("Training model...")
	theta0, theta1 = gradient_descent(mileage, price, learning_rate, num_iterations)

	save_model(theta0, theta1, "./saved_model/model.txt")
	print(f"Training completed: theta0 = {theta0:.4f}, theta1 = {theta1:.4f}")

if __name__ == "__main__":
	main()