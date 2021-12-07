from sklearn.kernel_approximation import Nystroem
from MultilabelPerceptron import *
from MNIST import label_set, batch_data_iter
from tqdm import tqdm
import torch
import time
import json

if __name__ == "__main__":
	(x_train, y_train), (x_test, y_test) = batch_data_iter(60_000, 10_000)

	results = {
		"epochs_amount": {},
	}

	epochs_iteration = tqdm(range(1, 11))

	for epochs in epochs_iteration:
		results["epochs_amount"][epochs] = {"degree": {}}

		for degree in range(1, 7):
			epochs_iteration.set_description(f"Training with {epochs} epoch(s) and degree {degree}")

			approximation_time = time.time()

			kernel_approx = Nystroem(kernel="polynomial", degree=degree, coef0=1, n_components=100)
			kernel_approx = kernel_approx.fit(x_train)

			x_train_ny = torch.tensor(kernel_approx.transform(x_train),  dtype=x_train.dtype)
			x_test_ny = torch.tensor(kernel_approx.transform(x_test),  dtype=x_test.dtype)

			approximation_time = time.time() - approximation_time

			epoch_training_time = time.time()

			NMP = MultilabelPerceptron(
				label_set,
				epochs,
				x_train_ny,
				y_train
			)

			NMP.fit()

			epoch_training_time = time.time() - epoch_training_time

			results["epochs_amount"][epochs] = {
				"error": NMP.predict(x_test_ny, y_test),
				"training_time": epoch_training_time,
				"approximation_time": approximation_time
			}

	dest = "./results/nmp.json"
	json.dump(results, open(dest, "w"), indent=True)
	print("Results saved in", dest)