import re
from numpy import degrees
from sklearn.kernel_approximation import Nystroem
from MultilabelPerceptron import *
from MNIST import label_set, batch_data_iter
from tqdm import tqdm
import torch
import time
import json

if __name__ == "__main__":
	(x_train, y_train), (x_test, y_test) = batch_data_iter(10_000, 500)

	results = {
		"degree": {},
	}

	progress = tqdm(range(1, 7))

	for degree in progress:
		results["degree"][degree] = {"epochs": {}}

		approximation_time = time.time()

		kernel_approx = Nystroem(kernel="polynomial", degree=degree, coef0=1, n_components=100)
		kernel_approx = kernel_approx.fit(x_train)

		x_train_ny = torch.tensor(kernel_approx.transform(x_train),  dtype=x_train.dtype)
		x_test_ny = torch.tensor(kernel_approx.transform(x_test),  dtype=x_test.dtype)

		results["degree"][degree]["approximation_time"] = time.time() - approximation_time

		for epochs in range(1, 11):
			progress.set_description(f"Training with {epochs} epoch(s) and degree {degree}")

			epoch_training_time = time.time()

			NMP = MultilabelPerceptron(
				label_set,
				epochs,
				x_train_ny,
				y_train
			)

			NMP.fit()

			epoch_training_time = time.time() - epoch_training_time

			results["degree"][degree]["epochs"][epochs] = {
				"error": NMP.predict(x_test_ny, y_test),
				"training_time": epoch_training_time,
			}

	dest = "./results/nmp.json"
	json.dump(results, open(dest, "w"), indent=True)
	print("Results saved in", dest)