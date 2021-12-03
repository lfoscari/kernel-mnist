from sklearn.kernel_approximation import Nystroem
from MultilabelPerceptron import *
from torch.utils.data import DataLoader
from MNIST import label_set, train_data, test_data
from tqdm import tqdm
import torch
import time
import json

if __name__ == "__main__":
	train_dataloader = DataLoader(train_data, batch_size=10_000, shuffle=True)
	test_dataloader = DataLoader(test_data, batch_size=500, shuffle=True)

	train_examples = iter(train_dataloader)
	test_examples = iter(test_dataloader)

	x_train, y_train = next(train_examples)
	x_test, y_test = next(test_examples)

	print("Nyström approximation step...", end=" ")
	start = time.time()

	kernel_approx = Nystroem(kernel="polynomial", degree=5, coef0=1, n_components=100)
	kernel_approx = kernel_approx.fit(x_train)

	x_train_ny = torch.tensor(kernel_approx.transform(x_train),  dtype=x_train.dtype)
	x_test_ny = torch.tensor(kernel_approx.transform(x_test),  dtype=x_test.dtype)

	compression_time = time.time() - start
	print("done")

	results = {
		"compression_time": compression_time,
		"width_compression": x_train_ny.shape[1] / x_train.shape[1],
		"epochs_amount": {},
	}

	epochs_iteration = tqdm(range(1, 11))

	for epochs in epochs_iteration:
		epochs_iteration.set_description(f"Training with {epochs} epoch(s)")
		epoch_training_time = time.time()

		NMP = MultilabelPerceptron(
			label_set,
			epochs,
			x_train_ny,
			y_train
		)

		NMP.fit()

		results["epochs_amount"][epochs] = {
			"error": NMP.predict(x_test_ny, y_test),
			"time": time.time() - epoch_training_time
		}

	dest = "./results/nmp.json"
	json.dump(results, open(dest, "w"), indent=True)
	print("Results saved in", dest)