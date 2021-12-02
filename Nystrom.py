from sklearn.kernel_approximation import Nystroem
from MultilabelPerceptron import *
from torch.utils.data import DataLoader
from MNIST import label_set, train_data, test_data
from tqdm import tqdm
import torch
import time
import json

# TODO: fix accuracy, clearly something is wrong

def polynomial(a, b, c = 1., degree = 5.):
    return (a @ b + c) ** degree

if __name__ == "__main__":
	train_dataloader = DataLoader(train_data, batch_size=10_000, shuffle=True)
	test_dataloader = DataLoader(test_data, batch_size=500, shuffle=True)

	train_examples = iter(train_dataloader)
	test_examples = iter(test_dataloader)

	x_train, y_train = next(train_examples)
	x_test, y_test = next(test_examples)

	print("Nyström approximation step...")
	start = time.time()

	kernel_approx = Nystroem(polynomial, n_components=500)
	x_train_ny = torch.tensor(kernel_approx.fit_transform(x_train), dtype=x_train.dtype)
	x_test_ny = torch.tensor(kernel_approx.fit_transform(x_test),  dtype=x_test.dtype)

	compression_time = time.time() - start

	start = time.time()
	results = {
		"compression_time": compression_time,
		"width_compression": x_train_ny.shape[1] / x_train.shape[1],
		"epochs_amount": {}
	}

	epochs_iteration = tqdm(range(1, 11))

	for epochs in epochs_iteration:
		epochs_iteration.set_description(f"Training with {epochs} epoch(s)")

		NMP = MultilabelPerceptron(
			label_set,
			epochs,
			x_train_ny,
			y_train
		)

		NMP.fit()
		results["epochs_amount"][epochs] = NMP.predict(x_test_ny, y_test)

	results["training_time"] = time.time() - start

	dest = "./results/nmp.json"
	json.dump(results, open(dest, "w"), indent=True)
	print("Results saved in", dest)