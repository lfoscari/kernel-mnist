from interface import Predictor
from dataclasses import dataclass
from typing import Callable
import torch

@dataclass(repr=False)
class MultilabelKernelPerceptron(Predictor):
	kernel: Callable
	labels: list
	epochs: int
	x_train: torch.Tensor
	y_train: torch.Tensor
	model: torch.Tensor = None
	kernel_matrix: torch.Tensor = None
	error: dict = None
	
	def __fit_label(self, label):
		alpha = torch.zeros(self.x_train.shape[0])
		y_train_norm = Predictor.sgn_label(self.y_train, label)

		for _ in range(self.epochs):
			update = False

			for index, (label_norm, kernel_row) in enumerate(zip(y_train_norm, self.kernel_matrix)):
				alpha_update = Predictor.sgn(torch.sum(alpha * y_train_norm * kernel_row, (0))) != label_norm
				update = alpha_update or update
				alpha[index] += alpha_update

			if not update:
				break
			
		return alpha

	def fit(self):
		self.error = {label: {} for label in self.labels}
		self.model = torch.empty((len(self.labels), self.x_train.shape[0]))
		self.kernel_matrix = self.kernel(self.x_train, self.x_train.T)

		for label in self.labels:
			self.model[label] = self.__fit_label(label)

	def predict(self, x_test, y_test):
		test_kernel_matrix = self.kernel(x_test, self.x_train.T)
		scores = [torch.sum(test_kernel_matrix * Predictor.sgn_label(self.y_train, label) * alpha, (1)) for label, alpha in enumerate(self.model)]
		predictions = torch.max(torch.stack(scores), 0)[1]
		return float(torch.sum(predictions != y_test) / y_test.shape[0])
		

if __name__ == "__main__":
	from functools import partial
	from tqdm import tqdm
	import json
	import time
	from torch.utils.data import DataLoader
	from MNIST import label_set, train_data, test_data

	def polynomial(a, b, c = 1., degree = 5.):
		return torch.float_power(a @ b + c, degree)

	train_examples = iter(DataLoader(train_data, batch_size=10_000, shuffle=True))
	test_examples = iter(DataLoader(test_data, batch_size=500, shuffle=True))

	x_train, y_train = next(train_examples)
	x_test, y_test = next(test_examples)

	start = time.time()
	results = {
		"compression_time": 0,
		"epochs_amount": {}
	}

	epochs_iteration = tqdm(range(1, 11))

	for epochs in epochs_iteration:
		results["epochs_amount"][epochs] = {"degree": {}}

		for degree in range(1, 7):
			epochs_iteration.set_description(f"Training with {epochs} epoch(s) and degree {degree}")
			epoch_training_time = time.time()

			MKP = MultilabelKernelPerceptron(
				partial(polynomial, degree=degree),
				label_set,
				epochs,
				x_train,
				y_train,
			)

			MKP.fit()
			results["epochs_amount"][epochs]["degree"][degree] = {
				"error": MKP.predict(x_test, y_test),
				"time": time.time() - epoch_training_time
			}

	results["training_time"] = time.time() - start

	dest = "./results/mkp.json"
	json.dump(results, open(dest, "w"), indent=True)
	print("Results saved in", dest)