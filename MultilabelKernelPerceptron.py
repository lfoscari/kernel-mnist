from interface import Predictor
from dataclasses import dataclass
from typing import Callable
from functools import partial
import json
import torch

@dataclass(repr=False)
class MultilabelKernelPerceptron(Predictor):
	kernel: Callable
	labels: list
	epochs: int
	x_train: torch.Tensor
	y_train: torch.Tensor
	x_test: torch.Tensor
	y_test: torch.Tensor
	model: torch.Tensor = None
	kernel_matrix: torch.Tensor = None
	error: dict = None
	save_file: str = None
	
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
			print("Training label", label)
			self.model[label] = self.__fit_label(label)

	def predict(self):
		test_kernel_matrix = self.kernel(self.x_test, self.x_train.T)
		scores = [torch.sum(test_kernel_matrix * Predictor.sgn_label(self.y_train, label) * alpha, (1)) for label, alpha in enumerate(self.model)]
		predictions = torch.max(torch.stack(scores), 0)[1]
		return float(torch.sum(predictions != self.y_test) / self.y_test.shape[0])
		

if __name__ == "__main__":
	from torch.utils.data import DataLoader
	from MNIST import label_set, train_data, test_data

	def polynomial(a, b, c = 1., degree = 5.):
		return torch.float_power(a @ b + c, degree)

	train_dataloader = DataLoader(train_data, batch_size=1_000, shuffle=True)
	test_dataloader = DataLoader(test_data, batch_size=100, shuffle=True)

	train_examples = iter(train_dataloader)
	test_examples = iter(test_dataloader)

	x_train, y_train = next(train_examples)
	x_test, y_test = next(test_examples)

	results = {}

	for epochs in range(10):
		print("-" * 10, "Training", epochs, "epochs")
		results[epochs] = {}

		for degree in range(6):
			print("-" * 5, "Training degree", degree+1)

			MKP = MultilabelKernelPerceptron(
				partial(polynomial, degree=degree+1),
				set(label_set),
				epochs+1,
				x_train,
				y_train,
				x_test,
				y_test,
				"./results/mkp.json"
			)

			MKP.fit()
			results[epochs+1][degree+1] = MKP.predict()
			print("[", "Error:", results[epochs+1][degree+1], "]")

	dest = "./results/mkp.json"
	json.dump(results, open(dest, "w"), indent=True)
	print("Results saved in", dest)