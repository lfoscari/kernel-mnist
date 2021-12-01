from interface import Predictor
from dataclasses import dataclass
from typing import Callable
import torch

@dataclass(repr=False)
class MultilabelKernelPerceptron(Predictor):
	kernel: Callable
	labels: set
	epochs: int
	x_train: torch.Tensor
	y_train: torch.Tensor
	x_test: torch.Tensor
	y_test: torch.Tensor
	model: torch.Tensor = None
	kernel_matrix: torch.Tensor = None

	def epoch_error(self, alpha, y_train_norm, y_test_norm):
		score = torch.sum(self.kernel(self.x_test, self.x_train.T) * y_train_norm * alpha, (1))
		return torch.sum(Predictor.sgn(score) != y_test_norm) / y_test_norm.shape[0]

	def fit_label(self, label):
		alpha = torch.zeros(self.x_train.shape[0])

		y_train_norm = Predictor.sgn_label(self.y_train, label)
		y_test_norm = Predictor.sgn_label(self.y_test, label)

		for e in range(self.epochs):
			update = False

			for index, (label, kernel_row) in enumerate(zip(y_train_norm, self.kernel_matrix)):
				alpha_update = Predictor.sgn(torch.sum(alpha * y_train_norm * kernel_row, (0))) != label
				update = alpha_update or update
				alpha[index] += alpha_update

			error = self.epoch_error(alpha, y_train_norm, y_test_norm)
			print("Epoch", e, "error", float(error))

			if not update:
				print("Skipping remaining epochs")
				break
			
		return alpha

	def fit(self):
		self.model = torch.empty((len(self.labels), self.x_train.shape[0]))
		self.kernel_matrix = self.kernel(self.x_train, self.x_train.T)

		for label in self.labels:
			print("-" * 5, "Training label", label)
			self.model[label] = self.fit_label(label)

	def predict(self):
		test_kernel_matrix = self.kernel(x_test, x_train.T)
		scores = [torch.sum(test_kernel_matrix * Predictor.sgn_label(self.y_train, label) * alpha, (1)) for label, alpha in enumerate(self.model)]
		predictions = torch.max(torch.stack(scores), 0)[1]
		return torch.sum(predictions != self.y_test) / self.x_test.shape[0]

if __name__ == "__main__":

	from torch.utils.data import DataLoader
	from torchvision.transforms import ToTensor, Compose, Lambda
	from torchvision import datasets

	def polynomial(a, b, c = 1., degree = 5.):
		return torch.float_power(a @ b + c, degree)

	label_set = range(10)

	training_data = datasets.MNIST(
		root="data",
		train=True,
		download=True,
		transform=Compose([
			ToTensor(),
			Lambda(lambda x: x.reshape((-1, )))
		])
	)

	test_data = datasets.MNIST(
		root="data",
		train=False,
		download=True,
		transform=Compose([
			ToTensor(),
			Lambda(lambda x: x.reshape((-1, )))
		])
	)

	training_size = 1_000 # training_data.data.shape[0]
	test_size = 500 # test_data.data.shape[0]

	train_dataloader = DataLoader(training_data, batch_size=training_size, shuffle=True)
	test_dataloader = DataLoader(test_data, batch_size=test_size, shuffle=True)

	train_examples = iter(train_dataloader)
	test_examples = iter(test_dataloader)

	x_train, y_train = next(train_examples)
	x_test, y_test = next(test_examples)

	MKP = MultilabelKernelPerceptron(
		polynomial,
		set(label_set),
		10,
		x_train,
		y_train,
		x_test,
		y_test
	)

	MKP.fit()
	print(MKP.predict())