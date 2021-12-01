from interface import Predictor
from dataclasses import dataclass
import torch

@dataclass(repr=False)
class MultilabelPerceptron(Predictor):
	labels: set
	epochs: int
	x_train: torch.Tensor
	y_train: torch.Tensor
	x_test: torch.Tensor
	y_test: torch.Tensor
	model: torch.Tensor = None

	def epoch_error(self, w, y_test_norm):
		return torch.sum(Predictor.sgn(self.x_test @ w.T) != y_test_norm) / self.x_test.shape[0]

	def fit(self):
		self.model = torch.zeros((len(self.labels), self.x_train.shape[1]))

		for label in self.labels:
			print("-" * 5, "Training label", label)
			self.model[label] = self.fit_label(label)

	def fit_label(self, label):
		w = torch.zeros(self.x_train.shape[1], dtype=self.x_train.dtype)
		e = 0

		y_train_norm = Predictor.sgn_label(self.y_train, label)
		y_test_norm = Predictor.sgn_label(self.y_test, label)

		while True:
			update = False
			for point, label in zip(self.x_train, y_train_norm):
				if label * w.dot(point) <= 0:
					w += label * point
					update = True

			error = self.epoch_error(w, y_test_norm)
			print("Epoch", e, "error", float(error))

			if not update:
				print("Skipping remaining epochs")
				break
			
			e += 1
			if self.epochs is not None and e >= self.epochs:
				break
			
		return w

	def predict(self):
		scores = self.x_test @ self.model.T
		predictions = torch.max(scores, 1)[1]
		return torch.sum(predictions != self.y_test) / self.y_test.shape[0]

if __name__ == "__main__":
	from torch.utils.data import DataLoader
	from MNIST import label_set, train_data, test_data

	train_dataloader = DataLoader(train_data, batch_size=10_000, shuffle=True)
	test_dataloader = DataLoader(test_data, batch_size=500, shuffle=True)

	train_examples = iter(train_dataloader)
	test_examples = iter(test_dataloader)

	x_train, y_train = next(train_examples)
	x_test, y_test = next(test_examples)

	MP = MultilabelPerceptron(
		label_set,
		10,
		x_train,
		y_train,
		x_test,
		y_test
	)

	MP.fit()
	print(MP.predict())