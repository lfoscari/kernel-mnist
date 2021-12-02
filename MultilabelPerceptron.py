from interface import Predictor
from dataclasses import dataclass
import json
import torch

# ERROR: the accuracy is terrible...

@dataclass(repr=False)
class MultilabelPerceptron(Predictor):
	labels: set
	epochs: int
	x_train: torch.Tensor
	y_train: torch.Tensor
	x_test: torch.Tensor
	y_test: torch.Tensor
	model: torch.Tensor = None

	def __fit_label(self, label):
		w = torch.zeros(self.x_train.shape[1], dtype=self.x_train.dtype)
		e = 0

		y_train_norm = Predictor.sgn_label(self.y_train, label)

		while True:
			update = False
			for point, label in zip(self.x_train, y_train_norm):
				if Predictor.sgn(label * w.dot(point)) != label:
					w += label * point
					update = True

			if not update:
				print("Skipping remaining epochs")
				break
			
			e += 1
			if self.epochs is not None and e >= self.epochs:
				break
			
		return w

	def fit(self):
		self.model = torch.zeros((len(self.labels), self.x_train.shape[1]))

		for label in self.labels:
			print("Training label", label)
			self.model[label] = self.__fit_label(label)

	def predict(self):
		scores = self.x_test @ self.model.T
		predictions = torch.max(scores, 1)[1]
		return float(torch.sum(predictions != self.y_test) / self.y_test.shape[0])

if __name__ == "__main__":
	from torch.utils.data import DataLoader
	from MNIST import label_set, train_data, test_data

	train_dataloader = DataLoader(train_data, batch_size=10_000, shuffle=True)
	test_dataloader = DataLoader(test_data, batch_size=500, shuffle=True)

	train_examples = iter(train_dataloader)
	test_examples = iter(test_dataloader)

	x_train, y_train = next(train_examples)
	x_test, y_test = next(test_examples)

	results = {}

	for epochs in range(10):
		print("-" * 5, "Training", epochs, "epochs")
		results[epochs+1] = {}

		MP = MultilabelPerceptron(
			label_set,
			epochs+1,
			x_train,
			y_train,
			x_test,
			y_test
		)

		MP.fit()
		results[epochs+1] = MP.predict()
		print("[", "Error:", results[epochs+1], "]")

	dest = "./results/mp.json"
	json.dump(results, open(dest, "w"), indent=True)
	print("Results saved in", dest)