from interface import Predictor
from dataclasses import dataclass
import torch

# TODO: fix terrible accuracy

@dataclass(repr=False)
class MultilabelPerceptron(Predictor):
	labels: set
	epochs: int
	x_train: torch.Tensor
	y_train: torch.Tensor
	model: torch.Tensor = None

	def __fit_label(self, label):
		w = torch.zeros(self.x_train.shape[1], dtype=self.x_train.dtype)
		e = 0

		y_train_norm = Predictor.sgn_label(self.y_train, label)

		while True:
			update = False
			for point, label in zip(self.x_train, y_train_norm):
				if label * w.dot(point) <= 0:
					w += label * point
					update = True

			if not update:
				# print("Skipping remaining epochs") # DEBUG
				break
			
			e += 1
			if self.epochs is not None and e >= self.epochs:
				break
			
		return w

	def fit(self): 
		self.model = torch.zeros((len(self.labels), self.x_train.shape[1]))

		for label in self.labels:
			self.model[label] = self.__fit_label(label)

	def predict(self, x_test, y_test):
		scores = x_test @ self.model.T
		predictions = torch.max(scores, 1)[1]
		return float(torch.sum(predictions != y_test) / y_test.shape[0])

if __name__ == "__main__":
	from tqdm import tqdm
	import json
	import time
	from torch.utils.data import DataLoader
	from MNIST import label_set, train_data, test_data

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
		epochs_iteration.set_description(f"Training with {epochs} epoch(s)")
		epoch_training_time = time.time()

		MP = MultilabelPerceptron(
			label_set,
			epochs,
			x_train,
			y_train
		)

		MP.fit()
		results["epochs_amount"][epochs] = {
			"error": MP.predict(x_test, y_test),
			"time": time.time() - epoch_training_time
		}

	results["training_time"] = time.time() - start

	dest = "./results/mp.json"
	json.dump(results, open(dest, "w"), indent=True)
	print("Results saved in", dest)