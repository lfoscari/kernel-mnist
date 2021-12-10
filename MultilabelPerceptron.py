from interface import Predictor
from dataclasses import dataclass
import torch

@dataclass(repr=False)
class MultilabelPerceptron(Predictor):
	labels: set
	epochs: int
	x_train: torch.Tensor
	y_train: torch.Tensor
	model: torch.Tensor = None

	def __fit_label(self, label):
		w = torch.zeros(self.x_train.shape[1], dtype=self.x_train.dtype)
		y_train_norm = Predictor.sgn_label(self.y_train, label)
		epoch = 0

		while True:
			update = False
			for point, label in zip(self.x_train, y_train_norm):
				if Predictor.sgn(w.dot(point)) != label:
					w += label * point
					update = True

			if not update:
				# print("Skipping remaining epochs") # DEBUG
				break
			
			epoch += 1
			if self.epochs is not None and epoch >= self.epochs:
				break
			
		return w

	def fit(self): 
		self.model = torch.zeros((len(self.labels), self.x_train.shape[1]))

		for label in self.labels:
			self.model[label] = self.__fit_label(label)

	def predict(self, xs, ys):
		scores = xs @ self.model.T
		predictions = torch.max(scores, 1)[1]
		return float(torch.sum(predictions != ys) / ys.shape[0])

if __name__ == "__main__":
	from MNIST import label_set, batch_data_iter
	from interface import EPOCHS, RESULTS
	from tqdm import tqdm
	import json
	import time

	(x_train, y_train), (x_test, y_test) = batch_data_iter(10_000, 500)
	epochs_iteration = tqdm(EPOCHS)

	for epochs in epochs_iteration:
		epochs_iteration.set_description(f"Training with {epochs} epoch(s)")
		training_time = time.time()

		MP = MultilabelPerceptron(
			label_set,
			epochs,
			x_train,
			y_train
		)

		MP.fit()

		training_time = time.time() - training_time

		RESULTS["epochs"][epochs] = {
			"training_time": training_time,
			"training_error": MP.predict(x_train, y_train),
			"test_error": MP.predict(x_test, y_test)
		}

	dest = "./results/mp.json"
	json.dump(RESULTS, open(dest, "w"), indent=True)
	print("Results saved in", dest)