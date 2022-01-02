from utils import Predictor
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
	
	def __fit_label(self, label, kernel_matrix):
		alpha = torch.zeros(self.x_train.shape[0])
		y_train_norm = Predictor.sgn_label(self.y_train, label)

		for _ in range(self.epochs):
			update = False

			for index, (label_norm, kernel_row) in enumerate(zip(y_train_norm, kernel_matrix)):
				alpha_update = Predictor.sgn(torch.sum(alpha * y_train_norm * kernel_row, (0))) != label_norm
				update = alpha_update or update
				alpha[index] += alpha_update

			if not update:
				break
			
		return alpha

	def fit(self):
		self.model = torch.empty((len(self.labels), self.x_train.shape[0]))
		kernel_matrix = self.kernel(self.x_train, self.x_train.T)

		for label in self.labels:
			self.model[label] = self.__fit_label(label, kernel_matrix)

	def predict(self, xs, ys):
		kernel_matrix = self.kernel(xs, self.x_train.T)
		scores = [torch.sum(kernel_matrix * Predictor.sgn_label(self.y_train, label) * alpha, (1)) for label, alpha in enumerate(self.model)]
		predictions = torch.max(torch.stack(scores), 0)[1]
		return float(torch.sum(predictions != ys) / ys.shape[0])
		

# if __name__ == "__main__":
# 	from MNIST import label_set, batch_data_iter
# 	from utils import RESULTS_TEMPLATE as RESULTS, EPOCHS, DEGREES, save_to_json, save_to_csv, polynomial
# 	from functools import partial
# 	from tqdm import tqdm
# 	import time

# 	TRAINING_SET_SIZE=10_000
# 	TEST_SET_SIZE=1_000

# 	(x_train, y_train), (x_test, y_test) = batch_data_iter(TRAINING_SET_SIZE, TEST_SET_SIZE)
# 	print(f"Running Multi-label Kernel Perceptron on {TRAINING_SET_SIZE}/{TEST_SET_SIZE} MNIST dataset")

# 	epochs_iteration = tqdm(EPOCHS)

# 	for epochs in epochs_iteration:
# 		for degree in DEGREES:
# 			epochs_iteration.set_description(f"Training with {epochs} epoch(s) and degree {degree}")
# 			training_time = time.time()

# 			MKP = MultilabelKernelPerceptron(
# 				partial(polynomial, degree=degree),
# 				label_set,
# 				epochs,
# 				x_train,
# 				y_train,
# 			)

# 			MKP.fit()

# 			training_time = time.time() - training_time
# 			RESULTS["epochs"][epochs]["degree"][degree] = {
# 				"training_time": training_time,
# 				"training_error": MKP.predict(x_train, y_train),
# 				"test_error": MKP.predict(x_test, y_test)
# 			}

# 	save_to_json(RESULTS, "mkp")
# 	save_to_csv(RESULTS, "mkp")