from utils import DEGREES, save_to_json, save_to_csv
from sklearn.kernel_approximation import Nystroem
from MultilabelPerceptron import *
from utils import RESULTS_WITH_DEGREE as RESULTS, EPOCHS, DEGREES
from MNIST import label_set, batch_data_iter
from tqdm import tqdm
import torch
import time

if __name__ == "__main__":
	(x_train, y_train), (x_test, y_test) = batch_data_iter(10_000, 500)
	degrees_iteration = tqdm(DEGREES)

	for degree in degrees_iteration:
		sketching_time = time.time()

		kernel_approx = Nystroem(kernel="polynomial", degree=degree, coef0=1, n_components=100)
		kernel_approx = kernel_approx.fit(x_train)

		x_train_ny = torch.tensor(kernel_approx.transform(x_train),  dtype=x_train.dtype)
		x_test_ny = torch.tensor(kernel_approx.transform(x_test),  dtype=x_test.dtype)

		sketching_time = time.time() - sketching_time

		for epochs in EPOCHS:
			degrees_iteration.set_description(f"Training with {epochs} epoch(s) and degree {degree}")

			training_time = time.time()

			NMP = MultilabelPerceptron(
				label_set,
				epochs,
				x_train_ny,
				y_train
			)

			NMP.fit()

			training_time = time.time() - training_time

			RESULTS["epochs"][epochs]["degree"][degree] = {
				"training_time": training_time,
				"training_error": NMP.predict(x_train_ny, y_train),
				"test_error": NMP.predict(x_test_ny, y_test),
				"sketching_time": sketching_time
			}

	save_to_json(RESULTS, "nmp")
	save_to_csv(RESULTS, "nmp")