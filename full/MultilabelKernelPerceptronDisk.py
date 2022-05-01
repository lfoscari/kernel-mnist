from functools import partial
from tqdm import tqdm
import torch
import shutil
import time
import os
import gc

from MultilabelKernelPerceptron import MultilabelKernelPerceptron
from MNIST import label_set, mnist_loader
from utils import *

EPOCHS = 6
DEGREE = 4
KERNEL_MATRIX_DIR = "visualization/kernelmatrix"
MODEL_FILENAME = "full-mnist-model.pt"


def kernel_matrix_generator():
	index = 0
	while True:
		yield torch.load(f"{KERNEL_MATRIX_DIR}/{index}.pt")
		index = (index + 1) % TRAINING_SET_SIZE


class MultilabelKernelPerceptronDisk(MultilabelKernelPerceptron):
	def fit(self):
		kernel_matrix = kernel_matrix_generator()
		self.model = torch.empty((len(self.labels), self.xs.shape[0]), device=self.device)

		for label in self.labels:
			if self.approach == "min":
				self.model[label] = self.__fit_label_min(label, kernel_matrix)
			elif self.approach == "mean":
				self.model[label] = self.__fit_label_mean(label, kernel_matrix)
			elif self.approach == "weight":
				self.model[label] = self.__fit_label_weight(label, kernel_matrix)
			elif self.approach == "last":
				self.model[label] = self.__fit_label_last(label, kernel_matrix)
			else:
				raise AttributeError(approach)

	def __fit_label_mean(self, label, kernel_matrix):
		y_train_norm = sgn_label(self.ys, label)

		alpha = torch.zeros(self.xs.shape[0], device=self.device)
		alpha_sum = alpha.detach().clone()

		for epoch in range(self.epochs):
			for index, (label_norm, kernel_row) in enumerate(zip(y_train_norm, kernel_matrix)):
				alpha_update = sgn(torch.sum(alpha * y_train_norm * kernel_row)) != label_norm
				alpha[index] += alpha_update
				alpha_sum += alpha

				del kernel_row

		alpha_mean = alpha_sum / (self.epochs * self.xs.shape[0])
		return alpha_mean

	def error(self, xs, ys, kernel_matrix=None):
		predictions = torch.zeros(xs.shape[0], device=self.device)
		ys_norm = torch.stack([sgn_label(self.ys, label) for label in label_set])

		for index, example in tqdm(enumerate(xs)):
			kernel = self.kernel(example, self.xs.T)
			scores = torch.zeros(self.model.shape[0], device=self.device)

			for label, alpha in enumerate(self.model):
				scores[label] = torch.sum(kernel * ys_norm[label] * alpha)

			predictions[index] = torch.argmax(scores)

		return float(torch.sum(predictions != ys) / ys.shape[0])


def matrix(x_train):
	if os.path.exists(KERNEL_MATRIX_DIR):
        print("Kernel matrix already computed... skipping")
        return
	
	print(f"Computing kernel matrix ({KERNEL_MATRIX_DIR})...")
	for index, example in tqdm(enumerate(x_train)):
		kernel_row = polynomial(example, x_train.T, degree=DEGREE)
		torch.save(kernel_row, f"{KERNEL_MATRIX_DIR}/{index}.pt")


def train(x_train, y_train):
	if os.path.exists(MODEL_FILENAME):
        print("Model already fitted... skipping")
        return None

	print(f"Training the model...")
	perceptron = MultilabelKernelPerceptronDisk(
		partial(polynomial, degree=DEGREE),
		label_set,
		EPOCHS,
		x_train,
		y_train,
		"mean",
		DEVICE
	)

	print("Training the perceptron on the full MNIST dataset")
	training_time = time.time()

	perceptron.fit()
	
	print("Training time:", time.time() - training_time)
	torch.save(perceptron.model, MODEL_FILENAME)
	return perceptron.model


def error(model, x_train, y_train, x_test, y_test):
	if model is None:
		model = torch.load(MODEL_FILENAME)

	perceptron = MultilabelKernelPerceptronDisk(
		partial(polynomial, degree=DEGREE),
		label_set,
		EPOCHS,
		x_train,
		y_train,
		"mean",
		DEVICE
	)

	perceptron.model = model
	print("Test error:", perceptron.error(x_test, y_test))	

if __name__ == "__main__":
	torch.manual_seed(SEED)
	(x_train, y_train), (x_test, y_test) = mnist_loader(TRAINING_SET_SIZE, TEST_SET_SIZE)

	matrix(x_train)
	model = train(x_train, y_train) # 9448
	error(model, x_train, y_train, x_test, y_test)