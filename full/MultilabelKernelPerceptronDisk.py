from dataclasses import dataclass
from functools import partial
from tqdm import tqdm
import shutil
import time
import json
import os

import sys

sys.path.append("../")

from MultilabelKernelPerceptron import MultilabelKernelPerceptron
from experiments import RESULTS_LOCATION
from MNIST import label_set, mnist_loader
from utils import *

KERNEL_MATRIX_TEMPORARY_DIR = "/tmp/kmmp-kernelmatrix"
KERNEL_MATRIX_DIR = "kernelmatrix"


def kernel_matrix_generator():
    index = 0
    while True:
        yield torch.load(f"{KERNEL_MATRIX_DIR}/{index}.pt")
        index = (index + 1) % TRAINING_SET_SIZE


@dataclass
class MultilabelKernelPerceptronDisk(MultilabelKernelPerceptron):
    def fit(self):
        kernel_matrix = kernel_matrix_generator()
        self.model = torch.empty((len(self.labels), self.xs.shape[0]), device=self.device)

        for label in self.labels:
            if self.approach == "min":
                self.model[label] = self._fit_label_min(label, kernel_matrix)
            elif self.approach == "mean":
                self.model[label] = self._fit_label_mean(label, kernel_matrix)
            elif self.approach == "weight":
                self.model[label] = self._fit_label_weight(label, kernel_matrix)
            elif self.approach == "last":
                self.model[label] = self._fit_label_last(label, kernel_matrix)
            else:
                raise AttributeError(self.approach)

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


def matrix(x_train, tuning_degree):
    if not os.path.exists(model_location):
        shutil.rmtree(KERNEL_MATRIX_DIR)

    if os.path.exists(KERNEL_MATRIX_DIR):
        print("Kernel matrix already computed... skipping")
        return

    os.mkdir(KERNEL_MATRIX_TEMPORARY_DIR)

    print(f"Computing kernel matrix...")
    for index, example in tqdm(enumerate(x_train)):
        kernel_row = polynomial(example, x_train.T, degree=tuning_degree)
        torch.save(kernel_row, f"{KERNEL_MATRIX_TEMPORARY_DIR}/{index}.pt")

    shutil.move(KERNEL_MATRIX_TEMPORARY_DIR, KERNEL_MATRIX_DIR)


def train(x_train, y_train, tuning_epochs, approach, tuning_degree, model_location):
    if os.path.exists(model_location):
        print("Model already fitted... skipping")
        return torch.load(model_location)

    perceptron = MultilabelKernelPerceptronDisk(
        partial(polynomial, degree=tuning_degree),
        label_set,
        tuning_epochs,
        x_train,
        y_train,
        approach,
        DEVICE
    )

    print("Training the perceptron on the full MNIST dataset")
    training_time = time.time()

    perceptron.fit()

    print("Training time:", time.time() - training_time)

    torch.save(perceptron.model, model_location)
    return perceptron.model


def error(model, x_train, y_train, x_test, y_test, approach, tuning_epochs, tuning_degree):
    print("Computing test error...")

    perceptron = MultilabelKernelPerceptronDisk(
        partial(polynomial, degree=tuning_degree),
        label_set,
        tuning_epochs,
        x_train,
        y_train,
        approach,
        DEVICE
    )

    perceptron.model = model
    print("Test error:", perceptron.error(x_test, y_test))


def main():
    (x_train, y_train), (x_test, y_test) = mnist_loader(TRAINING_SET_SIZE, TEST_SET_SIZE)
    results = json.load(open(RESULTS_LOCATION))

    for approach in APPROACHES:
        epochs = results["5000"][approach]["epochs"]
        degree = results["5000"][approach]["degree"]

        model_location = f"models/{approach}.pt"

        print(f"[{approach}]")

        matrix(x_train, degree)
        model = train(x_train, y_train, approach, epochs, degree, model_location)
        error(model, x_train, y_train, x_test, y_test, approach, epochs, degree)

        print("\n")


if __name__ == "__main__":
    torch.manual_seed(SEED)
    main()