#!/usr/bin/env python3

from functools import partial
from itertools import product
from tqdm import tqdm
import time
import json

from utils import *

from MultilabelKernelPerceptron import MultilabelKernelPerceptron
from KMeans import compress_dataset, REDUCTIONS, DATASET_LOCATION
from MNIST import label_set, mnist_loader
from HyperparameterTuning import tune


def run_tests():
    """
    Run the kernel perceptron implementation on the MNIST dataset using the sketched data-points, measure training
    time, test error and training error.
    """

    print("Loading dataset...")

    (x_train, y_train), (x_test, y_test) = mnist_loader(TRAINING_SET_SIZE, TEST_SET_SIZE)

    print(f"Running Multi-label Kernel Perceptron with k-means sketching on MNIST dataset")

    results = RESULTS_TEMPLATE.copy()

    with tqdm(total=6) as progress:
        for reduction, approach in product(REDUCTIONS, ["min", "mean"]):
            x_train_km = torch.load(f"{DATASET_LOCATION}/{reduction}/x_train_km.pt", map_location=DEVICE)
            y_train_km = torch.load(f"{DATASET_LOCATION}/{reduction}/y_train_km.pt", map_location=DEVICE)

            progress.set_description(f"Tuning hyperparameters with reduction {reduction} and {approach} approach")
            tuning_time = time.time()
            epochs, degree = tune(x_train, y_train, reduction, approach)
            tuning_time = time.time() - tuning_time

            progress.set_description(f"Evaluating test error with reduction {reduction} and {approach} approach")
            perceptron = MultilabelKernelPerceptron(
                partial(polynomial, degree=degree),
                label_set,
                epochs,
                x_train_km,
                y_train_km,
                approach,
                DEVICE
            )

            training_time = time.time()
            perceptron.fit()
            training_time = time.time() - training_time

            training_error_km = perceptron.error(x_train_km, y_train_km)
            test_error = perceptron.error(x_test, y_test)

            results[reduction][approach] = {
                "training_error_km": training_error_km,
                "test_error": test_error,
                "epochs": epochs,
                "degree": degree,
                "tuning_time": tuning_time,
                "training_time": training_time
            }

            progress.update(1)

    json.dump(results, open(f"{RESULTS_LOCATION}/kernel-perceptron-results.csv", "w"), indent=2)

if __name__ == "__main__":
    torch.manual_seed(SEED)

    compress_dataset()
    run_tests()
