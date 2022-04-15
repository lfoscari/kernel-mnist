#!/usr/bin/env python3

from functools import partial
from tqdm import tqdm
import time
import gc

from utils import *

from MultilabelKernelPerceptron import MultilabelKernelPerceptron
from KMeans import compress_dataset, REDUCTIONS, DATASET_LOCATION, RESULTS_LOCATION
from MNIST import label_set


def run_tests():
    """
    Run the kernel perceptron implementation on the MNIST dataset using the sketched data-points, measure training
    time, test error and training error.
    """

    x_train = torch.load(f"{DATASET_LOCATION}/x_train.pt", map_location=DEVICE)
    y_train = torch.load(f"{DATASET_LOCATION}/y_train.pt", map_location=DEVICE)

    x_test = torch.load(f"{DATASET_LOCATION}/x_test.pt", map_location=DEVICE)
    y_test = torch.load(f"{DATASET_LOCATION}/y_test.pt", map_location=DEVICE)

    print(f"Running Multi-label Kernel Perceptron with k-means sketching on MNIST dataset")

    for reduction in REDUCTIONS:
        x_train_km = torch.load(f"{DATASET_LOCATION}/{reduction}/x_train_km.pt", map_location=DEVICE)
        y_train_km = torch.load(f"{DATASET_LOCATION}/{reduction}/y_train_km.pt", map_location=DEVICE)

        results = RESULTS_TEMPLATE.copy()
        epochs_iteration = tqdm(EPOCHS)

        for epochs in epochs_iteration:
            for degree in DEGREES:
                epochs_iteration.set_description(f"Training with {epochs} epoch(s) and degree {degree} [{reduction} examples]")
                perceptron = MultilabelKernelPerceptron(
                    partial(polynomial, degree=degree),
                    label_set,
                    epochs,
                    x_train_km,
                    y_train_km,
                    DEVICE
                )

                training_time = time.time()
                perceptron.fit()
                training_time = time.time() - training_time

                results["epochs"][epochs]["degree"][degree] = {
                    "training_time": training_time,
                    # The training error should be computed on x_train or x_train_km?
                    "training_error": perceptron.error(x_train[:10_000], y_train[:10_000]),
                    "test_error": perceptron.error(x_test, y_test)
                }

        del x_train_km, y_train_km
        gc.collect()

        save_to_csv(results, f"{RESULTS_LOCATION}/{reduction}-kmmkp.csv")


if __name__ == "__main__":
    torch.manual_seed(SEED)

    compress_dataset()
    run_tests()
