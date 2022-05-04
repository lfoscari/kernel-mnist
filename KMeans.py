from pykeops.torch import LazyTensor
import shutil
import json
import time
import os

from utils import *
from MNIST import label_set, mnist_loader

DATASET_TEMPORARY_DIR = "/tmp/kmmkp-dataset-sketching"
DATASET_LOCATION = "./sketch"
SKETCHING_TIME_LOCATION = f"{RESULTS_DIR}/sketching-time.json"


def kmeans(x, k=10, iterations=10, verbose=True):
    """
    k-means using Lloyd's algorithm for the Euclidean metric.
    Implementation by kernel-operations.io
    """

    N, D = x.shape  # Number of samples, dimension of the ambient space

    c = x[:k, :].clone()  # Simplistic initialization for the centroids

    x_i = LazyTensor(x.view(N, 1, D))  # (N, 1, D) samples
    c_j = LazyTensor(c.view(1, k, D))  # (1, K, D) centroids

    # K-means loop:
    # - x  is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c  is the (K, D) cloud of cluster centroids
    for i in range(iterations):

        # E step: assign points to the closest cluster -------------------------
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        c.zero_()
        c.scatter_add_(0, cl[:, None].repeat(1, D), x)

        # Divide by the number of points per cluster:
        Ncl = torch.bincount(cl, minlength=k).type_as(c).view(k, 1)
        c /= Ncl  # in-place division to compute the average

    return cl, c


def compress(xs, ys, target_size):
    """
    Splits the training data according to the label into clusters and then find the right number of centers for each
    cluster. Given that each cluster has an associated label, the resulting centers will be classified with such label.
    """

    # Sort x_train by label in y_train
    _, indices = ys.sort()
    sorted_xs = xs[indices]

    # Count the occurrences of each label
    label_amount = ys.bincount()

    # Split the training points in one of 10 buckets according to their label
    label_split = sorted_xs.split(tuple(label_amount))

    # Compute centroid set for each bucket
    centers = {label: None for label in label_set}
    bucket_sizes = []

    for label, bucket in enumerate(label_split):
        # If L is the amount of data-points with a particular label in the dataset with size N,
        # we want to reduce the total size from N to N' we want the ratio L/N to stay
        # roughly the same when reducing, therefore we ask that L' / N' =~ L / N, to find the
        # new amount of data-points with the given label we must solve for L':
        #   L' = L * N' / N
        # Pick at least 2 centers for cluster for good measure.
        centers_amount = max(2, bucket.shape[0] * target_size // xs.shape[0])
        centers[label] = kmeans(bucket, k=centers_amount)
        bucket_sizes.append(centers_amount)

    # Create the new training set by joining the buckets, labeling the data
    xs_km = torch.cat([c for _, c in centers.values()])
    ys_km = torch.cat([torch.empty(centers_amount).fill_(label) for centers_amount, label in zip(bucket_sizes, label_set)])

    # Shuffle everything
    permutation = torch.randperm(xs_km.shape[0], device=DEVICE)
    xs_km = xs_km[permutation]
    ys_km = ys_km[permutation]

    return xs_km, ys_km


def compress_dataset():
    """
    Loads the original dataset from PyTorch and applies a sketching method based on K-means.
    Multiple reductions are tested. The results are saved in 'DATASET_LOCATION' and the
    time required to complete the sketching in 'SKETCHING_TIME_LOCATION'.
    """

    if os.path.exists(DATASET_LOCATION):
        print("Dataset sketch is present... skipping")
        return

    if os.path.exists(DATASET_TEMPORARY_DIR):
        shutil.rmtree(DATASET_TEMPORARY_DIR)

    os.mkdir(DATASET_TEMPORARY_DIR)

    print("Loading original MNIST dataset")

    (x_train, y_train), (_, _) = mnist_loader(TRAINING_SET_SIZE, TEST_SET_SIZE)

    print("Sketching dataset using K-means")

    sketching_time = {ts: None for ts in REDUCTIONS}

    for target_size in REDUCTIONS:
        print(f"\tK-means approximation with target size {target_size}")

        start = time.time()
        x_train_km, y_train_km = compress(x_train, y_train, target_size)
        sketching_time[target_size] = time.time() - start

        os.mkdir(f"{DATASET_TEMPORARY_DIR}/{target_size}")

        torch.save(x_train_km, f"{DATASET_TEMPORARY_DIR}/{target_size}/x_train_km.pt")
        torch.save(y_train_km, f"{DATASET_TEMPORARY_DIR}/{target_size}/y_train_km.pt")

    shutil.move(DATASET_TEMPORARY_DIR, DATASET_LOCATION)
    json.dump(sketching_time, open(SKETCHING_TIME_LOCATION, "w"), indent=4)

    print(f"Results saved in {DATASET_LOCATION}")
    print(f"Time measurements saved in {SKETCHING_TIME_LOCATION}")
