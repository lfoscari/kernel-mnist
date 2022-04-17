from kmeans_pytorch import kmeans
import shutil
import torch
import json
import time
import os
import gc

from utils import *

from MNIST import label_set, mnist_loader

DATASET_TEMPORARY_LOCATION = "/tmp/kmmkp-dataset-sketching"
DATASET_LOCATION = "./sketch"
TIME_MEASUREMENT_LOCATION = f"{RESULTS_LOCATION}/sketching-time.json"

REDUCTIONS = [200, 1000, 1500]


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
        centers[label] = kmeans(bucket, centers_amount, device=DEVICE, seed=SEED, tqdm_flag=False)
        bucket_sizes.append(centers_amount)

    # Create the new training set by joining the buckets, labeling the data
    xs_km = torch.cat([c for _, c in centers.values()])
    ys_km = torch.cat([torch.empty(centers_amount).fill_(label) for centers_amount, label in zip(bucket_sizes, label_set)])

    # Shuffle everything
    permutation = torch.randperm(xs_km.shape[0], device=DEVICE)
    xs_km = xs_km[permutation]
    ys_km = ys_km[permutation]

    # Alternative approach
    # This one simply sets a k and uses k-means on the whole dataset, then
    # assigns to each center the most common label in its circle.
    # Problem: it will be killed by the OS even with 5000 examples

    # x_train_km, indices = kmeans(x_train, math.ceil(root))
    #
    # y_train_km = torch.empty(x_train.shape[0])
    # for index in range(x_train_km.shape[0]):
    # 	members = [point_index for point_index, center_index in enumerate(range(x_train.shape[0]))
    # 	    if center_index == index]
    # 	labels = y_train[members]
    # 	y_train_km[index] = max(set(labels), key=labels.count)

    return xs_km, ys_km


def compress_dataset():
    f"""
    Loads the original dataset from PyTorch and applies a sketching method based on K-means.
    Multiple reductions are tested. The results are saved in './{DATASET_LOCATION}' and the
    time required to complete the sketching in './{TIME_MEASUREMENT_LOCATION}'.
    """

    if os.path.exists(DATASET_LOCATION):
        print("Dataset already downloaded and compressed... skipping")
        return

    if os.path.exists(DATASET_TEMPORARY_LOCATION):
        shutil.rmtree(DATASET_TEMPORARY_LOCATION)

    os.mkdir(DATASET_TEMPORARY_LOCATION)

    print("Loading original MNIST dataset")

    (x_train, y_train), (x_test, y_test) = mnist_loader(TRAINING_SET_SIZE, TEST_SET_SIZE)

    print("Sketching dataset using K-means")

    sketching_time = {ts: None for ts in REDUCTIONS}

    for target_size in REDUCTIONS[::-1]:
        print(f"\tK-means approximation with target size {target_size}")

        start = time.time()
        x_train_km, y_train_km = compress(x_train, y_train, target_size)
        sketching_time[target_size] = time.time() - start

        os.mkdir(f"{DATASET_TEMPORARY_LOCATION}/{target_size}")

        torch.save(x_train_km, f"{DATASET_TEMPORARY_LOCATION}/{target_size}/x_train_km.pt")
        torch.save(y_train_km, f"{DATASET_TEMPORARY_LOCATION}/{target_size}/y_train_km.pt")

        del x_train_km, y_train_km
        gc.collect()

    shutil.move(DATASET_TEMPORARY_LOCATION, DATASET_LOCATION)
    json.dump(sketching_time, open(TIME_MEASUREMENT_LOCATION, "w"), indent=4)

    print(f"Results saved in {DATASET_LOCATION}")
    print(f"Time measurements saved in {TIME_MEASUREMENT_LOCATION}")
