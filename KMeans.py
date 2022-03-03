from MNIST import label_set
from utils import SEED
from kmeans_pytorch import kmeans
import math
import torch


def compress(xs, ys, scaling=math.sqrt):
    # The idea is to split the training data according to the label and
    # then find the clusters, this way we'll have clusters for each possible
    # class, making it possible to label the centroids.

    size = scaling(xs.shape[0])

    # Sort x_train by label in y_train
    _, indices = ys.sort()
    sorted_x_train = xs[indices]

    # Count the occurrences of each label
    label_amount = ys.bincount()

    # Split the training points in one of 10 buckets according to their label
    label_split = sorted_x_train.split(tuple(label_amount))

    # Compute centroid set for each bucket
    centroids = {label: None for label in label_set}
    bucket_sizes = []

    for label, bucket in enumerate(label_split):
        centers_amount = max(2, int(bucket.shape[0] / size))
        centroids[label] = kmeans(bucket, centers_amount)
        bucket_sizes.append(centers_amount)

    # Create the new training set by joining the buckets,
    # labeling the data
    xs_km = torch.cat([c for _, c in centroids.values()])
    ys_km = torch.cat([torch.empty(K).fill_(label) for K, label in zip(bucket_sizes, label_set)])

    # Shuffle everything
    torch.manual_seed(SEED)
    permutation = torch.randperm(xs_km.shape[0])
    xs_km = xs_km[permutation]
    ys_km = ys_km[permutation]

    # Alternative approach
    # This one simply sets a K and uses k-means on the whole dataset, then
    # assigns to each centroid the most common label in its circle.
    # Problem: it will be killed by the OS even with 10_000 examples

    # x_train_km, indices = kmeans(x_train, math.ceil(root))
    #
    # y_train_km = torch.empty(x_train.shape[0])
    # for index in range(x_train_km.shape[0]):
    # 	members = [point_index for point_index, center_index in enumerate(range(x_train.shape[0])) if center_index == index]
    # 	labels = y_train[members]
    # 	y_train_km[index] = max(set(labels), key=labels.count)

    return xs_km, ys_km


def run_tests():
    from utils import EPOCHS, DEGREES, RESULTS_TEMPLATE, save_to_csv, polynomial
    from MultilabelKernelPerceptron import MultilabelKernelPerceptron
    from MNIST import batch_data_iter
    from tqdm import tqdm
    from functools import partial
    import time

    k_funcs = {
        "sqrt": math.sqrt,
        "half": lambda x: x / 2,
        "tenth": lambda x: x / 10,
    }

    TRAINING_SET_SIZE = 60_000
    TEST_SET_SIZE = 10_000

    (x_train, y_train), (x_test, y_test) = batch_data_iter(TRAINING_SET_SIZE, TEST_SET_SIZE)
    print(f"Running Multi-label Kernel Perceptron with k-means sketching on {TRAINING_SET_SIZE}/{TEST_SET_SIZE} MNIST dataset")

    for compression, k_func in k_funcs.items():
        RESULTS = RESULTS_TEMPLATE.copy()

        print(f"K-means approximation step with '{compression}' function...")
        sketching_time = time.time()

        x_train_km, y_train_km = compress(x_train, y_train, k_func)

        sketching_time = time.time() - sketching_time
        RESULTS["sketching_time"] = sketching_time

        epochs_iteration = tqdm(EPOCHS)

        for epochs in epochs_iteration:
            for degree in DEGREES:
                epochs_iteration.set_description(f"Training with {epochs} epoch(s) and degree {degree}")
                training_time = time.time()

                MKP = MultilabelKernelPerceptron(
                    partial(polynomial, degree=degree),
                    label_set,
                    epochs,
                    x_train_km,
                    y_train_km
                )

                MKP.fit()

                training_time = time.time() - training_time
                RESULTS["epochs"][epochs]["degree"][degree] = {
                    "training_time": training_time,
                    "training_error": MKP.predict(x_train_km, y_train_km),
                    "test_error": MKP.predict(x_test, y_test)
                }

        save_to_csv(RESULTS, f"{compression}-kmmkp")


if __name__ == "__main__":
    run_tests()
