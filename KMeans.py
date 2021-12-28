from kmeans_pytorch import kmeans
import math
import torch

def compress(x_train, y_train):
	# The idea is to split the training data according to the label and
	# then find the clusters, this way we'll have clusters for each possible
	# class, making it possible to label the centroids.

	root = math.sqrt(x_train.shape[0])

	# Sort x_train by label in y_train
	_, indices = y_train.sort()
	sorted_x_train = x_train[indices]

	# Count the occurrences of each label
	label_amount = y_train.bincount()

	# Split the training points in one of 10 buckets according to their label
	label_split = sorted_x_train.split(tuple(label_amount))

	# Compute centroid set for each bucket
	centroids = {label: None for label in label_set}
	bucket_sizes = []

	for label, bucket in enumerate(label_split):
		K = math.ceil(bucket.shape[0] / root)
		centroids[label] = kmeans(bucket, K)
		bucket_sizes.append(K)

	# Create the new training set by joining the buckets,
	# labeling the data and shuffleing everything
	x_train_km = torch.cat([c for _, c in centroids.values()])
	y_train_km = torch.cat([torch.empty(K).fill_(label) for K, label in zip(bucket_sizes, label_set)])

	# Alternative approach
	# This one simply sets a K and uses k-means on the whole dataset, then
	# assigns to each centroid the most common label in its circle.
	# Problem: it will be killed by the OS even with 10_000 examples

	if False:
		x_train_km, indices = kmeans(x_train, math.ceil(root))

		y_train_km = torch.empty(x_train.shape[0])
		for index in range(x_train_km.shape[0]):
			members = [point_index for point_index, center_index in enumerate(range(x_train.shape[0])) if center_index == index]
			labels = y_train[members]
			y_train_km[index] = max(set(labels), key=labels.count)

	return x_train_km, y_train_km


if __name__ == "__main__":
	from MultilabelKernelPerceptron import *
	from utils import RESULTS_WITH_DEGREE as RESULTS, EPOCHS, DEGREES, save_to_json, save_to_csv
	from MNIST import label_set, batch_data_iter
	from tqdm import tqdm
	from functools import partial
	import time

	def polynomial(a, b, c = 1., degree = 5.):
		return torch.float_power(a @ b + c, degree)

	(x_train, y_train), (x_test, y_test) = batch_data_iter(60_000, 10_000)

	print("K-means approximation step...")

	sketching_time = time.time()
	x_train_km, y_train_km = compress(x_train, y_train)
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

	save_to_json(RESULTS, "kmmkp")
	save_to_csv(RESULTS, "kmmkp")