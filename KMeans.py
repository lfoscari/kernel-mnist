from kmeans_pytorch import kmeans
import math
import torch

def compress(x_train, y_train):
	# The idea is to split the training data according to the label and
	# then find the clusters, this way we'll have clusters for each possible
	# class, making it possible to label the centroids.

	# Sort x_train by label in y_train
	_, indices = y_train.sort()
	_x_train = x_train[indices]

	# Count the occurrences of each label
	label_amount = y_train.bincount()

	# Split the training points in one of 10 buckets according to their label
	label_split = _x_train.split(tuple(label_amount))

	# Compute centroid set for each bucket
	centroids = {label: None for label in label_set}
	bucket_sizes = []

	for label, bucket in enumerate(label_split):
		# Taking K = sqrt(n) maked the kernel matrix linear in n
		K = int(math.sqrt(bucket.shape[0]))
		centroids[label] = kmeans(bucket, K)
		bucket_sizes.append(K)

	# Create the new training set by joining the buckets,
	# labeling the data and shuffleing everything

	_x_train = torch.cat([c for _, c in centroids.values()])
	_y_train = torch.cat([torch.empty(K).fill_(label) for K, label in zip(bucket_sizes, label_set)])

	return _x_train, _y_train


if __name__ == "__main__":
	from MultilabelKernelPerceptron import *
	from torch.utils.data import DataLoader
	from MNIST import label_set, train_data, test_data
	from tqdm import tqdm
	from functools import partial
	import time
	import json

	def polynomial(a, b, c = 1., degree = 5.):
		return torch.float_power(a @ b + c, degree)

	train_dataloader = DataLoader(train_data, batch_size=10_000, shuffle=True)
	test_dataloader = DataLoader(test_data, batch_size=500, shuffle=True)

	train_examples = iter(train_dataloader)
	test_examples = iter(test_dataloader)

	x_train, y_train = next(train_examples)
	x_test, y_test = next(test_examples)

	old_shape = x_train.shape

	print("K-means approximation step...")
	start = time.time()
	x_train, y_train = compress(x_train, y_train)
	compression_time = time.time() - start

	start = time.time()
	results = {
		"compression_time": compression_time,
		"height_compression": x_train.shape[0] / old_shape[0],
		"epochs_amount": {},
	}
	
	epochs_iteration = tqdm(range(1, 11))

	for epochs in epochs_iteration:
		results["epochs_amount"][epochs] = {"degree": {}}

		for degree in range(1, 7):
			epochs_iteration.set_description(f"Training with {epochs} epoch(s) and degree {degree}")
			epoch_training_time = time.time()

			MKP = MultilabelKernelPerceptron(
				partial(polynomial, degree=degree),
				label_set,
				epochs,
				x_train,
				y_train
			)

			MKP.fit()
			results["epochs_amount"][epochs]["degree"][degree] = {
				"error": MKP.predict(x_test, y_test),
				"time": time.time() - epoch_training_time
			}

	results["training_time"] = time.time() - start

	dest = "./results/kmmkp.json"
	json.dump(results, open(dest, "w"), indent=True)
	print("Results saved in", dest)