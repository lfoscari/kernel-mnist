from kmeans_pytorch import kmeans
import math
import torch

# Vale che sia sum sqrt(t) >= sqrt(sum t), questo fa schifo perché non posso dire che la matrice di kernel
# sia lineare su n, effettivamente i risultati sono buoni, ma bisogna fare scena. Quindi è il caso di
# provare a calcolare il numero di centroidi con t_i e con n, in particolare calcolo sqrt(n) e lo divido
# tra gli insiemi S_i in base alla loro numerosità, in modo che sum t_i = sqrt(n).
# Quindi K = sqrt(n) * (n / t_i) per ogni i.

def compress(x_train, y_train):
	# The idea is to split the training data according to the label and
	# then find the clusters, this way we'll have clusters for each possible
	# class, making it possible to label the centroids.

	# Sort x_train by label in y_train
	_, indices = y_train.sort()
	x_train = x_train[indices]

	# Count the occurrences of each label
	label_amount = y_train.bincount()

	# Split the training points in one of 10 buckets according to their label
	label_split = x_train.split(tuple(label_amount))

	# Compute centroid set for each bucket
	centroids = {label: None for label in label_set}
	bucket_sizes = []

	for label, bucket in enumerate(label_split):
		# Taking K = sqrt(n) makes the kernel matrix linear in n NOOOOOOO
		K = int(math.sqrt(bucket.shape[0]))
		centroids[label] = kmeans(bucket, K)
		bucket_sizes.append(K)

	# Create the new training set by joining the buckets,
	# labeling the data and shuffleing everything

	x_train = torch.cat([c for _, c in centroids.values()])
	y_train = torch.cat([torch.empty(K).fill_(label) for K, label in zip(bucket_sizes, label_set)])

	return x_train, y_train


if __name__ == "__main__":
	from MultilabelKernelPerceptron import *
	from interface import RESULTS_WITH_DEGREE as RESULTS, EPOCHS, DEGREES
	from MNIST import label_set, batch_data_iter
	from tqdm import tqdm
	from functools import partial
	import time
	import json

	def polynomial(a, b, c = 1., degree = 5.):
		return torch.float_power(a @ b + c, degree)

	(x_train, y_train), (x_test, y_test) = batch_data_iter(60_000, 10_00)

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

	dest = "./results/kmmkp.json"
	json.dump(RESULTS, open(dest, "w"), indent=True)
	print("Results saved in", dest)