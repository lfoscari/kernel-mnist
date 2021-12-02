from MultilabelKernelPerceptron import MultilabelKernelPerceptron
from kmeans_pytorch import kmeans
from MultilabelKernelPerceptron import *
import math
import torch

if __name__ == "__main__":
	from torch.utils.data import DataLoader
	from MNIST import label_set, train_data, test_data

	def polynomial(a, b, c = 1., degree = 5.):
		return torch.float_power(a @ b + c, degree)

	train_dataloader = DataLoader(train_data, batch_size=10_000, shuffle=True)
	test_dataloader = DataLoader(test_data, batch_size=500, shuffle=True)

	train_examples = iter(train_dataloader)
	test_examples = iter(test_dataloader)

	x_train, y_train = next(train_examples)
	x_test, y_test = next(test_examples)

	# The idea is to split the training data according to the label and
	# then find the clusters, this way we'll have clusters for each possible
	# class, making it possible to label the centroids.

	print("Before compression (train):", x_train.shape)

	# Sort x_train by label in y_train
	_y_train, indices = y_train.sort()
	_x_train = x_train[indices]

	# Count the occurrences of each label
	label_amount = y_train.bincount()

	# Split the training points in one of 10 buckets according to their label
	label_split = _x_train.split(tuple(label_amount))

	# Compute centroid set for each bucket
	centroids = {label: None for label in label_set}
	bucket_sizes = []

	for label, bucket in enumerate(label_split):
		K = int(math.sqrt(bucket.shape[0]))
		centroids[label] = kmeans(bucket, K)
		bucket_sizes.append(K)

	# Create the new training set by joining the buckets,
	# labeling the data and shuffleing everything

	_x_train = torch.cat([c for _, c in centroids.values()])
	_y_train = torch.cat([torch.empty(K).fill_(label) for K, label in zip(bucket_sizes, label_set)])

	print("After compression (train):", _x_train.shape)

	results = {}

	for epochs in range(1, 11):
		print("-" * 10, "Training", epochs, "epochs")
		results[epochs] = {}

		for degree in range(1, 7):
			print("-" * 5, "Training degree", degree)

			MKP = MultilabelKernelPerceptron(
				partial(polynomial, degree=degree),
				set(label_set),
				epochs,
				_x_train,
				_y_train,
				x_test,
				y_test,
				"./results/mkp.json"
			)

			MKP.fit()
			results[epochs][degree] = MKP.predict()
			print("[", "Error:", results[epochs][degree], "]")

	dest = "./results/kmmkp.json"
	json.dump(results, open(dest, "w"), indent=True)
	print("Results saved in", dest)