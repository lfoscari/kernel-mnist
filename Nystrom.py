from sklearn.kernel_approximation import Nystroem
from MultilabelPerceptron import *
import torch

# ERROR: the accuracy is terrible...

def polynomial(a, b, c = 1., degree = 5.):
    return (a @ b + c) ** degree

if __name__ == "__main__":
	from torch.utils.data import DataLoader
	from MNIST import label_set, train_data, test_data

	train_dataloader = DataLoader(train_data, batch_size=10_000, shuffle=True)
	test_dataloader = DataLoader(test_data, batch_size=500, shuffle=True)

	train_examples = iter(train_dataloader)
	test_examples = iter(test_dataloader)

	x_train, y_train = next(train_examples)
	x_test, y_test = next(test_examples)

	kernel_approx = Nystroem(polynomial, n_components=500)
	x_train_ny = torch.tensor(kernel_approx.fit_transform(x_train), dtype=x_train.dtype)
	x_test_ny = torch.tensor(kernel_approx.fit_transform(x_test),  dtype=x_test.dtype)

	print(x_train_ny.shape)

	results = {}

	NMP = MultilabelPerceptron(
		label_set,
		10, 
		x_train_ny,
		y_train,
		x_test_ny,
		y_test
	)

	NMP.fit()
	print("[", "Error:", NMP.predict(), "]")

