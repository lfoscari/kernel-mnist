from torchvision.transforms import ToTensor, Compose, Lambda
from torch.utils.data import DataLoader
from torchvision import datasets

# TODO: find a way to remove numpy warning

label_set = list(range(10))

train_data = datasets.MNIST(
	root="data",
	train=True,
	download=True,
	transform=Compose([
		ToTensor(),
		Lambda(lambda x: x.reshape((-1, )))
	])
)

test_data = datasets.MNIST(
	root="data",
	train=False,
	download=True,
	transform=Compose([
		ToTensor(),
		Lambda(lambda x: x.reshape((-1, )))
	])
)

def batch_data_iter(batch_train, batch_test):
	train_dataloader = DataLoader(train_data, batch_size=batch_train, shuffle=True)
	test_dataloader = DataLoader(test_data, batch_size=batch_test, shuffle=True)

	train_examples = iter(train_dataloader)
	test_examples = iter(test_dataloader)

	x_train, y_train = next(train_examples)
	x_test, y_test = next(test_examples)

	return (x_train, y_train), (x_test, y_test)