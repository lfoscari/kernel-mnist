from torchvision.transforms import ToTensor, Compose, Lambda
from torchvision import datasets

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