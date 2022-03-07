import torch
from torchvision.transforms import ToTensor, Compose, Lambda
from torch.utils.data import DataLoader
from torchvision import datasets

label_set = list(range(10))


def batch_data_iter(training_batch_size, test_batch_size):
    """
    Loads the training and test examples into memory in batches
    and shuffles them according to the SEED set in utils.
    """

    train_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=Compose([
            ToTensor(),
            Lambda(lambda x: x.reshape((-1,)))
        ])
    )

    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=Compose([
            ToTensor(),
            Lambda(lambda x: x.reshape((-1,)))
        ])
    )

    train_dataloader = DataLoader(train_data, batch_size=training_batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=test_batch_size, shuffle=True)

    train_examples = iter(train_dataloader)
    test_examples = iter(test_dataloader)

    x_train, y_train = next(train_examples)
    x_test, y_test = next(test_examples)

    return (x_train, y_train), (x_test, y_test)
