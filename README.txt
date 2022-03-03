Naive implementation of the multilabel kernel perceptron for the MNIST dataset.
To speed-up computation a K-mean approach is used to reduce dimensionality
without loosing too much information. The tests can be replicated by running
the KMeans.py file inside the shell.nix environment.

$ git clone https://github.com/lfoscari/mnist-perceptron
$ nix-shell
$ python3 KMeans.py

Thanks to the kmeans-pytorch library the K-means implementation used can
utilize the GPU, because it relies on PyTorch.
