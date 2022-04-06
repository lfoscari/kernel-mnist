Naive implementation of the multilabel kernel perceptron for the MNIST dataset.
To speed-up computation a K-mean approach is used to reduce dimensionality
without loosing too much information. The experiments can be replicated by
running the following commands:

$ git clone git@github.com:lfoscari/mnist-perceptron.git
$ nix-shell
$ python3 experiments.py

Thanks to the kmeans-pytorch library is possible to run achieve a k-means
approximation of the dataset using a CUDA-compatible GPU.