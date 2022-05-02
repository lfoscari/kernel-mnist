Naive implementation of the multilabel kernel perceptron for the MNIST dataset.
To speed-up computation a K-mean approach is used to reduce dimensionality
without loosing too much information. The experiments can be replicated by
running the following commands inside the cloned repository:

$ nix-shell
& python3 experiments.py

In the 'full' directory is possible to run a modified version of the
algorithm on the whole MNIST dataset, technically also the version
contained in MultilabelKernelPerceptron.py should work, but computes the
kernel matrix before starting the training; in the case of the full
MNIST dataset this matrix occupies 28.8GB, the modified version saves
the matrix on the disk and not in memory.

Note: the algorithm requires a pre-processing step to be completed, which
requires a hefty amount of memory available (roughly +4GB of free RAM), as this
step is fully reproducible, the results can already be found inside the
'sketch' directory, if you wish to rerun the pre-processing just delete the
folder and run the experiments.py executable.

Thanks to the kmeans-pytorch library is possible to run achieve a k-means
approximation of the dataset using a CUDA-compatible GPU.
