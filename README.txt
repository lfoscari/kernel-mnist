Naive implementation of the multilabel kernel perceptron for the MNIST dataset.
To speed-up computation a K-mean approach is used to reduce dimensionality
without loosing too much information. The experiments can be replicated by
running the following commands inside the cloned repository:

$ nix-shell
& python3 experiments.py

Note: the algorithm requires a pre-processing step to be completed, which
requires a hefty amount of memory available (roughly +4GB of free RAM), as this
step is fully reproducible, the results can already be found inside the
'sketch' directory, if you wish to rerun the pre-processing just delete the
folder and run the experiments.py executable.

Thanks to the kmeans-pytorch library is possible to run achieve a k-means
approximation of the dataset using a CUDA-compatible GPU.

TODO:
- Fill the confusion matrix
- Minimum requirements? Maybe at least 5GB of free RAM for the k-means
  compression, but the datasets can be archived and shipped with the project
- Check for contaminations between training and test set, which could explain
  the big difference between training error on the original data and test error.
