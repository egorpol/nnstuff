# nnstuff

Resources I use for teaching topics related to machine learning and neural networks.

- `neuron.ipynb`: A quick introduction to neurons, weights, biases, and dot products.
- `dufour_peaks_regression2.ipynb`: An introduction to linear and polynomial regression based on micro timing analysis of peak data from "Bocalises Prélude" by Denis Dufour.
- `gradient2d.ipynb`, `gradient3d.ipynb`: Interactive visualizations for gradient descent in 2D and 3D.
- `mnist_dcgan` folder: Simple DCGAN implementation for generating handwritten numbers based on the MNIST dataset, following the 2015 paper by Radford et al. This implementation includes slight modifications, such as using Adam instead of SGD as the optimizer and using ReLU instead of LeakyReLU within the generator. It contains three different variants with different network sizes (128, 256, and 512 top layer sizes).

### References
- [Radford et al., "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks," arXiv:1511.06434v2](https://arxiv.org/abs/1511.06434v2)
