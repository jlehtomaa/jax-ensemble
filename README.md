# jax-ensemble
A JAX template for neural network ensembles with randomized priors.

Ensembling multiple individual neural networks into a single model is a simple
and effective method for quantifying predictive uncertainty.
This repository uses a toy dataset to set up an ensemble template in [JAX](https://github.com/google/jax).
Using JAX makes ensembling particularly simple, as we can just vectorize
(```jax.vmap```) the network initialization and training steps, and use otherwise standard neural network architectures and code.
To improve the uncertianty estimates of neural network models, this repo considers
[randomized prior functions](https://proceedings.neurips.cc/paper/2018/file/5a7b238ba0f6502e5d6be14424b20ded-Paper.pdf)
 &agrave; la Osband et al. 2018.

## Training results

The ensemble behaviour is very intuitive. In regions with plentiful training examples, all ensemble members agree
on the prediction. However, in areas with fewer data points, the variance within the ensemble members
effectively quantifies the epistemic uncertainty. The figure below illustrates the idea.

![](experimets/results/ens.png)

### References

The implementation draws from:

https://willwhitney.com/parallel-training-jax.html#training-more-networks-in-parallel