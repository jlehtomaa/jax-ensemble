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
effectively quantifies the epistemic uncertainty. The figure below illustrates the idea. Ensembles are useful
also because they can easily be evaluated in an *optimistic* manner using a softmax function to aggregate over individual
member's predictions. For instance in reinforcement learning, this can be exploited to guide exploration towards
high-uncertainty areas, to boost the information gain of the future control sequences.

![plot](./experimets/results/ens.png)

## Running the code

Simply run use

```bash 
python experiments/train_ensemble/run.py
```

to train the ensemble models. All hyperparameter tweaks can be directly done in the ```conf/config.yaml``` file.


## References

The implementation heavily borrows ideas from following excellent blog posts:

[Parallel JAX training](https://willwhitney.com/parallel-training-jax.html#training-more-networks-in-parall) by Will Whitney,
[Introduction to randomized prior functions](https://gdmarmerola.github.io/intro-randomized-prior-functions/) by Guilherme Marmerola, and the [corresponding JAX translation](https://github.com/petergchang/randomized_priors) by Peter G. Chang.

The relevant papers are:

[Randomized Prior Functions for Deep Reinforcement Learning](https://proceedings.neurips.cc/paper/2018/file/5a7b238ba0f6502e5d6be14424b20ded-Paper.pdf) by Ian Osband et al. 2018.

The toy data is from the [Weight Uncertianty in Neural Networks](https://arxiv.org/pdf/1505.05424.pdf) paper by Blundell et al. 2015.

The ideas of using optimistic evaluation of the ensemble to guide exploration has been used for instance in the [POLO](https://arxiv.org/abs/1811.01848) algorithm by Lowrey et al. 2019 and in the [AOP](https://arxiv.org/abs/1912.01188) paper by Lu et al. 2020.

