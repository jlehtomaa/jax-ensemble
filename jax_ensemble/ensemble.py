"""
This module implements a simple neural network ensemble model.
The idea is to first define a standard MLP, and then use the jax.vmap
functionality to vectorize the relevant initialization, training,
and prediction functions.

This module borrows ideas from the following blog post:
https://willwhitney.com/parallel-training-jax.html#training-more-networks-in-parall
"""

import jax
import jax.numpy as jnp
import jax.random as jrn

from jax_ensemble.mlp import (create_train_state,
                              pred_fn,
                              train_step_fn)

# Parallelize the network initializer to set up an ensemble of neural nets.
# create_train_state(key, cfg) is vmapped to take an array of key as input.
vec_init_fn = jax.vmap(create_train_state, in_axes=(0, None))

# Parallelize the predict function to take in multiple train states (an
# ensemble of networks) but only a single input vector. That is, map
# pred_fn(state, x) only over the first argument. Calling the function
# outputs the prediction by each _individual_ ensemble member.
vec_pred_single_in_fn = jax.vmap(pred_fn, in_axes=(0, None))

# Parallelize the training step. Each ensemble member receives its
# own training data batch. That is, vectorize the train_step_fn(state, batch)
# call both over the training states and input batches.
bootstrap_train_step_fn = jax.vmap(train_step_fn, in_axes=(0, 0))

@jax.jit
def ensemble_pred(states, feat, temp):
    """Evaluate an ensemble network model as a whole.

    This function takes a single input feature, runs it through all
    ensemble members individually, and then takes a softmax of the
    predictions to produce one model output.

    Parameters
    ----------
    states : train_state.TrainState
        A Flax training state object that contains the network weights etc.

    feat : float array-like
        The input feature to the model

    temp : float
        The ensemble temperature. It controls how optimistic the ensemble
        evaluation is. A value converging to zero means that the function
        will output a value close to the ensemble mean. The higher the
        temperature, the closer the output will be to the ensemble maximum.

    Notes
    -----
    The implementation follow the repository for the
    Adaptive Online Planning for Continual Lifelong Learning (AOP) algorithm
    by Lu et al. 2020: https://github.com/kzl/aop/blob/master/models/Ensemble.py
    """

    # Prediction for each ensemble member at the same input vector.
    preds = vec_pred_single_in_fn(states, feat)

    # Normalize based on the number of ensemble members.
    exp_term = temp * preds - jnp.log(preds.shape[0])

    lse = jax.scipy.special.logsumexp(exp_term, axis=0)
    return (1 / temp) * lse

# Vectorize the ensemble_pred function to take in a batch of input vectors.
vec_ensemble_pred = jax.vmap(ensemble_pred, in_axes=(None, 0, None))

class Ensemble:
    """Ensemble neural network model.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for initializing the network weights.

    mlp_cfg : dict
        Configuration file for an individual ensemble member MLP.

    ens_cfg : dict
        Configuration file for the ensemble-level parameters.
    """

    def __init__(self, key, mlp_cfg, ens_cfg):

        self.num_models = ens_cfg["num_models"]
        self.temperature = ens_cfg["temperature"]

        keys = jrn.split(key, self.num_models)
        self.states = vec_init_fn(keys, mlp_cfg)

    def train(self, batch):
        """Train the network ensemble.

        Parameters
        ----------
        batch : dict
            A dictionary with the keys 'inputs' and 'labels', each
            containing a training batch. Importantly, this training
            specification assumes that each ensemble member gets a
            (potentially) different batch. Therefore, the inputs and labels
            need to have leading dimensions of (num_models, batch_size).

        Returns
        -------
        float
            Mean loss over all ensemble members.
        """
        self.states, losses = bootstrap_train_step_fn(self.states, batch)
        return losses.mean()

    def pred_all(self, feat):
        """Evaluate all ensemble models individually using a single input.

        Parameters
        ----------
        feat : float array-like
            A single model input feature.

        Returns
        -------
        float array-like, shape=(num_models, output_dim)
            Ensemble member predictions.
        """
        return vec_pred_single_in_fn(self.states, feat).squeeze()

    def pred(self, feat):
        """Evaluate the full ensemble as a whole.

        Parameters
        ----------
        feat : float array-like
            Input feature or batch of features.

        Returns
        -------
        float array-like, shape=(batch_size, output_dim)
            A single (optimistic) ensemble prediction.
        """
        return vec_ensemble_pred(self.states, feat, self.temperature).squeeze()
