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
# pred_fn(state, x) only over the first input.
vec_pred_single_in_fn = jax.vmap(pred_fn, in_axes=(0, None))

# Parallelize the predict function to take in multiple train states (an
# ensemble of networks) and a batch of input vectors. That is, map
# pred_fn(state, x) over both inputs. Run all ensemble members in parallel,
# each receiving its own input batch.
vec_pred_multi_in_fn = jax.vmap(pred_fn, in_axes=(0, 0))

@jax.jit
def ensemble_pred(states, inputs, temp):
    """
    Takes a single input sample, runs it through all models in the ensemble.
    forward_fn should be something that takes a single input and returns the
    output of each ensemble member.
    very high temp ----> ensemble max
    temp --> 0 ----> ensemble mean

    https://github.com/kzl/aop/blob/master/models/Ensemble.py
    """

    # Prediction for each ensemble member at the same input vector.
    preds = vec_pred_single_in_fn(states, inputs)

    exp_term = temp * preds - jnp.log(preds.shape[0])

    lse = jax.scipy.special.logsumexp(exp_term, axis=0)
    return (1 / temp) * lse

# Parallelize the ensemble_pred function to take in a batch of input vectors.
vec_ensemble_pred = jax.vmap(ensemble_pred, in_axes=(None, 0, None))

# Parallelize the training step. Again, each ensemble member receives its
# own training data batch.
bootstrap_train_step_fn = jax.vmap(train_step_fn, in_axes=(0, 0))


class Ensemble:

    def __init__(self, key, mlp_cfg, ens_cfg):

        self.num_models = ens_cfg["num_models"]
        self.temperature = ens_cfg["temperature"]

        keys = jrn.split(key, self.num_models)
        self.states = vec_init_fn(keys, mlp_cfg)

    



    # def _update_targets(self):

    #     if self._target_update_schedule == "hard":

    #         if self._n_updates % self._target_update_freq == 0:
    #             self.targets = self.targets.replace(
    #                 params=self.states.params)

    #     elif self._target_update_schedule == "soft":
    #         self.targets = polyak(self.states, self.targets, self._polyak_tau)

    #     else:
    #         raise ValueError("Incorrect target update schedule.")

    def train(self, batch):
        self.states, losses = bootstrap_train_step_fn(self.states, batch)
        return losses.mean()

    def pred_all(self, input):
        """Predict all ensemble models individually using a single input."""
        return vec_pred_single_in_fn(self.states, input).squeeze()

    def pred(self, input):
        """Evaluate the ensemble optimistically."""
        return vec_ensemble_pred(self.states, input, self.temperature).squeeze()

    # def pred_single_in(self, x):
    #     return parallel_pred_single_in_fn(self.targets, x)

    # def pred_multi_in(self, x): # NOTE, THIS GOES TO GENERATE TARGETS
    #     """
    #     Evaluate the ensemble model, each member receiving its own input batch.
    #     """
    #     return parallel_pred_multi_in_fn(self.targets, x)

    # def batch_eval(self, x): # NOTE, THIS GOES TO THE MPPI ACTION SELECTION
    #     return parallel_ensemble_pred(self.states, x, self.temp).squeeze()

    # def save_checkpoint(self, chkpt_dir):
    #     step = self.states.step[0]
    #     checkpoints.save_checkpoint(chkpt_dir, self.states, step, overwrite=True)

    # def restore_checkpoint(self, chkpt_dir):
    #     states = checkpoints.restore_checkpoint(chkpt_dir, self.states)
    #     self.states = self.states.replace(
    #         params=states.params, priors=states.priors)
    #     self.targets = deepcopy(self.states)
    #     #replace(params=new_tar)



