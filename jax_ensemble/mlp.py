from functools import partial
from typing import Callable, Sequence, Dict
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training import train_state

class TrainState(train_state.TrainState):
    priors: Dict = None

class MLP(nn.Module):
    """Standard multilayer perceptron."""
    features: Sequence[int] # Last element must match the output dimension.
    activation: Callable = nn.tanh
    kernel_init: Callable = nn.initializers.glorot_normal()

    @nn.compact
    def __call__(self, x):
        """Forward pass of the MLP."""

        dense = partial(nn.Dense, kernel_init=self.kernel_init)

        for i, feat in enumerate(self.features):

            x = dense(features=feat)(x)

            if i != len(self.features) - 1:
                x = self.activation(x)

        return x


class PriorModel(nn.Module):
    """A network with randomized prior functions.

    Parameters
    ----------
    cfg : dict
        Configuration file with the network layer architecture and prior
        network weight.

    Notes
    -----
    Osband et al. 2018: Randomized Prior Functions for Deep Reinforcement Learning.
    """
    cfg: Dict

    def setup(self):
        self.prior_net = MLP(self.cfg.features)
        self.train_net = MLP(self.cfg.features)
        self.beta = self.cfg.prior_beta

    def __call__(self, x):
        """Forward pass of the PriorModel."""
        out_prior = self.prior_net(x)
        out_train = self.train_net(x)
        return self.beta * out_prior + out_train

def create_train_state(key, cfg):
    """Initializes the PriorModel training state.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for weight initialization.
    cfg : dict
        An MLP configuration file.

    Returns
    -------
    TrainState
        A Flax training state.
    """

    model = PriorModel(cfg)

    dummy_input = jnp.ones((1, cfg.input_dim))

    params = model.init(key, dummy_input)["params"]
    optimizer = optax.adamw(cfg.learning_rate, weight_decay=cfg.weight_decay)

    return TrainState.create(apply_fn=model.apply,
                             params=params["train_net"],
                             priors=params["prior_net"],
                             tx=optimizer)

def pred_fn(state, feat):
    """Run inference with the PriorModel.

    Parameters
    ----------
    state : TrainState
        Flax model representation.
    feat : float array-like
        Model input feature.

    Returns
    -------
    float array-like, shape=(output_dim,)
        Model prediction.
    """
    params = {"train_net": state.params, "prior_net": state.priors}
    return state.apply_fn({"params": params}, feat)


def apply_model(state, batch):
    """Train for a single gradient step.

    Parameters
    ----------
    state : TrainState
        Flax model representation.
    batch : dict
        Training batch with keys 'inputs' and 'labels'.

    Returns
    -------
    grads : float array-like
        Loss function gradients w.r.t. model weights.
    loss : float
        Model mean squared error value.
    """

    def loss_fn(params):
        """MSE loss function."""
        params_ = {"train_net": params, "prior_net": state.priors}
        preds = state.apply_fn({"params": params_}, batch['inputs'])
        loss = jnp.square(preds - batch["labels"]).mean()
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)

    return grads, loss

@jax.jit
def train_step_fn(state, batch):
    """Take one training step by applying the model and updating the weights.

    Parameters
    ----------
    state : TrainState
        Flax model representation.
    batch : dict
        Training batch with keys 'inputs' and 'labels'.

    Returns
    -------
    state : TrainState
        Updated model.
    loss : float
        Model mean squared error value.
    """

    grads, loss = apply_model(state, batch)
    state = state.apply_gradients(grads=grads)

    return state, loss
