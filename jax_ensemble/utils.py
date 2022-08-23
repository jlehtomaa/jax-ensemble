import jax
import jax.numpy as jnp
import jax.random as jrn
import seaborn as sns
import matplotlib.pyplot as plt

def create_1d_dataset(key, size, minval=-3.5, maxval=3.5, sort=False):

    key1, key2 = jax.random.split(key)

    features = jrn.uniform(key1, shape=(size, 1), minval=minval, maxval=maxval)

    if sort:
        features = jnp.sort(features, axis=0)

    noise = jax.random.normal(key2, shape=features.shape) * 0.1
    labels = jnp.sin(features) + 0.25 * features + noise

    return features, labels

def draw_ensemble_batch(key, x_data, y_data, batch_size, num_models):

    n_data = x_data.shape[0]
    num_draws = batch_size * num_models

    inds = jax.random.randint(key, minval=0, maxval=n_data, shape=(num_draws,))
    shape = (num_models, batch_size, -1)
    return dict(inputs=x_data[inds].reshape(*shape),
                labels=y_data[inds].reshape(*shape))


def set_fig_layout(cfg):
    """Set consistent plot layouts."""

    sns.set_palette(cfg["palette"])
    sns.set_style(**cfg["seaborn"])
    plt.style.use(cfg["matplotlib"])
