import hydra
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax_ensemble.ensemble import Ensemble
from jax_ensemble.utils import (create_dataset,
                                draw_ensemble_batch,
                                set_fig_layout)

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg):
    """Train a neural network ensemble on a toy dataset."""

    # INITIALIZE MODEL AND DATA
    # -------------------------
    key = jax.random.PRNGKey(cfg["experiment_seed"])

    key, subkey = jax.random.split(key)
    ens = Ensemble(subkey, cfg["mlp"], cfg["ensemble"])

    key, subkey = jax.random.split(key)
    x_train, y_train = create_dataset(
        subkey, size=cfg["train"]["num_train_data"])

    x_eval = jnp.linspace(-0.5, 1.0, cfg["train"]["num_eval_data"]).reshape((-1, 1))


    # TRAINING LOOP
    # -------------
    for step in range(cfg["train"]["num_steps"]):
        key, subkey = jax.random.split(key)
        batch = draw_ensemble_batch(
            subkey, x_train, y_train, cfg["train"]["batch_size"], ens.num_models)

        mean_loss = ens.train(batch)

        if step % 100 == 0:
            print(f"Step: {step}, mean loss: {mean_loss:.5f}")

    # EVALUATION
    # ----------
    ens_pred = ens.pred(x_eval)
    ens_pred_all = ens.pred_all(x_eval)

    ens_mean = ens_pred_all.mean(axis=0)
    ens_std = ens_pred_all.std(axis=0)

    ub_2std = ens_mean + 2 * ens_std
    lb_2std = ens_mean - 2 * ens_std

    ub_1std = ens_mean + ens_std
    lb_1std = ens_mean - ens_std

    set_fig_layout(cfg["plots"])
    plt.plot(x_train, y_train, 'kx', label="Training data")
    plt.plot(x_eval, ens_pred, lw=3, color="black", label="Optimistic (softmax)")
    plt.plot(x_eval, ens_mean, color="tab:red", lw=2, label="Ensemble mean")
    plt.fill_between(x_eval.squeeze(), lb_2std, ub_2std, color="tab:blue", alpha=0.3)
    plt.fill_between(x_eval.squeeze(), lb_1std, ub_1std, color="tab:blue", alpha=0.7)
    plt.title("Ensemble training")
    plt.legend()
    plt.savefig(cfg["paths"]["results"] + "ens.png")


if __name__ == "__main__":
    main()
