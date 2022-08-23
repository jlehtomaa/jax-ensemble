import hydra
import jax
import matplotlib.pyplot as plt
from jax_ensemble.mlp import create_train_state, train_step_fn, pred_fn
from jax_ensemble.utils import create_1d_dataset, set_fig_layout


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg):
    """Train a single MLP on a toy dataset."""

    # INITIALIZE MODEL AND DATA
    # -------------------------
    key = jax.random.PRNGKey(cfg["experiment_seed"])

    # This toy experiments requires exactly three random keys: one for
    # generating the training dataset, one for creating the evaluation dataset,
    # and one for initializing the neural network.
    keys = iter(jax.random.split(key, 3))
    x_train, y_train = create_1d_dataset(
        next(keys), size=cfg["train"]["num_train_data"])

    state = create_train_state(next(keys), cfg["mlp"])
    batch = {"inputs": x_train, "labels": y_train}

    # TRAINING LOOP
    # -------------
    for step in range(cfg["train"]["num_steps"]):
        state, loss = train_step_fn(state, batch)

        if step % 100 == 0:
            print(f"Step: {step}, loss: {loss:.5f}")

    # EVALUATION
    # ----------
    x_eval, _ = create_1d_dataset(
        next(keys), minval=-5, maxval=5, size=cfg["train"]["num_eval_data"], sort=True)
    preds = pred_fn(state, x_eval)

    set_fig_layout(cfg["plots"])
    plt.plot(x_train, y_train, 'kx', label="Training data")
    plt.plot(x_eval, preds, lw=3, label="Prediction")
    plt.title("MLP training")
    plt.legend()
    plt.savefig(cfg["paths"]["results"] + "mlp.png")

if __name__ == "__main__":
    main()
