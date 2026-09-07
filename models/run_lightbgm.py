"""
run_lightbgm.py
===================
Small executable script that runs the existing LightGBM implementation
through the shared experiment infrastructure for both dataset layers,
then prints a comparison table.

No new logic lives here -- this only wires together ExperimentConfig,
experiment_runner.run_experiment(), and LightGBMModel.
"""

from experiment_config import ExperimentConfig, Layer
from experiment_runner import run_experiment, results_to_dataframe
from model_implementations.lightbgm_model import LightGBMModel

RANDOM_STATE = 42


def main() -> None:
    results = []

    # 1. LightGBM on Layer 1
    config_layer1 = ExperimentConfig(
        layer=Layer.LAYER1,
        random_state=RANDOM_STATE,
        experiment_name="lightbgm_layer1_baseline",
    )
    model_layer1 = LightGBMModel(random_state=RANDOM_STATE)
    result_layer1 = run_experiment(
        config=config_layer1,
        model=model_layer1,
        model_name="lightbgm",
        save_model_flag=True,
    )
    results.append(result_layer1)

    # 2. LightGBM on Layer 2 (p11)
    config_layer2 = ExperimentConfig(
        layer=Layer.LAYER2_P11,
        random_state=RANDOM_STATE,
        experiment_name="lightbgm_layer2_p11_baseline",
    )
    model_layer2 = LightGBMModel(random_state=RANDOM_STATE)
    result_layer2 = run_experiment(
        config=config_layer2,
        model=model_layer2,
        model_name="lightbgm",
        save_model_flag=True,
    )
    results.append(result_layer2)

    # Comparison table
    comparison_df = results_to_dataframe(results)
    print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    main()