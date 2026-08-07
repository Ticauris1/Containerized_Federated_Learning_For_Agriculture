from __future__ import annotations

import config as cfg

from data.preparation import prepare_all_datasets
from experiments.deep_runner import run_deep_experiment
from experiments.definitions import build_experiment_grid
from experiments.traditional_runner import run_traditional_experiment
from reporting.outputs import initialize_summary_csv


def main() -> None:
    cfg.initialize_runtime()
    cfg.validate_configuration()
    cfg.print_configuration_summary()

    prepared_data = prepare_all_datasets()
    experiments = build_experiment_grid()

    initialize_summary_csv(
        path=cfg.SUMMARY_CSV,
        fieldnames=cfg.SUMMARY_FIELDS,
    )

    for experiment in experiments:
        for source_name, source_data in prepared_data.items():
            if cfg.RUN_MODE in {"deep", "both"}:
                run_deep_experiment(
                    experiment=experiment,
                    source_name=source_name,
                    source_data=source_data,
                )

            if cfg.RUN_MODE in {"traditional", "both"}:
                run_traditional_experiment(
                    experiment=experiment,
                    source_name=source_name,
                    source_data=source_data,
                )


if __name__ == "__main__":
    main()