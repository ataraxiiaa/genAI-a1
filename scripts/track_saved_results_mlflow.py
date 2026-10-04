"""Log the repository's saved evaluation results to a local MLflow experiment.

This command does not rerun inference. It records the immutable result artifacts,
model hashes, environment metadata, aggregate metrics, and per-condition metrics
that were produced by the reproducible Task 1--4 evaluations.

Run from any directory:

    python scripts/track_saved_results_mlflow.py

The command writes the current run IDs and local UI links to
``evaluation_results/mlflow_run_index.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping

import mlflow
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "evaluation_results"
TASKS_1_3_JSON = RESULTS_DIR / "tasks_1_3" / "full_official_test_results.json"
TASK4_JSON = RESULTS_DIR / "task4_test_metrics.json"
DEFAULT_DB = RESULTS_DIR / "mlflow.db"
DEFAULT_ARTIFACT_ROOT = RESULTS_DIR / "mlflow_artifacts"
DEFAULT_INDEX = RESULTS_DIR / "mlflow_run_index.json"
EXPERIMENT_NAME = "genai-assignment-official-evaluation"


def load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required result artifact not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sqlite_uri(path: Path) -> str:
    return f"sqlite:///{path.resolve().as_posix()}"


def log_metrics(metrics: Mapping[str, Any]) -> None:
    numeric = {
        key: float(value)
        for key, value in metrics.items()
        if value is not None and isinstance(value, (int, float, np.number))
    }
    if numeric:
        mlflow.log_metrics(numeric)


def corrupted_only_means(data: Mapping[str, Any], system: str) -> Dict[str, float]:
    rows = [
        manifest["systems"][system]
        for name, manifest in data["results_by_manifest"].items()
        if name != "clean"
    ]
    total = sum(row["samples"] for row in rows)
    return {
        key: sum(row[key] * row["samples"] for row in rows) / total
        for key in ("mean_l1", "mean_ssim", "mean_psnr_db")
    }


def log_common_restoration_metrics(prefix: str, values: Mapping[str, Any]) -> None:
    log_metrics(
        {
            f"{prefix}.samples": values["samples"],
            f"{prefix}.mean_l1": values["mean_l1"],
            f"{prefix}.mean_ssim": values["mean_ssim"],
            f"{prefix}.mean_psnr_db": values["mean_psnr_db"],
            f"{prefix}.inference_ms_per_image": values["inference_ms_per_image"],
        }
    )


def log_source(path: Path, artifact_path: str = "source") -> None:
    mlflow.log_artifact(str(path), artifact_path=artifact_path)


def run_link(base_url: str, experiment_id: str, run_id: str) -> str:
    return f"{base_url.rstrip('/')}/#/experiments/{experiment_id}/runs/{run_id}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Track saved Tasks 1--4 results with MLflow")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="Local MLflow SQLite database")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
        help="Local MLflow artifact directory",
    )
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX, help="Output run-index JSON")
    parser.add_argument("--experiment", default=EXPERIMENT_NAME, help="MLflow experiment name")
    parser.add_argument("--ui-url", default="http://127.0.0.1:5000", help="Documented MLflow UI base URL")
    args = parser.parse_args()

    task13 = load_json(TASKS_1_3_JSON)
    task4 = load_json(TASK4_JSON)
    args.db.parent.mkdir(parents=True, exist_ok=True)
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    args.index.parent.mkdir(parents=True, exist_ok=True)

    tracking_uri = sqlite_uri(args.db)
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(args.experiment)
    if experiment is None:
        experiment_id = client.create_experiment(
            args.experiment,
            artifact_location=args.artifact_root.resolve().as_uri(),
            tags={"repository": "genAI-a1", "scope": "official saved evaluation"},
        )
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(args.experiment)

    runs: Dict[str, Dict[str, str]] = {}
    with mlflow.start_run(
        run_name="official-evaluation-tasks-1-4",
        tags={
            "run_kind": "saved_artifact_evaluation",
            "task_scope": "1,2,3,4",
            "source_created_at_utc": task13["created_at_utc"],
        },
    ) as parent:
        parent_id = parent.info.run_id
        runs["parent"] = {
            "run_id": parent_id,
            "run_name": "official-evaluation-tasks-1-4",
            "link": run_link(args.ui_url, experiment_id, parent_id),
        }
        mlflow.log_params(
            {
                "seed": task13["seed"],
                "batch_size": task13["batch_size"],
                "onnx_provider": ",".join(task13["onnx_providers"]),
                "task13_cases": task13["overall"]["task1_universal"]["samples"],
                "task4_pairs": task4["overall"]["total_test_pairs"],
                "task4_checkpoint_epoch": task4["checkpoint_epoch"],
            }
        )
        log_metrics(
            {
                "task1.overall_ssim": task13["overall"]["task1_universal"]["mean_ssim"],
                "task2.predicted.overall_ssim": task13["overall"]["task2_hard_predicted"]["mean_ssim"],
                "task2.routing_accuracy": task13["overall"]["task2_hard_predicted"]["routing_accuracy"],
                "task3.overall_ssim": task13["overall"]["task3_soft_moe"]["mean_ssim"],
                "task4.test_ssim": task4["overall"]["test_ssim"],
            }
        )
        for model_name, metadata in task13["models"].items():
            mlflow.set_tag(f"model_sha256.{model_name}", metadata["sha256"])
        log_source(TASKS_1_3_JSON)
        log_source(TASK4_JSON)
        log_source(ROOT / "report" / "main.tex", artifact_path="report")

        with mlflow.start_run(run_name="task1-universal", nested=True) as child:
            run_id = child.info.run_id
            runs["task1"] = {
                "run_id": run_id,
                "run_name": "task1-universal",
                "link": run_link(args.ui_url, experiment_id, run_id),
            }
            overall = task13["overall"]["task1_universal"]
            corrupted_samples = sum(
                manifest["systems"]["task1_universal"]["samples"]
                for name, manifest in task13["results_by_manifest"].items()
                if name != "clean"
            )
            log_common_restoration_metrics("overall", overall)
            log_common_restoration_metrics("corrupted_only", {
                **corrupted_only_means(task13, "task1_universal"),
                "samples": corrupted_samples,
                "inference_ms_per_image": overall["inference_ms_per_image"],
            })
            for name, manifest in task13["results_by_manifest"].items():
                values = manifest["systems"]["task1_universal"]
                log_metrics({
                    f"condition.{name}.l1": values["mean_l1"],
                    f"condition.{name}.ssim": values["mean_ssim"],
                    f"condition.{name}.psnr_db": values["mean_psnr_db"],
                })
            log_source(RESULTS_DIR / "tasks_1_3" / "task1_severity_breakdown.csv")

        with mlflow.start_run(run_name="task2-hard-routing", nested=True) as child:
            run_id = child.info.run_id
            runs["task2"] = {
                "run_id": run_id,
                "run_name": "task2-hard-routing",
                "link": run_link(args.ui_url, experiment_id, run_id),
            }
            for route in ("task2_hard_oracle", "task2_hard_predicted"):
                log_common_restoration_metrics(route, task13["overall"][route])
                corrupt = corrupted_only_means(task13, route)
                log_metrics({f"{route}.corrupted_only.{key}": value for key, value in corrupt.items()})
            predicted = task13["overall"]["task2_hard_predicted"]
            log_metrics({"predicted.routing_accuracy": predicted["routing_accuracy"]})
            mlflow.log_dict(predicted["confusion_matrix"], "derived/official_test_confusion_matrix.json")
            log_source(RESULTS_DIR / "tasks_1_3" / "full_official_test_results.csv")
            confusion_figure = ROOT / "report" / "figures" / "fig_confusion_matrix.png"
            if confusion_figure.is_file():
                log_source(confusion_figure, artifact_path="figures")

        with mlflow.start_run(run_name="task3-soft-moe", nested=True) as child:
            run_id = child.info.run_id
            runs["task3"] = {
                "run_id": run_id,
                "run_name": "task3-soft-moe",
                "link": run_link(args.ui_url, experiment_id, run_id),
            }
            overall = task13["overall"]["task3_soft_moe"]
            log_common_restoration_metrics("overall", overall)
            log_metrics({
                f"overall.routing_weight.{name}": value
                for name, value in zip(("clean", "noise", "blur", "occlusion"), overall["mean_routing_weights"])
            })
            corrupt = corrupted_only_means(task13, "task3_soft_moe")
            log_metrics({f"corrupted_only.{key}": value for key, value in corrupt.items()})
            for name, manifest in task13["results_by_manifest"].items():
                values = manifest["systems"]["task3_soft_moe"]
                log_metrics({
                    f"condition.{name}.l1": values["mean_l1"],
                    f"condition.{name}.ssim": values["mean_ssim"],
                    f"condition.{name}.psnr_db": values["mean_psnr_db"],
                    **{
                        f"condition.{name}.weight.{branch}": weight
                        for branch, weight in zip(
                            ("clean", "noise", "blur", "occlusion"), values["mean_routing_weights"]
                        )
                    },
                })
            log_source(RESULTS_DIR / "tasks_1_3" / "task3_routing_breakdown.csv")
            gating_figure = ROOT / "report" / "figures" / "fig_moe_gating_distribution.png"
            if gating_figure.is_file():
                log_source(gating_figure, artifact_path="figures")

        with mlflow.start_run(run_name="task4-face-to-sketch", nested=True) as child:
            run_id = child.info.run_id
            runs["task4"] = {
                "run_id": run_id,
                "run_name": "task4-face-to-sketch",
                "link": run_link(args.ui_url, experiment_id, run_id),
            }
            mlflow.log_params(
                {
                    "checkpoint": task4["checkpoint"],
                    "checkpoint_epoch": task4["checkpoint_epoch"],
                    "split": task4["split"],
                }
            )
            log_metrics(task4["overall"])
            for style, values in task4["per_style"].items():
                slug = style.lower().replace(" ", "_")
                log_metrics({f"{slug}.{key}": value for key, value in values.items()})
            log_source(TASK4_JSON)
            grid = ROOT / "face-to-sketch Generator" / "visual_samples" / "test_qualitative_grid.png"
            if grid.is_file():
                log_source(grid, artifact_path="figures")

        index = {
            "schema_version": 1,
            "tracker": "mlflow",
            "mlflow_version": mlflow.__version__,
            "experiment_name": args.experiment,
            "experiment_id": experiment_id,
            "tracking_uri": tracking_uri,
            "artifact_root": args.artifact_root.resolve().as_uri(),
            "ui_base_url": args.ui_url.rstrip("/"),
            "runs": runs,
        }
        with args.index.open("w", encoding="utf-8") as handle:
            json.dump(index, handle, indent=2)
        log_source(args.index, artifact_path="tracking")

    print(json.dumps(index, indent=2))
    print("\nStart the local UI with:")
    print(
        f'mlflow ui --backend-store-uri "{tracking_uri}" '
        f'--default-artifact-root "{args.artifact_root.resolve().as_uri()}" '
        '--host 127.0.0.1 --port 5000'
    )


if __name__ == "__main__":
    main()
