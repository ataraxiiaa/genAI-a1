"""Reproducible official-test evaluation for assignment Tasks 1, 2, and 3.

The command regenerates the ten deterministic Oxford-IIIT Pet test manifests,
then evaluates the deployed ONNX models on exactly the same inputs:

    python scripts/evaluate_tasks_1_3.py

Use ``--manifests-only`` to create/validate the manifests without inference and
``--max-samples N`` for a clearly labelled smoke run. Full evaluation is the
default; the official test set is never used for training or model selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader
from torchmetrics.functional.image import structural_similarity_index_measure


ROOT = Path(__file__).resolve().parents[1]
UNIVERSAL_DIR = ROOT / "universal-restoration"
sys.path.insert(0, str(UNIVERSAL_DIR))

from loader import (  # noqa: E402
    OxfordPetDataset,
    generate_test_manifests,
    get_dataset_splits,
    pet_collate_fn,
)


EXPECTED_MANIFESTS: Tuple[str, ...] = (
    "clean",
    "salt_and_pepper_low",
    "salt_and_pepper_medium",
    "salt_and_pepper_high",
    "gaussian_blur_low",
    "gaussian_blur_medium",
    "gaussian_blur_high",
    "rectangular_occlusion_low",
    "rectangular_occlusion_medium",
    "rectangular_occlusion_high",
)

SYSTEM_NAMES: Tuple[str, ...] = (
    "task1_universal",
    "task2_hard_oracle",
    "task2_hard_predicted",
    "task3_soft_moe",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_providers(requested: str) -> List[str]:
    available = ort.get_available_providers()
    if requested == "cpu":
        return ["CPUExecutionProvider"]
    if requested == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "CUDAExecutionProvider was requested but is unavailable. "
                f"Available providers: {available}"
            )
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def make_session(path: Path, providers: Sequence[str]) -> ort.InferenceSession:
    if not path.is_file():
        raise FileNotFoundError(f"Required ONNX model not found: {path}")
    return ort.InferenceSession(str(path), providers=list(providers))


def run_single_input(session: ort.InferenceSession, batch: np.ndarray) -> List[np.ndarray]:
    input_name = session.get_inputs()[0].name
    return session.run(None, {input_name: batch})


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def route_batch(
    corrupted: np.ndarray,
    route_labels: np.ndarray,
    specialists: Mapping[int, ort.InferenceSession],
) -> np.ndarray:
    restored = np.empty_like(corrupted, dtype=np.float32)
    for class_id in range(4):
        mask = route_labels == class_id
        if not np.any(mask):
            continue
        if class_id == 0:
            restored[mask] = corrupted[mask]
            continue
        session = specialists[class_id]
        restored[mask] = run_single_input(session, corrupted[mask])[0]
    return restored


@dataclass
class MetricAccumulator:
    count: int = 0
    l1_sum: float = 0.0
    ssim_sum: float = 0.0
    psnr_sum: float = 0.0
    exact_match_count: int = 0
    inference_ms: float = 0.0

    def update(self, prediction: np.ndarray, target: torch.Tensor, inference_ms: float) -> None:
        pred = torch.from_numpy(np.clip(prediction, 0.0, 1.0)).float()
        truth = target.detach().cpu().float()
        batch_size = int(pred.shape[0])

        l1 = torch.mean(torch.abs(pred - truth), dim=(1, 2, 3))
        mse = torch.mean((pred - truth) ** 2, dim=(1, 2, 3))
        exact = mse == 0
        # Exact identity-bypass matches are mathematically infinite PSNR. Cap at
        # 100 dB so JSON/CSV remain standards-compliant and machine-readable.
        psnr = 10.0 * torch.log10(1.0 / torch.clamp(mse, min=1e-10))
        ssim = structural_similarity_index_measure(
            pred,
            truth,
            data_range=1.0,
            reduction="none",
        )

        self.count += batch_size
        self.l1_sum += float(l1.sum().item())
        self.ssim_sum += float(ssim.sum().item())
        self.psnr_sum += float(psnr.sum().item())
        self.exact_match_count += int(exact.sum().item())
        self.inference_ms += float(inference_ms)

    def merge(self, other: "MetricAccumulator") -> None:
        self.count += other.count
        self.l1_sum += other.l1_sum
        self.ssim_sum += other.ssim_sum
        self.psnr_sum += other.psnr_sum
        self.exact_match_count += other.exact_match_count
        self.inference_ms += other.inference_ms

    def summary(self) -> Dict[str, Any]:
        if self.count == 0:
            raise RuntimeError("Cannot summarize an empty metric accumulator")
        return {
            "samples": self.count,
            "mean_l1": self.l1_sum / self.count,
            "mean_ssim": self.ssim_sum / self.count,
            "mean_psnr_db": self.psnr_sum / self.count,
            "exact_match_samples": self.exact_match_count,
            "psnr_cap_db": 100.0,
            "inference_ms_total": self.inference_ms,
            "inference_ms_per_image": self.inference_ms / self.count,
        }


@dataclass
class RoutingAccumulator:
    count: int = 0
    correct: int = 0
    confusion: np.ndarray = field(default_factory=lambda: np.zeros((4, 4), dtype=np.int64))
    weight_sum: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float64))

    def update_predictions(self, labels: np.ndarray, predictions: np.ndarray) -> None:
        self.count += int(labels.size)
        self.correct += int(np.sum(labels == predictions))
        for truth, pred in zip(labels.tolist(), predictions.tolist()):
            self.confusion[int(truth), int(pred)] += 1

    def update_weights(self, weights: np.ndarray) -> None:
        self.weight_sum += weights.sum(axis=0, dtype=np.float64)

    def merge(self, other: "RoutingAccumulator") -> None:
        self.count += other.count
        self.correct += other.correct
        self.confusion += other.confusion
        self.weight_sum += other.weight_sum


def validate_manifest(path: Path, expected_count: int, expected_key: str) -> Dict[str, Any]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if len(records) != expected_count:
        raise ValueError(f"{path.name}: expected {expected_count} records, found {len(records)}")

    if expected_key == "clean":
        expected_type, expected_severity = "clean", None
    else:
        expected_type, expected_severity = expected_key.rsplit("_", 1)

    seen_names = set()
    for record in records:
        if record.get("corruption_type") != expected_type:
            raise ValueError(f"{path.name}: unexpected corruption record {record}")
        if record.get("severity_level") != expected_severity:
            raise ValueError(f"{path.name}: unexpected severity record {record}")
        name = record.get("image_name")
        if name in seen_names:
            raise ValueError(f"{path.name}: duplicate image_name {name}")
        seen_names.add(name)

    return {
        "path": str(path.relative_to(ROOT)),
        "records": len(records),
        "corruption_type": expected_type,
        "severity": expected_severity,
        "sha256": sha256_file(path),
    }


def generate_and_validate_manifests(manifest_dir: Path, seed: int) -> Dict[str, Dict[str, Any]]:
    _, _, test_names = get_dataset_splits(str(ROOT), val_ratio=0.2, seed=seed)
    generated = generate_test_manifests(test_names, str(manifest_dir), seed=seed)
    if tuple(generated.keys()) != EXPECTED_MANIFESTS:
        raise RuntimeError(
            f"Manifest generator returned {tuple(generated.keys())}; expected {EXPECTED_MANIFESTS}"
        )

    metadata = {}
    for key in EXPECTED_MANIFESTS:
        metadata[key] = validate_manifest(Path(generated[key]), len(test_names), key)
    return metadata


def limited_batches(
    dataloader: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]],
    max_samples: Optional[int],
) -> Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    consumed = 0
    for corrupted, clean, labels, _ in dataloader:
        if max_samples is not None:
            remaining = max_samples - consumed
            if remaining <= 0:
                break
            corrupted = corrupted[:remaining]
            clean = clean[:remaining]
            labels = labels[:remaining]
        if len(labels) == 0:
            break
        consumed += len(labels)
        yield corrupted, clean, labels


def evaluate_manifest(
    manifest_path: Path,
    images_dir: Path,
    batch_size: int,
    num_workers: int,
    max_samples: Optional[int],
    universal: ort.InferenceSession,
    classifier: ort.InferenceSession,
    specialists: Mapping[int, ort.InferenceSession],
    soft_moe: ort.InferenceSession,
) -> Tuple[Dict[str, Dict[str, Any]], RoutingAccumulator]:
    dataset = OxfordPetDataset(
        image_dir=str(images_dir),
        manifest_path=str(manifest_path),
        mode="test",
        target_size=(128, 128),
        seed=42,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=pet_collate_fn,
        drop_last=False,
    )

    metrics = {name: MetricAccumulator() for name in SYSTEM_NAMES}
    routing = RoutingAccumulator()

    for corrupted_t, clean_t, labels_t in limited_batches(dataloader, max_samples):
        corrupted = np.ascontiguousarray(corrupted_t.numpy(), dtype=np.float32)
        labels = labels_t.numpy().astype(np.int64, copy=False)

        start = time.perf_counter()
        universal_out = run_single_input(universal, corrupted)[0]
        universal_ms = (time.perf_counter() - start) * 1000.0
        metrics["task1_universal"].update(universal_out, clean_t, universal_ms)

        start = time.perf_counter()
        logits = run_single_input(classifier, corrupted)[0]
        classifier_ms = (time.perf_counter() - start) * 1000.0
        probabilities = softmax(logits)
        predictions = np.argmax(probabilities, axis=1).astype(np.int64)
        routing.update_predictions(labels, predictions)

        start = time.perf_counter()
        oracle_out = route_batch(corrupted, labels, specialists)
        oracle_ms = (time.perf_counter() - start) * 1000.0
        metrics["task2_hard_oracle"].update(oracle_out, clean_t, oracle_ms)

        start = time.perf_counter()
        predicted_out = route_batch(corrupted, predictions, specialists)
        predicted_expert_ms = (time.perf_counter() - start) * 1000.0
        metrics["task2_hard_predicted"].update(
            predicted_out,
            clean_t,
            classifier_ms + predicted_expert_ms,
        )

        start = time.perf_counter()
        soft_outputs = run_single_input(soft_moe, corrupted)
        soft_ms = (time.perf_counter() - start) * 1000.0
        soft_out, weights = soft_outputs[0], soft_outputs[1]
        metrics["task3_soft_moe"].update(soft_out, clean_t, soft_ms)
        routing.update_weights(weights)

    result = {name: accumulator.summary() for name, accumulator in metrics.items()}
    result["task2_hard_predicted"]["routing_accuracy"] = routing.correct / routing.count
    result["task2_hard_predicted"]["confusion_matrix"] = routing.confusion.tolist()
    result["task3_soft_moe"]["mean_routing_weights"] = (
        routing.weight_sum / routing.count
    ).tolist()
    return result, routing


def write_csv(path: Path, manifest_results: Mapping[str, Any], overall: Mapping[str, Any]) -> None:
    fieldnames = [
        "scope",
        "manifest",
        "corruption_type",
        "severity",
        "system",
        "samples",
        "mean_l1",
        "mean_ssim",
        "mean_psnr_db",
        "inference_ms_per_image",
        "routing_accuracy",
        "weight_clean",
        "weight_noise",
        "weight_blur",
        "weight_occlusion",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for manifest_key, payload in manifest_results.items():
            for system, values in payload["systems"].items():
                weights = values.get("mean_routing_weights", [None] * 4)
                writer.writerow({
                    "scope": "manifest",
                    "manifest": manifest_key,
                    "corruption_type": payload["corruption_type"],
                    "severity": payload["severity"] or "none",
                    "system": system,
                    "samples": values["samples"],
                    "mean_l1": values["mean_l1"],
                    "mean_ssim": values["mean_ssim"],
                    "mean_psnr_db": values["mean_psnr_db"],
                    "inference_ms_per_image": values["inference_ms_per_image"],
                    "routing_accuracy": values.get("routing_accuracy"),
                    "weight_clean": weights[0],
                    "weight_noise": weights[1],
                    "weight_blur": weights[2],
                    "weight_occlusion": weights[3],
                })
        for system, values in overall.items():
            weights = values.get("mean_routing_weights", [None] * 4)
            writer.writerow({
                "scope": "overall",
                "manifest": "all_official_conditions",
                "corruption_type": "all",
                "severity": "all",
                "system": system,
                "samples": values["samples"],
                "mean_l1": values["mean_l1"],
                "mean_ssim": values["mean_ssim"],
                "mean_psnr_db": values["mean_psnr_db"],
                "inference_ms_per_image": values["inference_ms_per_image"],
                "routing_accuracy": values.get("routing_accuracy"),
                "weight_clean": weights[0],
                "weight_noise": weights[1],
                "weight_blur": weights[2],
                "weight_occlusion": weights[3],
            })


def write_breakdown_files(output_dir: Path, manifest_results: Mapping[str, Any]) -> Tuple[Path, Path, Path]:
    """Write report-ready Task 1 severity and Task 3 routing breakdowns."""
    task1_path = output_dir / "task1_severity_breakdown.csv"
    with task1_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "condition",
            "corruption_type",
            "severity",
            "samples",
            "mean_l1",
            "mean_ssim",
            "mean_psnr_db",
            "inference_ms_per_image",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for key in EXPECTED_MANIFESTS:
            payload = manifest_results[key]
            metrics = payload["systems"]["task1_universal"]
            writer.writerow({
                "condition": key,
                "corruption_type": payload["corruption_type"],
                "severity": payload["severity"] or "none",
                **{name: metrics[name] for name in fieldnames[3:]},
            })

    task3_path = output_dir / "task3_routing_breakdown.csv"
    branch_names = ("clean_identity", "noise_expert", "blur_expert", "occlusion_expert")
    with task3_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "condition",
            "corruption_type",
            "severity",
            "samples",
            "weight_clean",
            "weight_noise",
            "weight_blur",
            "weight_occlusion",
            "dominant_branch",
            "dominant_weight",
            "mean_l1",
            "mean_ssim",
            "mean_psnr_db",
            "inference_ms_per_image",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for key in EXPECTED_MANIFESTS:
            payload = manifest_results[key]
            metrics = payload["systems"]["task3_soft_moe"]
            weights = metrics["mean_routing_weights"]
            dominant_index = int(np.argmax(weights))
            writer.writerow({
                "condition": key,
                "corruption_type": payload["corruption_type"],
                "severity": payload["severity"] or "none",
                "samples": metrics["samples"],
                "weight_clean": weights[0],
                "weight_noise": weights[1],
                "weight_blur": weights[2],
                "weight_occlusion": weights[3],
                "dominant_branch": branch_names[dominant_index],
                "dominant_weight": weights[dominant_index],
                "mean_l1": metrics["mean_l1"],
                "mean_ssim": metrics["mean_ssim"],
                "mean_psnr_db": metrics["mean_psnr_db"],
                "inference_ms_per_image": metrics["inference_ms_per_image"],
            })

    markdown_path = output_dir / "official_test_breakdowns.md"
    lines = [
        "# Official Oxford-IIIT Pet Test Breakdowns",
        "",
        "Each row contains all 3,669 images from the untouched official test split. "
        "Corruptions come from the deterministic manifests generated with seed 42.",
        "",
        "## Task 1: Universal Restoration by Severity",
        "",
        "| Condition | Severity | L1 | SSIM | PSNR (dB) |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in EXPECTED_MANIFESTS:
        payload = manifest_results[key]
        metrics = payload["systems"]["task1_universal"]
        lines.append(
            f"| {payload['corruption_type']} | {payload['severity'] or 'none'} | "
            f"{metrics['mean_l1']:.4f} | {metrics['mean_ssim']:.4f} | "
            f"{metrics['mean_psnr_db']:.2f} |"
        )

    lines.extend([
        "",
        "## Task 3: Soft-MoE Routing by Severity",
        "",
        "| Condition | Severity | Clean | Noise | Blur | Occlusion | Dominant branch | SSIM | PSNR (dB) |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|",
    ])
    for key in EXPECTED_MANIFESTS:
        payload = manifest_results[key]
        metrics = payload["systems"]["task3_soft_moe"]
        weights = metrics["mean_routing_weights"]
        dominant_index = int(np.argmax(weights))
        lines.append(
            f"| {payload['corruption_type']} | {payload['severity'] or 'none'} | "
            f"{weights[0]:.4f} | {weights[1]:.4f} | {weights[2]:.4f} | {weights[3]:.4f} | "
            f"{branch_names[dominant_index]} | {metrics['mean_ssim']:.4f} | "
            f"{metrics['mean_psnr_db']:.2f} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return task1_path, task3_path, markdown_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate official Oxford test manifests and evaluate Tasks 1-3 with ONNX Runtime."
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--provider", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--manifest-dir", type=Path, default=ROOT / "manifests")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "evaluation_results" / "tasks_1_3")
    parser.add_argument("--manifests-only", action="store_true")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Evaluate only the first N images per manifest (smoke testing only).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be positive")

    seed_everything(args.seed)
    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_metadata = generate_and_validate_manifests(args.manifest_dir.resolve(), args.seed)
    print(f"Validated {len(manifest_metadata)} manifests with "
          f"{next(iter(manifest_metadata.values()))['records']} records each.")

    if args.manifests_only:
        print("Manifest-only run complete.")
        return

    providers = select_providers(args.provider)
    model_paths = {
        "task1_universal": ROOT / "universal-restoration" / "checkpoints" / "universal_ae.onnx",
        "task2_classifier": ROOT / "checkpoints" / "corruption_classifier.onnx",
        "task2_noise": ROOT / "checkpoints" / "specialist_noise.onnx",
        "task2_blur": ROOT / "checkpoints" / "specialist_blur.onnx",
        "task2_occlusion": ROOT / "checkpoints" / "specialist_occlusion.onnx",
        "task3_soft_moe": ROOT / "checkpoints" / "soft_moe.onnx",
    }
    sessions = {name: make_session(path, providers) for name, path in model_paths.items()}
    specialists = {
        1: sessions["task2_noise"],
        2: sessions["task2_blur"],
        3: sessions["task2_occlusion"],
    }

    images_dir = ROOT / "images"
    if not images_dir.is_dir():
        images_dir = ROOT / "Images"
    if not images_dir.is_dir():
        raise FileNotFoundError("Oxford-IIIT Pet images directory was not found")

    manifest_results: Dict[str, Any] = {}
    overall_metrics = {name: MetricAccumulator() for name in SYSTEM_NAMES}
    overall_routing = RoutingAccumulator()

    for index, key in enumerate(EXPECTED_MANIFESTS, start=1):
        info = manifest_metadata[key]
        manifest_path = ROOT / info["path"]
        print(f"[{index:02d}/10] Evaluating {key} ...", flush=True)
        systems, routing = evaluate_manifest(
            manifest_path=manifest_path,
            images_dir=images_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_samples=args.max_samples,
            universal=sessions["task1_universal"],
            classifier=sessions["task2_classifier"],
            specialists=specialists,
            soft_moe=sessions["task3_soft_moe"],
        )
        manifest_results[key] = {
            "corruption_type": info["corruption_type"],
            "severity": info["severity"],
            "systems": systems,
        }

        # Reconstruct additive accumulators from the summaries so all overall
        # figures use exactly the same per-image weighting as manifest results.
        for system, summary in systems.items():
            count = summary["samples"]
            accumulator = overall_metrics[system]
            accumulator.count += count
            accumulator.l1_sum += summary["mean_l1"] * count
            accumulator.ssim_sum += summary["mean_ssim"] * count
            accumulator.psnr_sum += summary["mean_psnr_db"] * count
            accumulator.exact_match_count += summary["exact_match_samples"]
            accumulator.inference_ms += summary["inference_ms_total"]
        overall_routing.merge(routing)

        print(
            "    "
            + " | ".join(
                f"{name}: SSIM={values['mean_ssim']:.4f}, PSNR={values['mean_psnr_db']:.2f}"
                for name, values in systems.items()
            ),
            flush=True,
        )

    overall = {name: accumulator.summary() for name, accumulator in overall_metrics.items()}
    overall["task2_hard_predicted"]["routing_accuracy"] = (
        overall_routing.correct / overall_routing.count
    )
    overall["task2_hard_predicted"]["confusion_matrix"] = overall_routing.confusion.tolist()
    overall["task3_soft_moe"]["mean_routing_weights"] = (
        overall_routing.weight_sum / overall_routing.count
    ).tolist()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_kind = "smoke" if args.max_samples is not None else "full_official_test"
    payload = {
        "schema_version": 1,
        "run_kind": run_kind,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "max_samples_per_manifest": args.max_samples,
        "onnx_providers": providers,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "onnxruntime": ort.__version__,
            "platform": platform.platform(),
        },
        "models": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}
            for name, path in model_paths.items()
        },
        "manifests": manifest_metadata,
        "results_by_manifest": manifest_results,
        "overall": overall,
    }

    json_path = args.output_dir / f"{run_kind}_results.json"
    csv_path = args.output_dir / f"{run_kind}_results.csv"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(csv_path, manifest_results, overall)
    task1_path, task3_path, markdown_path = write_breakdown_files(
        args.output_dir,
        manifest_results,
    )

    print(f"Saved JSON results: {json_path}")
    print(f"Saved CSV results:  {csv_path}")
    print(f"Saved Task 1 breakdown: {task1_path}")
    print(f"Saved Task 3 breakdown: {task3_path}")
    print(f"Saved report tables:    {markdown_path}")
    if args.max_samples is not None:
        print("NOTE: This is a smoke run and must not be reported as full test-set performance.")


if __name__ == "__main__":
    main()
