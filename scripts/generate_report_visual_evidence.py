"""Generate genuine qualitative and failure-case figures for the report.

All panels are produced by the submitted checkpoints on held-out test data.
The companion JSON records the image identifier, condition, metric, route, and
selection rule so the report figures can be audited and regenerated.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader
from torchmetrics.functional.image import structural_similarity_index_measure

ROOT = Path(__file__).resolve().parents[1]
UNIVERSAL_DIR = ROOT / "universal-restoration"
FACE_DIR = ROOT / "face-to-sketch Generator"
for path in (UNIVERSAL_DIR, FACE_DIR):
    sys.path.insert(0, str(path))

from loader import OxfordPetDataset, pet_collate_fn  # noqa: E402
from fs2k_loader import denormalize_image, get_fs2k_dataloaders  # noqa: E402
from models import UNetGenerator  # noqa: E402

FIGURES = ROOT / "report" / "figures"
EVIDENCE_PATH = ROOT / "evaluation_results" / "report_visual_evidence.json"
CLASS_NAMES = ("clean", "noise", "blur", "occlusion")


def make_session(path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def run(session: ort.InferenceSession, batch: np.ndarray) -> List[np.ndarray]:
    return session.run(None, {session.get_inputs()[0].name: batch})


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def per_image_ssim(prediction: np.ndarray, target: torch.Tensor) -> np.ndarray:
    pred = torch.from_numpy(np.clip(prediction, 0.0, 1.0)).float()
    scores = structural_similarity_index_measure(
        pred, target.float(), data_range=1.0, reduction="none"
    )
    return scores.detach().cpu().numpy()


def route(
    corrupted: np.ndarray,
    labels: np.ndarray,
    specialists: Mapping[int, ort.InferenceSession],
) -> np.ndarray:
    output = np.empty_like(corrupted, dtype=np.float32)
    for class_id in range(4):
        mask = labels == class_id
        if not np.any(mask):
            continue
        output[mask] = corrupted[mask] if class_id == 0 else run(specialists[class_id], corrupted[mask])[0]
    return output


def oxford_loader(condition: str, batch_size: int = 32) -> DataLoader:
    images = ROOT / "images"
    if not images.is_dir():
        images = ROOT / "Images"
    dataset = OxfordPetDataset(
        image_dir=str(images),
        manifest_path=str(ROOT / "manifests" / f"test_{condition}.json"),
        mode="test",
        target_size=(128, 128),
        seed=42,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=pet_collate_fn,
    )


def to_hwc(tensor_or_array: Any) -> np.ndarray:
    if isinstance(tensor_or_array, torch.Tensor):
        value = tensor_or_array.detach().cpu().numpy()
    else:
        value = np.asarray(tensor_or_array)
    if value.ndim == 4:
        value = value[0]
    if value.shape[0] in (1, 3):
        value = np.transpose(value, (1, 2, 0))
    return np.clip(value, 0.0, 1.0)


def record(
    title: str,
    image_name: str,
    condition: str,
    source: Any,
    output: Any,
    target: Any,
    ssim: float,
    **extra: Any,
) -> Dict[str, Any]:
    return {
        "title": title,
        "image_name": image_name,
        "condition": condition,
        "source": to_hwc(source),
        "output": to_hwc(output),
        "target": to_hwc(target),
        "ssim": float(ssim),
        **extra,
    }


def scan_oxford(
    sessions: Mapping[str, ort.InferenceSession],
    specialists: Mapping[int, ort.InferenceSession],
    max_samples: int = 384,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    # Representative panels use the same medium-noise test input for a direct
    # comparison of Tasks 1--3.
    rep_batch = next(iter(oxford_loader("salt_and_pepper_medium", batch_size=1)))
    corrupted_t, clean_t, labels_t, metadata = rep_batch
    corrupted = np.ascontiguousarray(corrupted_t.numpy(), dtype=np.float32)
    labels = labels_t.numpy().astype(np.int64)
    logits = run(sessions["classifier"], corrupted)[0]
    predicted = np.argmax(softmax(logits), axis=1).astype(np.int64)
    task1 = run(sessions["universal"], corrupted)[0]
    task2 = route(corrupted, predicted, specialists)
    task3, weights = run(sessions["soft_moe"], corrupted)[:2]
    representatives = [
        record("Task 1 — Universal AE", metadata[0]["image_name"], "noise medium", corrupted[0], task1[0], clean_t[0], per_image_ssim(task1, clean_t)[0]),
        record("Task 2 — Predicted hard route", metadata[0]["image_name"], "noise medium", corrupted[0], task2[0], clean_t[0], per_image_ssim(task2, clean_t)[0], predicted_route=CLASS_NAMES[int(predicted[0])]),
        record("Task 3 — Soft MoE", metadata[0]["image_name"], "noise medium", corrupted[0], task3[0], clean_t[0], per_image_ssim(task3, clean_t)[0], routing_weights=[float(x) for x in weights[0]]),
    ]

    task1_candidates: List[Dict[str, Any]] = []
    task3_candidates: List[Dict[str, Any]] = []
    route_candidates: List[Dict[str, Any]] = []

    for condition in ("rectangular_occlusion_high", "clean", "gaussian_blur_low", "gaussian_blur_medium", "gaussian_blur_high"):
        seen = 0
        for corrupted_t, clean_t, labels_t, metadata in oxford_loader(condition):
            remaining = max_samples - seen
            if remaining <= 0:
                break
            corrupted_t, clean_t, labels_t = corrupted_t[:remaining], clean_t[:remaining], labels_t[:remaining]
            metadata = metadata[:remaining]
            corrupted = np.ascontiguousarray(corrupted_t.numpy(), dtype=np.float32)
            labels = labels_t.numpy().astype(np.int64)

            if condition == "rectangular_occlusion_high":
                task1 = run(sessions["universal"], corrupted)[0]
                task3 = run(sessions["soft_moe"], corrupted)[0]
                task1_scores = per_image_ssim(task1, clean_t)
                task3_scores = per_image_ssim(task3, clean_t)

            logits = run(sessions["classifier"], corrupted)[0]
            probabilities = softmax(logits)
            predicted = np.argmax(probabilities, axis=1).astype(np.int64)
            oracle_out = route(corrupted, labels, specialists)
            predicted_out = route(corrupted, predicted, specialists)
            oracle_scores = per_image_ssim(oracle_out, clean_t)
            predicted_scores = per_image_ssim(predicted_out, clean_t)

            for i, meta in enumerate(metadata):
                base = dict(image_name=meta["image_name"], condition=condition.replace("_", " "))
                if condition == "rectangular_occlusion_high":
                    task1_candidates.append(record("Task 1 — High occlusion", source=corrupted[i], output=task1[i], target=clean_t[i], ssim=task1_scores[i], **base))
                    task3_candidates.append(record("Task 3 — High occlusion", source=corrupted[i], output=task3[i], target=clean_t[i], ssim=task3_scores[i], **base))
                if labels[i] != predicted[i]:
                    route_candidates.append(record(
                        "Task 2 — Routing error",
                        source=corrupted[i], output=predicted_out[i], target=clean_t[i],
                        ssim=predicted_scores[i], oracle_ssim=float(oracle_scores[i]),
                        ssim_loss_vs_oracle=float(oracle_scores[i] - predicted_scores[i]),
                        true_route=CLASS_NAMES[int(labels[i])],
                        predicted_route=CLASS_NAMES[int(predicted[i])],
                        predicted_confidence=float(probabilities[i, predicted[i]]),
                        **base,
                    ))
            seen += len(labels)

    if not route_candidates:
        raise RuntimeError("No genuine Task 2 routing error found in the scanned held-out samples")
    failure1 = min(task1_candidates, key=lambda item: item["ssim"])
    failure2 = max(route_candidates, key=lambda item: item["ssim_loss_vs_oracle"])
    task3_distinct = [item for item in task3_candidates if item["image_name"] != failure1["image_name"]]
    failure3 = min(task3_distinct, key=lambda item: item["ssim"])
    return representatives, [failure1, failure2, failure3]


def scan_task4() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = ROOT / "checkpoints" / "best_cgan_generator.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    model = UNetGenerator(
        in_channels=3,
        out_channels=3,
        base_channels=config.get("base_channels", 64),
        embed_dim=config.get("embed_dim", 16),
        num_styles=3,
        dropout=config.get("dropout", 0.0),
    ).to(device)
    model.load_state_dict(checkpoint.get("generator_state_dict", checkpoint))
    model.eval()

    _, _, loader = get_fs2k_dataloaders(
        base_dir=str(ROOT), batch_size=32, num_workers=0, image_size=128,
        val_ratio=0.15, seed=42, manifest_dir=str(ROOT / "manifests"),
    )
    cases: List[Dict[str, Any]] = []
    offset = 0
    with torch.no_grad():
        for photos, sketches, styles, _metadata in loader:
            fake = model(photos.to(device), styles.to(device)).cpu()
            fake_01 = denormalize_image(fake)
            target_01 = denormalize_image(sketches)
            photo_01 = denormalize_image(photos)
            scores = structural_similarity_index_measure(
                fake_01, target_01, data_range=1.0, reduction="none"
            ).cpu().numpy()
            for i, score in enumerate(scores):
                annotation = loader.dataset.annotations[offset + i]
                cases.append(record(
                    "Task 4 — Face-to-sketch cGAN",
                    image_name=annotation["image_name"],
                    condition=f"FS2K style {int(styles[i]) + 1}",
                    source=photo_01[i], output=fake_01[i], target=target_01[i],
                    ssim=float(score), style=int(styles[i]) + 1,
                ))
            offset += len(styles)

    worst = min(cases, key=lambda item: item["ssim"])
    # A representative sample is the observation nearest the recorded overall
    # test SSIM, avoiding subjective cherry-picking of the best output.
    representative = min(cases, key=lambda item: abs(item["ssim"] - 0.511720788054325))
    return representative, worst


def save_grid(cases: Sequence[Dict[str, Any]], path: Path, heading: str) -> None:
    fig, axes = plt.subplots(4, len(cases), figsize=(3.0 * len(cases), 10.0), dpi=180)
    if len(cases) == 1:
        axes = np.expand_dims(axes, axis=1)
    for column, case in enumerate(cases):
        error = np.mean(np.abs(case["output"] - case["target"]), axis=2)
        axes[0, column].imshow(case["source"])
        axes[1, column].imshow(case["output"])
        axes[2, column].imshow(case["target"])
        axes[3, column].imshow(error, cmap="magma", vmin=0.0, vmax=0.5)
        subtitle = f"{case['condition']}\nSSIM {case['ssim']:.3f}"
        if "predicted_route" in case:
            if "true_route" in case:
                subtitle += f"\nroute: {case['true_route']}→{case['predicted_route']}"
            else:
                subtitle += f"\nroute: {case['predicted_route']}"
        axes[0, column].set_title(f"{case['title']}\n{subtitle}", fontsize=8, fontweight="bold")
        for row in range(4):
            axes[row, column].axis("off")
    row_labels = ("Corrupted / photo input", "Model output", "Held-out target", "Absolute-error map")
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=9, fontweight="bold")
    fig.suptitle(heading, fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout(rect=(0, 0, 1, 0.98))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def serializable(case: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in case.items() if key not in {"source", "output", "target"}}


def main() -> None:
    sessions = {
        "universal": make_session(ROOT / "universal-restoration" / "checkpoints" / "universal_ae.onnx"),
        "classifier": make_session(ROOT / "checkpoints" / "corruption_classifier.onnx"),
        "soft_moe": make_session(ROOT / "checkpoints" / "soft_moe.onnx"),
    }
    specialists = {
        1: make_session(ROOT / "checkpoints" / "specialist_noise.onnx"),
        2: make_session(ROOT / "checkpoints" / "specialist_blur.onnx"),
        3: make_session(ROOT / "checkpoints" / "specialist_occlusion.onnx"),
    }
    representatives, failures = scan_oxford(sessions, specialists)
    task4_representative, task4_failure = scan_task4()
    representatives.append(task4_representative)
    failures.append(task4_failure)

    save_grid(
        representatives,
        FIGURES / "fig_representative_outputs.png",
        "Representative Held-Out Outputs from the Four Implemented Tasks",
    )
    save_grid(
        failures,
        FIGURES / "fig_failure_cases.png",
        "Four Genuine Held-Out Failure Cases Selected by Recorded Metrics",
    )
    evidence = {
        "schema_version": 1,
        "selection": {
            "representative_tasks_1_3": "first sample in official medium-noise manifest",
            "representative_task_4": "held-out sample closest to saved overall Task 4 SSIM",
            "task1_failure": "lowest SSIM among first 384 official high-occlusion samples",
            "task2_failure": "largest predicted-vs-oracle SSIM loss among routing errors in first 384 samples of each scanned blur/occlusion manifest",
            "task3_failure": "lowest SSIM among first 384 official high-occlusion samples, excluding the Task 1 image",
            "task4_failure": "lowest SSIM over all 1,046 official FS2K test pairs",
        },
        "representative_cases": [serializable(item) for item in representatives],
        "failure_cases": [serializable(item) for item in failures],
    }
    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_PATH.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {FIGURES / 'fig_representative_outputs.png'}")
    print(f"Saved {FIGURES / 'fig_failure_cases.png'}")
    print(f"Saved {EVIDENCE_PATH}")


if __name__ == "__main__":
    main()
