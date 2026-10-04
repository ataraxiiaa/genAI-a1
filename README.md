# OmniRestorer AI — Generative AI Assignment 1

Four image-generation systems in one browser application:

1. a universal multi-corruption denoising autoencoder;
2. a classifier with hard-routed specialist autoencoders;
3. a differentiable soft mixture-of-experts (MoE); and
4. a style-conditioned face-to-sketch cGAN.

The restoration tasks use the Oxford-IIIT Pet dataset. Face-to-sketch uses
FS2K. Training is implemented in PyTorch, deployment inference uses ONNX
Runtime, the API uses FastAPI, and the React/Tailwind frontend is served by
Nginx. The complete application starts with Docker Compose.

## Submission links

- GitHub: <https://github.com/ataraxiiaa/genAI-a1>
- YouTube demonstration: **URL pending upload.** The assignment requires one
  five-to-seven-minute public/unlisted video. Replace `YOUTUBE_URL_PENDING` in
  this README and `report/main.tex` with the final URL before submission.
- AI-use disclosure: [AI_USE_APPENDIX.md](AI_USE_APPENDIX.md)
- Experiment tracking and run IDs: [TRACKING.md](TRACKING.md)

## Saved official-test results

These are the results currently stored in `evaluation_results/`; they are not
estimates copied from documentation.

| System | Evaluation set | SSIM | PSNR |
|---|---:|---:|---:|
| Task 1 universal AE | 10 Oxford conditions (36,690 cases) | 0.7336 | — |
| Task 2 oracle hard route | 10 Oxford conditions | 0.7986 | — |
| Task 2 predicted hard route | 10 Oxford conditions | 0.8010 | — |
| Task 3 soft MoE | 10 Oxford conditions | 0.8062 | — |
| Task 4 cGAN | 1,046 official FS2K test pairs | 0.5117 | 15.48 dB |

For Tasks 1–3, mean PSNR is defined over the nine corrupted conditions because
clean identity reconstruction has zero MSE/infinite PSNR. Those corrupted-only
means are 23.78 dB (Task 1), 25.41 dB (Task 2 oracle), 25.40 dB (Task 2
predicted), and 25.54 dB (Task 3). Full per-condition results and routing counts
are in `evaluation_results/tasks_1_3/`.

## Prerequisites

- Git and [Git LFS](https://git-lfs.com/)
- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- Node.js 20+ for local frontend development
- Docker Desktop with Docker Compose for the submission deployment
- An NVIDIA CUDA environment is optional; every command also supports CPU

## Clone and obtain trained models

The `.pt`, `.pth`, and `.onnx` files are configured for Git LFS. They total
approximately 462 MiB in the current working tree and are not normal Git blobs.

```powershell
git lfs install
git clone https://github.com/ataraxiiaa/genAI-a1.git
Set-Location genAI-a1
git lfs pull
git lfs ls-files
```

The last command should list model objects from `checkpoints/` and
`universal-restoration/checkpoints/`. If it is empty after this repository is
published, the maintainer still needs to commit and push the LFS objects.

## Dataset layout

Datasets are intentionally ignored by Git. Download Oxford-IIIT Pet from the
[official Oxford VGG page](https://robots.ox.ac.uk/~vgg/data/pets/) and FS2K
from the [official FS2K repository](https://github.com/DengPingFan/FS2K). Extract
them so the root looks like this:

```text
genAI-a1/
├── images/                         # Oxford images (*.jpg)
├── annotations/
│   ├── trainval.txt
│   └── test.txt
└── FS2K/
    ├── photo/{photo1,photo2,photo3}/
    ├── sketch/{sketch1,sketch2,sketch3}/
    ├── anno_train.json
    └── anno_test.json
```

Do not regenerate or combine the official test partitions. Tasks 1–3 derive an
80/20 development split with seed 42 and generate corruptions at load time.
Their validation/test corruptions are deterministic manifests. Task 4 derives
a stratified 15% validation subset from `anno_train.json` and preserves all
1,046 entries in `anno_test.json` for final evaluation.

## Python environment

Create the environment and install all dependencies needed for training,
tuning, evaluation, ONNX export, and report figures:

```powershell
uv venv --python 3.12
uv pip install --python .\.venv\Scripts\python.exe -r requirements.txt
```

The root `requirements.txt` includes both `requirements-training.txt` and
`requirements-tracking.txt`. The training file pins the verified
CPU-compatible package versions.
For NVIDIA acceleration, install the matching PyTorch/CUDA build using the
[PyTorch installation selector](https://pytorch.org/get-started/locally/), then
install the remaining requirements. The recorded environment used PyTorch
2.6.0+cu124 and torchvision 0.21.0+cu124.

The inference-only backend has its own smaller pinned file at
`backend/requirements.txt`.

## Training and Optuna requirements

Run commands from the repository root. All tuning and training entry points use
seed 42 by default. On Windows, use `--num_workers 0` if worker processes cause
problems; add `--no_cuda` to force CPU training.

### Task 1 — universal restoration

```powershell
.\.venv\Scripts\python.exe universal-restoration\tune.py --n_trials 10 --trial_epochs 3 --num_workers 0 --seed 42
.\.venv\Scripts\python.exe universal-restoration\train.py --config checkpoints\best_hyperparameters.json --seed 42
.\.venv\Scripts\python.exe universal-restoration\export_onnx.py
```

### Task 2 — classifier and hard-routed specialists

```powershell
.\.venv\Scripts\python.exe specialist-autoencoders\tune_specialists.py --n_trials 15 --trial_epochs 4 --num_workers 0 --seed 42
.\.venv\Scripts\python.exe specialist-autoencoders\train_specialists.py --specialist all --config checkpoints\best_specialist_hyperparameters.json --num_workers 0 --seed 42

.\.venv\Scripts\python.exe specialist-autoencoders\tune_classifier.py --n_trials 15 --trial_epochs 4 --num_workers 0 --seed 42
.\.venv\Scripts\python.exe specialist-autoencoders\train_classifier.py --config checkpoints\best_classifier_hyperparameters.json --num_workers 0 --seed 42

.\.venv\Scripts\python.exe specialist-autoencoders\export_specialists_onnx.py
.\.venv\Scripts\python.exe specialist-autoencoders\export_classifier_onnx.py
```

### Task 3 — soft mixture-of-experts

Task 3 normally initializes from the Task 2 checkpoints. Use `--scratch` only
for a deliberate ablation.

```powershell
.\.venv\Scripts\python.exe MoE-restoration\tune_soft_moe.py --n_trials 15 --trial_epochs 3 --num_workers 0 --seed 42
.\.venv\Scripts\python.exe MoE-restoration\train_soft_moe.py --config checkpoints\best_soft_moe_hyperparameters.json --num_workers 0 --seed 42
.\.venv\Scripts\python.exe MoE-restoration\export_onnx.py
```

### Task 4 — style-conditioned face-to-sketch cGAN

```powershell
.\.venv\Scripts\python.exe "face-to-sketch Generator\tune_cgan.py" --n_trials 10 --epochs_per_trial 5 --num_workers 0 --seed 42
.\.venv\Scripts\python.exe "face-to-sketch Generator\train_cgan.py" --config checkpoints\best_cgan_hyperparameters.json --num_workers 0 --seed 42
.\.venv\Scripts\python.exe "face-to-sketch Generator\export_onnx.py"
```

Optuna SQLite studies, selected parameter JSON files, epoch logs, checkpoints,
and visual samples are retained in the repository. Training is expensive and
is not required merely to reproduce evaluation of the supplied model files.

## Reproduce evaluation

First, a fast smoke test evaluates up to eight images per Oxford manifest:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_tasks_1_3.py --provider cpu --max-samples 8
```

The official Tasks 1–3 command regenerates and validates all ten manifests,
then evaluates every test image. It can take tens of minutes on CPU:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_tasks_1_3.py --provider cpu
```

Task 4 evaluation uses every official FS2K test pair:

```powershell
.\.venv\Scripts\python.exe "face-to-sketch Generator\eval_cgan.py" --no_cuda --num_workers 0 --skip_excel
```

Regenerate the report's representative-output and genuine-failure figures from
the submitted checkpoints and held-out data:

```powershell
.\.venv\Scripts\python.exe scripts\generate_report_visual_evidence.py
```

This writes `fig_representative_outputs.png`, replaces `fig_failure_cases.png`,
and records every sample identifier, metric, and selection rule in
`evaluation_results/report_visual_evidence.json`. Task 4 is scanned across all
1,046 official test pairs, so this command can take several minutes.

To register the saved results in the local MLflow experiment:

```powershell
.\.venv\Scripts\python.exe scripts\track_saved_results_mlflow.py
```

See [TRACKING.md](TRACKING.md) for the recorded run IDs and UI command.

## Run the browser application

After `git lfs pull` and dataset extraction, the required one-command startup
is:

```powershell
docker compose up --build
```

Open <http://localhost:5173>. The FastAPI service is available at
<http://localhost:8000>, with interactive API documentation at
<http://localhost:8000/docs>.

Stop the stack with:

```powershell
docker compose down
```

For frontend-only development:

```powershell
Set-Location frontend
npm ci
npm run dev
```

## Repository map

| Path | Purpose |
|---|---|
| `universal-restoration/` | Task 1 architecture, loader, tuning, training, and export |
| `specialist-autoencoders/` | Task 2 specialists, classifier, hard router, tuning, and export |
| `MoE-restoration/` | Task 3 soft gating, loss, training, analysis, and export |
| `face-to-sketch Generator/` | Task 4 FS2K loader, cGAN models, tuning, training, evaluation, and export |
| `scripts/evaluate_tasks_1_3.py` | Ten-manifest reproducible Tasks 1–3 evaluator |
| `scripts/track_saved_results_mlflow.py` | Registers saved Tasks 1–4 results in MLflow |
| `backend/`, `frontend/` | FastAPI/ONNX backend and React/Tailwind frontend |
| `evaluation_results/` | Machine-readable official-test results and MLflow store |
| `report/` | IEEE LaTeX report and generated figures |

## Reproducibility notes

- The canonical seed is 42.
- Official Oxford test output records SHA-256 hashes for every ONNX model.
- `--max-samples` is smoke-test only; never report it as an official result.
- Report values must be regenerated from the JSON/CSV artifacts, not copied
  from terminal output or estimated from plots.
- Git LFS tracks trained `.pt`, `.pth`, and `.onnx` assets; datasets remain
  excluded because of their size and distribution terms.
