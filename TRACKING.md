# MLflow experiment tracking

The saved Tasks 1–4 evaluation results are registered in a repository-local
MLflow 3.16.1 experiment. No external account or cloud service is required.

## Recorded experiment

- Experiment: `genai-assignment-official-evaluation`
- Experiment ID: `1`
- Backend: `evaluation_results/mlflow.db`
- Artifacts: `evaluation_results/mlflow_artifacts/`
- Machine-readable index: `evaluation_results/mlflow_run_index.json`

| Scope | Run ID | Local UI link |
|---|---|---|
| Tasks 1–4 parent | `3e1b28afab57490fbe6b02b07ca5d454` | [Open run](http://127.0.0.1:5000/#/experiments/1/runs/3e1b28afab57490fbe6b02b07ca5d454) |
| Task 1 Universal AE | `0f13f7bfa2c549329a6506a5e5f3049c` | [Open run](http://127.0.0.1:5000/#/experiments/1/runs/0f13f7bfa2c549329a6506a5e5f3049c) |
| Task 2 Hard Routing | `0b82054fbe2e4591ac2a8a8824168a03` | [Open run](http://127.0.0.1:5000/#/experiments/1/runs/0b82054fbe2e4591ac2a8a8824168a03) |
| Task 3 Soft MoE | `2366348a9d714525bc5211c1f80f56b8` | [Open run](http://127.0.0.1:5000/#/experiments/1/runs/2366348a9d714525bc5211c1f80f56b8) |
| Task 4 Face-to-Sketch | `2955b70f1c084313872ef35abf3d6e57` | [Open run](http://127.0.0.1:5000/#/experiments/1/runs/2955b70f1c084313872ef35abf3d6e57) |

The links resolve after the local UI is started.

## Start the UI

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\mlflow.exe ui `
  --backend-store-uri "sqlite:///E:/genAI-a1/evaluation_results/mlflow.db" `
  --default-artifact-root "file:///E:/genAI-a1/evaluation_results/mlflow_artifacts" `
  --host 127.0.0.1 `
  --port 5000
```

Then open <http://127.0.0.1:5000> or one of the run links above.

## Create another tracked snapshot

Install the pinned tracking dependency if needed:

```powershell
uv pip install --python .\.venv\Scripts\python.exe -r requirements-tracking.txt
```

Log the currently saved evaluation artifacts:

```powershell
.\.venv\Scripts\python.exe scripts\track_saved_results_mlflow.py
```

Each invocation creates a new parent run and four child runs, then replaces
`evaluation_results/mlflow_run_index.json` with the IDs and links from that
snapshot. It does not rerun model inference.

## What is logged

- Task 1–3 aggregate metrics, CPU timings, provider, environment, and model hashes.
- Task 1 metrics for all ten official conditions.
- Task 2 oracle/predicted metrics, routing accuracy, and confusion matrix.
- Task 3 per-condition metrics and all four mean routing weights.
- Task 4 overall and per-style metrics plus checkpoint metadata.
- Source JSON/CSV artifacts and the report figures associated with routing and
  face-to-sketch evaluation.
