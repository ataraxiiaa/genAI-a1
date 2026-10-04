# YouTube Demonstration Script (Target: 6 minutes 15 seconds)

This script covers every video item required by the assignment: application
startup, image uploading, runtime corruption, universal restoration, hard
routing, soft expert weights, face-to-sketch generation, result downloading,
and experiment-tracking records.

## Before recording

1. Use a 1920x1080 recording canvas and increase browser/terminal text enough
   to remain readable after YouTube compression.
2. Put one ordinary pet image and one front-facing portrait in an easy-to-find
   folder. Do not use private or copyrighted-sensitive images.
3. Start Docker Desktop, but do not start the Compose stack yet.
4. Open two PowerShell windows in the repository root:
   `E:\genAI-a1`.
5. In the second PowerShell window, start MLflow before recording:

   ```powershell
   .\.venv\Scripts\mlflow.exe ui `
     --backend-store-uri "sqlite:///E:/genAI-a1/evaluation_results/mlflow.db" `
     --default-artifact-root "file:///E:/genAI-a1/evaluation_results/mlflow_artifacts" `
     --host 127.0.0.1 `
     --port 5000
   ```

6. Open browser tabs for `http://localhost:5173` and
   `http://127.0.0.1:5000`, but leave the application tab unrefreshed until the
   Docker startup section.
7. Close notifications and anything containing passwords, tokens, email,
   personal files, or the `.env` file.
8. Rehearse once. Do not wait silently during builds: explain the architecture
   while Docker uses its cached layers.

## Recording script

### 0:00–0:25 — Introduction

**On screen:** Repository root in File Explorer or a terminal running
`Get-ChildItem`.

**Say:**

> Hello, this is my Generative AI Assignment One implementation. It contains
> four integrated computer-vision systems: a universal restoration
> autoencoder, a hard-routed specialist ensemble, a soft mixture-of-experts,
> and a style-conditioned face-to-sketch conditional GAN. The React frontend,
> FastAPI backend, and exported ONNX models run as one Docker Compose
> application. I will demonstrate startup, live inference for all four tasks,
> output downloading, and the saved MLflow experiment records.

### 0:25–0:55 — One-command Docker startup

**On screen:** First PowerShell window at the repository root.

**Do:** Run:

```powershell
docker compose up --build
```

Wait until the backend prints that all four models loaded and Uvicorn is
running. Point briefly to the frontend and backend container messages.

**Say:**

> This is the single documented startup command. Docker builds the React and
> Nginx frontend and the FastAPI inference backend. The trained models are
> mounted read-only and loaded through ONNX Runtime. The same command was also
> verified from a clean clone using only the README setup instructions.

### 0:55–1:20 — Application overview and health

**On screen:** Refresh `http://localhost:5173`.

**Do:** Point to **ONNX Backend Active** and the four navigation tabs.

**Say:**

> The browser now reports that the ONNX backend is active. All four tasks are
> available through this single interface. The inputs are resized to 128 by
> 128 RGB, and inference is performed by the submitted ONNX models rather than
> placeholder frontend logic.

### 1:20–2:10 — Task 1: upload, runtime corruption, universal restoration

**On screen:** Open **Universal Denoise**.

**Do:**

1. Click **Upload any custom photo** and select your prepared pet image.
2. Select **Salt & Pepper** and **High** severity.
3. Explain the selection, then click **Restore Image**.
4. Point to **Input (Damaged)**, **Restored Output (ONNX)**,
   **Absolute Error Map**, latency, L1 error, and PSNR.

**Say:**

> First I upload a previously unseen image. I select high salt-and-pepper
> corruption. The clean file is not replaced by a stored corrupted copy; this
> degradation is generated at request time using the assignment severity
> definition. The universal autoencoder is not told the corruption class. It
> produces a restored image and an absolute-error map. The interface also
> reports this run's latency, L1 error, and PSNR against the clean source.

### 2:10–3:05 — Task 2: predicted hard routing

**On screen:** Open **Specialist Routing**.

**Do:**

1. Select a sample image.
2. Choose **Gaussian Blur**, **Medium**, and **Classifier Auto-Route**.
3. Click **Execute Hard-Routing Restoration**.
4. Point to the four classifier probabilities, selected expert, predicted
   mode, restored output, error map, and metrics.
5. Briefly click **Oracle Ground-Truth** only to show that the benchmark/debug
   mode exists, then return to **Classifier Auto-Route**.

**Say:**

> Task Two first classifies the runtime degradation, then makes a discrete
> routing decision. I am using predicted routing, so the model is not given
> the correct class. The confidence bars show all four classifier
> probabilities, and the winning class selects either the clean bypass or one
> of the noise, blur, and occlusion specialist autoencoders. Here the chosen
> expert processes the image and the application displays the output, error
> map, quality metrics, and inference time. Oracle routing is retained as a
> controlled comparison, but predicted mode is the deployable path.

### 3:05–3:55 — Task 3: soft mixture-of-experts

**On screen:** Open **Soft MoE**.

**Do:**

1. Select a sample image.
2. Choose **Rectangular Box Occlusion** and **High** severity.
3. Click **Execute Soft MoE Restoration**.
4. Point to all four percentages under **Continuous Routing Gating Weights
   (Softmax Blending)**, the dominant specialist, soft-blended output, error
   map, and metrics.

**Say:**

> Task Three replaces the single hard decision with differentiable soft
> routing. The gate assigns continuous weights to the clean, noise, blur, and
> occlusion branches. All four weights are visible here and combine to produce
> the soft-blended restoration. For this high occlusion example, I can inspect
> the dominant specialist without hiding the contributions from the other
> branches. The app again shows the output, error map, metrics, and ONNX
> latency.

### 3:55–4:55 — Task 4: upload, style conditioning, generation, download

**On screen:** Open **Face to Sketch**.

**Do:**

1. Click **Upload Photo** and select your prepared front-facing portrait.
2. Choose **Style 1**, then click **Generate Face Sketch (Style 1)**.
3. Point to the input and generated sketch.
4. Use the re-render toolbar to generate **Style 2** or **Style 3**.
5. Click **Save** beside the generated sketch and briefly show the downloaded
   file in the browser download indicator.

**Say:**

> Task Four accepts an uploaded portrait and a style condition. The Pix2Pix
> U-Net conditional GAN generates the paired sketch using the selected style
> vector. I can re-render the same photograph in another learned style without
> changing the input. The result is produced by ONNX Runtime, its latency is
> displayed, and the generated sketch can be downloaded directly using the
> Save control.

### 4:55–5:45 — MLflow experiment records

**On screen:** Switch to `http://127.0.0.1:5000`.

**Do:**

1. Open experiment **genai-assignment-official-evaluation**.
2. Show the parent run and the four nested Task runs.
3. Open one run, preferably **Task 2 Hard Routing**.
4. Scroll through metrics, parameters, and artifacts. Show the routing
   accuracy/confusion-matrix artifact if visible.

**Say:**

> The evaluation snapshot is recorded in repository-local MLflow. This
> experiment contains a parent run and separate child runs for all four tasks.
> The records include official-test metrics, environment and provider data,
> model hashes, per-condition results, routing accuracy and distributions, and
> Task Four per-style results. The associated machine-readable JSON and CSV
> files and report figures are retained as artifacts, so the paper's claims can
> be traced back to saved results.

### 5:45–6:15 — Reproducibility and conclusion

**On screen:** Briefly show `README.md`, `evaluation_results/`, `report/`, and
`AI_USE_APPENDIX.md` in File Explorer or GitHub. End on the running app.

**Say:**

> The repository includes the complete training, Optuna tuning, deterministic
> test-manifest, evaluation, ONNX export, application, Docker, and dependency
> files. The official evaluation uses all ten Oxford test conditions and the
> complete held-out FS2K test set. The IEEE LaTeX report, qualitative examples,
> four genuine failure cases, implemented-application screenshot, MLflow run
> IDs, and AI-use appendix are included. This completes the end-to-end
> demonstration of all four systems. Thank you.

## After recording

1. Confirm the final duration is between **5:00 and 7:00**.
2. Check that the upload action, runtime corruption, four task outputs, soft
   weights, Task 4 download, Docker startup, and MLflow records are readable.
3. Upload to YouTube as **Public** or **Unlisted**, not Private.
4. Copy the YouTube URL into both locations currently marked pending:
   `README.md` and `report/main.tex`.
5. Recompile the final report after adding the URL, then open the PDF and test
   both the GitHub and YouTube links.
6. Stop the application after recording with `Ctrl+C`, followed by:

   ```powershell
   docker compose down
   ```
