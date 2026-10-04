# AI-use appendix

This appendix discloses AI-assisted work on the submitted system. The student
remains responsible for the implementation, results, report, and ability to
explain or modify every component.

## Tools and purposes

| Tool | Tasks for which it was used |
|---|---|
| OpenAI Codex / ChatGPT | Repository inspection; debugging the Task 2 oracle-routing and Task 4 evaluation paths; creating the reproducible Tasks 1–3 evaluator and ten official test manifests; adding MLflow tracking; checking report claims against saved JSON/CSV results; and drafting repository documentation. |
| Google Stitch | Producing the original interface design evidence shown in the report and informing the implemented React/Tailwind visual layout. |

No model metric in the report was accepted from an AI response. Reported
numbers were taken from the machine-readable artifacts under
`evaluation_results/`.

## Testing and correction of AI-assisted output

AI-assisted changes were checked using the following evidence and procedures:

1. Tasks 1–3 were run on the ten deterministic Oxford-IIIT Pet manifests with
   `scripts/evaluate_tasks_1_3.py`. The evaluator records model SHA-256 hashes,
   ONNX Runtime provider, per-condition metrics, Task 1 severity breakdowns,
   and Task 2/3 routing breakdowns.
2. Hard routing was evaluated twice on the identical samples: once using the
   manifest label (oracle route) and once using the classifier prediction.
   Counts and the confusion matrix are stored with the metrics.
3. Task 4 was rerun on all 1,046 official held-out FS2K pairs. The loader/path
   pairing issue was corrected before accepting the saved aggregate and
   per-style results.
4. `scripts/track_saved_results_mlflow.py` was run against the saved results.
   The local experiment IDs and artifact paths are recorded in `TRACKING.md`
   and `evaluation_results/mlflow_run_index.json`.
5. Report tables and quantitative prose were rewritten from saved JSON/CSV
   outputs. Unsupported latency, quality, and causal claims were removed.
6. Python entry points were syntax/import checked where practical, frontend
   production builds were checked, and the MLflow UI was started and queried
   locally. Long-running training was not repeated merely to validate
   documentation.
7. Representative and failure-case report panels were regenerated through
   actual checkpoint inference on held-out data. Their sample identifiers,
   metrics, and deterministic selection rules are stored in
   `evaluation_results/report_visual_evidence.json`; a prior schematic failure
   illustration was removed from the report workflow.
8. The implemented-app screenshot was captured from the running React frontend
   while the FastAPI `/health` endpoint returned `healthy`. Stale overview
   metrics exposed by that capture were corrected from the saved evaluation
   artifacts before the final screenshot was taken.

## Human review responsibilities

Before submission, the student should independently:

- review the diff and understand each architecture, loss, dataset split, and
  evaluation metric;
- run the documented evaluation and Docker Compose commands from a clean clone;
- inspect representative outputs and failure cases rather than relying only on
  aggregate metrics;
- confirm that Git LFS objects can be downloaded from GitHub;
- insert the final YouTube demonstration URL after uploading the required
  five-to-seven-minute video; and
- be prepared to explain and modify every submitted component during the
  evaluation.

## Known limitations

AI assistance can introduce plausible but incorrect code or prose. The saved
artifacts reduce that risk for quantitative claims, but they do not prove
generalization beyond the specified datasets, corruption levels, model hashes,
or runtime environment. The disclosure above covers the AI-assisted work in
the current repository session; the student must add any other AI tools or
uses that occurred outside this recorded work.
