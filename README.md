# Agentic Incident Investigator for Surveillance Video

CPU-only hybrid CV + agentic pipeline. A CLIP/YOLO detection layer flags candidate
activity segments in surveillance footage; an agentic layer with memory,
corroboration, and reflection re-evaluates those raw detections and produces a
structured incident report instead of raw alarms.

Classes (initial scope): Fighting, Vandalism, Fire/Smoke, Normal/idle.

**Full writeup: [report.md](report.md)** — architecture rationale, the memory-ablation
finding, evaluation results and their caveats, and a debugging writeup of the one real
bug hit along the way.

**Architecture diagram: [docs/architecture.png](docs/architecture.png)**

## Setup

```bash
pip install -r requirements.txt
```

Needs Python 3.10+ (uses `list[...]`/`X | None` builtin generics, no `typing` import
needed for those). First run downloads CLIP (ViT-B-32, openai weights) and YOLOv8n
weights automatically — that's a few hundred MB, only happens once.

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Required for Stages 5/6 (agent reasoning + reflection) and anything downstream of
them (Stage 7 report, Stage 8 log, Stage 9 agent-eval half). Stages 1–4 and Part A of
Stage 9 run fine without it.

## Running the whole pipeline on a clip

```bash
python -c "
from incident_investigator.orchestrator import run_pipeline
log = run_pipeline('data/raw/Fighting/two_boys_fighting_10505848.mp4', camera_id='cam_01')
print(log['timing'], log['llm_calls'])
"
```

or just:

```bash
python run_orchestrator.py data/raw/Fighting/two_boys_fighting_10505848.mp4
```

`run_pipeline()` runs Stages 1–7 in sequence and writes `logs/<video_id>.json` (Stage
8's consolidated per-stage log) and `reports/<video_id>.json` + `.md` (Stage 7's
report, both JSON and human-readable). Point it at any clip under `data/raw/`, or your
own footage — nothing in Stages 1–8 is dataset-specific.

## Running a single stage

Each stage also has its own module (`incident_investigator/<stage>.py`) and a thin
demo script at the repo root (`run_<stage>.py`) that runs just that stage in
isolation and prints its output — useful for checking one stage's behavior without
paying for the LLM calls in the stages after it. E.g.:

```bash
python run_ingestion.py data/raw/Fighting/two_boys_fighting_10505848.mp4
python run_features.py    # runs stages 1-2 on 4 default clips
python run_segments.py    # runs stages 1-3
python run_cases.py       # runs stages 1-4
python run_agent.py       # runs stages 1-5
python run_reflection.py  # runs stages 1-6, plus one synthetic test case
python run_report.py      # runs stages 1-7
```

## Evaluation

```bash
python run_evaluate.py
```

Runs Stage 9 both halves: Part A scores Stage 3's segments against
`data/ground_truth.json` (hand-labeled by eyeballing sampled frames, not
CLIP-self-labeled — see report.md for why). Part B re-runs Stages 4–6 three times per
case under three memory configs (none / short-term / short+long-term) and reports the
confidence-shift comparison. Writes `evaluation/evaluation_results.json`,
`evaluation/evaluation_summary.md`, and `evaluation/memory_ablation.png`.

This spends real API calls — 3 memory configs × however many cases exist across
whatever clips you point it at, times 2 (Stage 5 + Stage 6 each) — check
`evaluation_summary.md`'s `LLM calls` column before rerunning on a larger clip set.

## Data

`data/raw/<class>/*.mp4` — see [data/raw/README.md](data/raw/README.md). Currently
placeholder stock footage standing in for UCF-Crime clips (Kaggle auth blocked, see
report.md's limitations section for specifics). `download_data.py` is left in place,
unused, ready to point at the real dataset once valid Kaggle credentials exist.

`data/ground_truth.json` — hand-labeled activity intervals for the 4 test clips, used
by Stage 9 Part A.

`data/long_term_memory.db` — SQLite long-term memory store. Seeded with 4 synthetic
placeholder rows (`incident_investigator/memory.py`'s `seed_demo_data`) since no real
incident history exists yet.

## Pipeline stages

| # | Stage | Module | Status |
|---|---|---|---|
| 1 | Ingestion (frame sampling, windowing) | `incident_investigator/ingestion.py` | done |
| 2 | Per-window features (CLIP, YOLO, optical flow) | `incident_investigator/features.py` | done |
| 3 | Segment merging | `incident_investigator/segments.py` | done |
| 4 | Case bundling + memory | `incident_investigator/cases.py`, `incident_investigator/memory.py` | done |
| 5 | Agent reasoning | `incident_investigator/agent.py` | done |
| 6 | Reflection | `incident_investigator/agent.py` | done |
| 7 | Report generation | `incident_investigator/report.py` | done |
| 8 | Logging | `incident_investigator/orchestrator.py` | done |
| 9 | Evaluation | `incident_investigator/evaluate.py` | done |
| 10 | Visualization | *(covered by Stage 9's `plot_ablation` — no separate module)* | n/a |

## Outputs

- `reports/<video_id>.json` / `.md` — per-video structured incident report (Stage 7)
- `logs/<video_id>.json` — full per-video pipeline trace: every stage's intermediate
  output, per-stage timing, LLM call count (Stage 8)
- `evaluation/` — Stage 9's CV metrics, memory-ablation summary, and chart
