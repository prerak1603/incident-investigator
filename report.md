# Agentic Incident Investigator for Surveillance Video

## Overview

This is a CPU-only pipeline that watches surveillance video, flags candidate activity
segments (fighting, vandalism, fire/smoke) using pretrained CLIP and YOLO models, and
then hands those raw detections to an LLM agent that re-evaluates them against
historical false-positive data and short-term case context before writing a final
incident report. The point of the agent layer isn't to add another detector — it's to
catch the cases where the CV layer is confidently wrong, using signals a single-frame
classifier has no access to (has this camera cried wolf on this class before? what
else happened nearby in the last minute?).

## Architecture

The pipeline is nine stages, each its own module, chained by a single orchestrator
function (`incident_investigator/orchestrator.py`).

**Stage 1 — Ingestion.** Opens a video with OpenCV, samples frames at a fixed 2fps
(reading sequentially and picking frames off elapsed time, not seeking by frame index
— seeking drifts on variable-frame-rate footage, sequential reads don't), and slices
the frame stream into overlapping 2s windows with 1s stride. Handles the annoying
edge cases directly: fps=0 reported by a corrupt container falls back to a default
instead of dividing by zero, and a clip shorter than one window returns a single
partial window instead of an empty result.

**Stage 2 — Features.** Three signals per window: CLIP ViT-B-32 zero-shot
classification against four hand-written class prompts, YOLOv8n person counting
(max across the window, not average — a fight can have someone step out of frame for
a second without the incident stopping), and Farneback optical flow motion magnitude
between consecutive frames. CLIP was the obvious choice given the CPU-only, no-training
constraint — there's no labeled training set for this task and no GPU to train one on
anyway, so a pretrained vision-language model that can be pointed at a plain-English
description of the target class ("surveillance footage of two or more people
physically fighting") is a much better fit than trying to train a classifier from
scratch on four categories with a handful of clips. Farneback optical flow was chosen
over anything learned for the same CPU-budget reason — it's a classical, well
understood algorithm already built into OpenCV, needs no model weights, and gives a
genuinely useful auxiliary signal (motion magnitude correlates with physical
altercations and vandalism, and is low for a static aerial shot of smoke) without
adding another network to run per frame. The Farneback parameters used are the
standard values from OpenCV's own documentation, not tuned on our four clips.

**Stage 3 — Segments.** Thresholds each window's CLIP score against a named constant
(`ACTIVE_THRESHOLD = 0.5`), merges consecutive active windows of the same class into
segments, bridges gaps of `MAX_GAP_TO_BRIDGE = 1.0s`, drops anything shorter than
`MIN_SEGMENT_DURATION = 1.5s`, and computes the idle/background gaps as their own
segments so the report can state how much of the footage had nothing happening at all.

**Stage 4 — Case bundling + memory.** Segments within `CASE_PROXIMITY = 20.0s` of each
other get bundled into one "case" (this matters once a video has more than one
activity type happening close together — none of our four test clips actually
exercise this, since each one only produces a single segment). Two kinds of memory
get attached to each case: short-term (an in-session Python object holding recently
seen cases, for spotting related activity within the same camera session) and
long-term (a SQLite table of past camera/class/hour/verdict rows, queried for a
historical false-positive rate). SQLite was picked for the long-term store because it
needs zero setup, ships with Python, and the actual query pattern here — "how often
has this camera/class/hour combination been a false alarm" — is a two-line SQL
`SELECT`, not something that needs a real database server. The seed data in that
table right now is four hand-written placeholder rows (clearly marked as such in
`memory.py`) since no real incident history exists yet for a codebase this new.

**Stage 5 — Agent reasoning.** For each case, a prompt bundles the segment evidence
(raw confidence, duration, peak person count, motion magnitude), the historical
false-positive rate, and any related recent cases, and asks Claude for a revised
confidence, a corroboration status (corroborated / contradicted / insufficient
evidence), and a reasoning string. The system prompt forces JSON-only output so
parsing is deterministic; a failed API call or a response that doesn't parse gets
marked `needs_review` instead of crashing the run or silently keeping a garbage
confidence value.

**Stage 6 — Reflection.** A second LLM pass audits Stage 5's own verdict rather than
re-deriving it: does the stated reasoning actually support the confidence and status
assigned, is there any contradiction between segments in the same case, and should
this be kicked to a human instead of auto-resolved. If Stage 5 holds up, Stage 6 is
supposed to say so rather than changing something just to have changed something —
and in testing, it did exactly that for the fire/smoke case, where the audit
explicitly stated "no changes made to the first-pass verdict."

**Stage 7 — Report generation.** Assembles everything above into one JSON structure
per video (class, time range, raw/revised/final confidence, final verdict, a one-line
evidence summary, a citation string, and the reflection note — only included when
Stage 6 actually changed something) and renders a markdown version by walking that
same dict, so the two outputs can't drift apart from each other.

**Stage 8 — Logging.** The orchestrator function that runs Stages 1–7 in sequence
also assembles one consolidated JSON log per video as it goes — every stage's
intermediate output, per-stage timing, and the LLM call count — rather than having
each stage write its own log file that then needs to be stitched back together
afterward.

**Stage 9 — Evaluation.** Two independent halves. Part A scores Stage 3's raw
segments against hand-labeled ground truth (temporal IoU, mAP at three thresholds,
per-class precision/recall) — this only tells you whether the CV layer's boundaries
are roughly right, it has nothing to do with the agent. Part B is the memory
ablation: Stages 4–6 are re-run three times per case with the memory config set to
none / short-term-only / short-and-long-term, recording the confidence shift and
verdict change for each. Getting the "none" condition right took an extra pass —
the first instinct is to just pass empty `historical_fp_rate`/`related_cases` through
the same prompt-building code, but that still produces a prompt that says "no history
for this camera/class/hour," which tells the model a memory subsystem exists and came
up empty. That's not the same experimental condition as a system that has no memory
subsystem at all. `_build_prompt()` in `agent.py` takes a `memory_mode` argument that
skips those lines of the prompt entirely in the `none` case — verified directly by
grepping the actual generated prompt text for the phrase `historical_false_positive_rate`
before spending any API budget on the real ablation run.

## Key finding

The clearest evidence that the memory layer does real work, not just theater, is the
fire/smoke case from the Arson test clip. Stage 3's raw CLIP confidence for that
segment was **0.997** — about as confident as the CV layer gets. The seeded long-term
memory table happens to record a 1.00 historical false-positive rate for `fire_smoke`
on that camera at that hour (every past detection of this class at this camera/hour
was wrong). With that context available, Stage 5 revised the confidence down to
**0.05**, and Stage 6's reflection pass confirmed that verdict outright rather than
second-guessing it. In the memory ablation (`evaluation/evaluation_summary.md`), the
`short_and_long_term` config produced the largest average confidence shift across all
three test cases (−0.580), clearly larger than `none` (−0.356) or `short_term`
(−0.290) — and that gap is driven almost entirely by this one case. A raw CNN
confidence of 0.997 turning into a final confidence of 0.05 because of what happened
on that camera in the past is exactly the "reduce false positives instead of dumping
raw alarms" behavior this whole project is testing for.

## Metrics and evaluation

**Part A (CV):** mAP@0.3, mAP@0.5, and mAP@0.7 all came out to 1.0, with 1.0
precision/recall on all three classes. This is not a meaningful accuracy claim — it's
a direct consequence of having exactly one hand-labeled ground-truth interval per
class across the whole test set. A metric like mAP saturates instantly with N=1; all
it's actually confirming is that Stage 3's segment boundaries landed reasonably close
(IoU between 0.83 and 1.0) to where I hand-labeled the activity on these three
specific clips. It says nothing about how the pipeline would perform on footage it
hasn't seen.

**Part B (memory ablation):**

| memory config | avg confidence shift | % verdict changed | LLM calls |
|---|---|---|---|
| none | −0.356 | 100% | 6 |
| short_term | −0.290 | 100% | 6 |
| short_and_long_term | −0.580 | 100% | 6 |

The `% verdict changed` column is 100% in every config, which sounds dramatic but
mostly reflects sample size — with 3 total cases, even the `none` condition (which
has zero memory context at all) crossed the confidence-divergence threshold every
time, because Stage 5 already discounts an overconfident raw score using motion
magnitude and person count alone. That column needs a lot more cases before it
distinguishes anything. What is a real, if small-sample, signal is the ordering of
average confidence shift: full memory shifted confidence more than short-term-only,
which shifted more than no memory at all — consistent with the fire/smoke result
above being a genuine long-term-memory effect and not noise. One thing I noticed but
can't fully explain from this sample: `short_term`'s shift (−0.290) was smaller than
`none`'s (−0.356), which is not what I expected going in. All three clips only ever
produce one case each, so `related_cases` is empty in every mode regardless — the
only difference `short_term` mode introduces over `none` is the prompt explicitly
stating that no related cases were found, instead of never mentioning short-term
memory as a concept. That's a real, observed difference in the output, but with three
cases and no fixed sampling temperature, I can't rule out plain LLM variance as the
explanation instead of a genuine effect of that wording change.

## Limitations and future work

**Placeholder data, not UCF-Crime.** The four test clips are free stock footage from
Pexels standing in for the UCF-Crime dataset. Kaggle authentication for the real
dataset never worked over three attempts — two different `KGAT_`-prefixed tokens were
rejected with a 403 even after switching between `KAGGLE_API_TOKEN`,
`~/.kaggle/access_token`, and upgrading the CLI itself, and the tokens didn't match
the format Kaggle actually issues. `download_data.py` (a `kagglehub` call) is still
in the repo, unused, ready to point at the real dataset once valid credentials exist.
Everything downstream of ingestion is dataset-agnostic, but the CV evaluation numbers
above only mean anything for these four specific clips.

**The cross-segment contradiction check was never exercised.** Stage 6's reflection
prompt explicitly asks the model to check for contradictions between segments *within
the same case*, but every case bundled from these four clips has exactly one segment
— none of them mix two activity classes close together in time. That check is
implemented and untested; the first real multi-class case would be the first real
test of it.

**Small sample size throughout.** Four videos, three of which produce any incidents
at all, one ground-truth interval per class, three cases total in the memory
ablation. Every number in this report should be read as "this is what happened on
these four specific clips," not as a claim about the pipeline's general accuracy.

**Long-term memory is seeded with synthetic data.** The four rows in
`case_history` (`memory.py`) are hand-written placeholders, not real incident
history — there's no real deployment history to draw from yet for a pipeline this
new. The false-positive-rate lookup mechanism itself is real and tested (both a hit
and a correct `None`-for-no-history miss were verified), but the actual numbers it
returns right now don't reflect any real camera's track record.

## Failure analysis

Stage 5 hit one real blocking bug worth documenting because the error message
actively pointed in the wrong direction. Every call through the `anthropic` Python
SDK failed with `anthropic.APIConnectionError: Connection error.` — which reads like
a network or auth problem. Before assuming that, I checked the obvious alternative
explanations directly with `curl` against `https://api.anthropic.com/v1/messages`
using the same API key and model name: it returned a normal 200 with a real
response. That ruled out the key, the model name, and the network path in one step,
which meant the bug had to be somewhere inside the SDK itself, not in what it was
calling. Catching the actual underlying exception (rather than the wrapped
`APIConnectionError`) traced it to
`anthropic/../httpx2/_decoders.py`: `TypeError: process() takes no keyword
arguments`, inside the Brotli response-decompression path. The environment had
`Brotli 1.0.9` installed, whose `.process()` method doesn't accept the
`output_buffer_limit` keyword argument that this version of the SDK's HTTP client
passes to it — a version mismatch between an installed system library and what the
SDK expected, with nothing wrong with the request itself. `pip install --upgrade
Brotli` to 1.2.0 fixed it immediately. The generalizable lesson: a wrapped exception
message from an SDK is a starting hypothesis, not a diagnosis — isolating the
request with a raw `curl` call and then reading the *real* traceback underneath the
wrapper found the actual bug in about five minutes instead of chasing API-key or
network red herrings.
