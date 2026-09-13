import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from incident_investigator.ingestion import ingest_video
from incident_investigator.features import extract_window_features, CLASSES
from incident_investigator.segments import merge_segments
from incident_investigator.cases import bundle_cases
from incident_investigator.memory import (
    init_db, seed_demo_data, attach_memory, ShortTermMemory, MEMORY_MODES,
)
from incident_investigator.agent import evaluate_case, reflect_case

IOU_THRESHOLDS = (0.3, 0.5, 0.7)
CONFIDENCE_DIVERGENCE_THRESHOLD = 0.15
ACTIVITY_CLASSES = [c for c in CLASSES if c != "normal"]


# ---------------------------------------------------------------------------
# Part A: CV evaluation — Stage 3 segments vs hand-labeled ground truth
# ---------------------------------------------------------------------------

def load_ground_truth(path="data/ground_truth.json"):
    with open(path) as f:
        raw = json.load(f)
    flat = []
    for video_id, intervals in raw.items():
        for interval in intervals:
            flat.append({"video_id": video_id, "class": interval["class"],
                         "start": interval["start"], "end": interval["end"]})
    return flat


def temporal_iou(a_start, a_end, b_start, b_end):
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = (a_end - a_start) + (b_end - b_start) - inter
    return inter / union if union > 0 else 0.0


def _match(predictions, ground_truth, iou_threshold):
    gts_by_video = defaultdict(list)
    for gt in ground_truth:
        gts_by_video[gt["video_id"]].append({**gt, "matched": False})

    preds_sorted = sorted(predictions, key=lambda p: p["confidence"], reverse=True)
    tp, fp = [], []
    for pred in preds_sorted:
        best_iou, best_gt = 0.0, None
        for gt in gts_by_video.get(pred["video_id"], []):
            if gt["matched"]:
                continue
            iou = temporal_iou(pred["start"], pred["end"], gt["start"], gt["end"])
            if iou > best_iou:
                best_iou, best_gt = iou, gt
        if best_gt is not None and best_iou >= iou_threshold:
            best_gt["matched"] = True
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)

    num_gt = sum(len(v) for v in gts_by_video.values())
    return tp, fp, num_gt


def _average_precision(tp, fp, num_gt):
    if num_gt == 0:
        return None
    if not tp:
        return 0.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / num_gt
    precision = tp_cum / (tp_cum + fp_cum)

    # standard VOC-style all-point interpolation
    precision = np.concatenate(([0.0], precision, [0.0]))
    recall = np.concatenate(([0.0], recall, [1.0]))
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    idx = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[idx + 1] - recall[idx]) * precision[idx + 1]))


def compute_map(predictions, ground_truth, iou_threshold):
    per_class_ap = {}
    for cls in ACTIVITY_CLASSES:
        preds = [p for p in predictions if p["class"] == cls]
        gts = [g for g in ground_truth if g["class"] == cls]
        tp, fp, num_gt = _match(preds, gts, iou_threshold)
        per_class_ap[cls] = _average_precision(tp, fp, num_gt)

    valid = [ap for ap in per_class_ap.values() if ap is not None]
    mean_ap = sum(valid) / len(valid) if valid else None
    return mean_ap, per_class_ap


def precision_recall_per_class(predictions, ground_truth, iou_threshold=0.5):
    results = {}
    for cls in ACTIVITY_CLASSES:
        preds = [p for p in predictions if p["class"] == cls]
        gts = [g for g in ground_truth if g["class"] == cls]
        tp, fp, num_gt = _match(preds, gts, iou_threshold)
        tp_count, fp_count = sum(tp), sum(fp)
        fn_count = num_gt - tp_count
        results[cls] = {
            "precision": tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else None,
            "recall": tp_count / num_gt if num_gt > 0 else None,
            "tp": tp_count, "fp": fp_count, "fn": fn_count,
        }
    return results


def run_cv_evaluation(paths, ground_truth):
    predictions = []
    for path in paths:
        result = ingest_video(path)
        if result.error:
            continue
        video_id = _video_id(path)
        features = [extract_window_features(w) for w in result.windows]
        segments = merge_segments(result.windows, features, result.duration)
        for seg in segments:
            if seg.label == "idle":
                continue
            predictions.append({
                "video_id": video_id, "class": seg.label,
                "start": seg.start, "end": seg.end, "confidence": seg.confidence,
            })

    map_by_threshold = {}
    for thresh in IOU_THRESHOLDS:
        mean_ap, per_class_ap = compute_map(predictions, ground_truth, thresh)
        map_by_threshold[str(thresh)] = {"mAP": mean_ap, "per_class_ap": per_class_ap}

    return {
        "map_by_iou_threshold": map_by_threshold,
        "precision_recall_at_0.5": precision_recall_per_class(predictions, ground_truth, 0.5),
        "predictions": predictions,
    }


def _video_id(path):
    return Path(path).stem


# ---------------------------------------------------------------------------
# Part B: agent evaluation — memory ablation
# ---------------------------------------------------------------------------

def run_memory_ablation(paths, camera_id="cam_01", reference_time=None):
    conn = init_db()
    seed_demo_data(conn)

    rows = []
    for path in paths:
        result = ingest_video(path)
        if result.error:
            continue
        video_id = _video_id(path)
        features = [extract_window_features(w) for w in result.windows]
        segments = merge_segments(result.windows, features, result.duration)

        for mode in MEMORY_MODES:
            # cases are rebuilt from scratch per mode — attach_memory mutates
            # them in place, so reusing case objects across modes would leak
            # one mode's memory context into another's
            short_term = ShortTermMemory()
            cases = bundle_cases(segments, camera_id=camera_id)
            attach_memory(cases, conn, short_term, reference_time=reference_time, memory_mode=mode)

            for case_index, case in enumerate(cases):
                verdict = evaluate_case(case, memory_mode=mode)
                reflection = reflect_case(case, verdict)
                rows.extend(_ablation_rows(video_id, mode, case_index, case, verdict, reflection))

    return rows


def _ablation_rows(video_id, mode, case_index, case, verdict, reflection):
    rows = []
    for segment in case.segments:
        raw_confidence = segment.confidence
        sv = next((s for s in verdict.segments if s.segment_class == segment.label), None) if not verdict.needs_review else None
        rs = next((s for s in reflection.segments if s.segment_class == segment.label), None) if not reflection.needs_review else None

        final_confidence = rs.final_confidence if rs else None
        final_verdict = rs.final_verdict if rs else "pipeline_error"
        shift = (final_confidence - raw_confidence) if final_confidence is not None else None

        rows.append({
            "video_id": video_id,
            "memory_mode": mode,
            "case_index": case_index,
            "class": segment.label,
            "raw_confidence": raw_confidence,
            "revised_confidence": sv.revised_confidence if sv else None,
            "final_confidence": final_confidence,
            "final_verdict": final_verdict,
            "confidence_shift": shift,
            "diverged": abs(shift) > CONFIDENCE_DIVERGENCE_THRESHOLD if shift is not None else None,
            "llm_calls": 2,
        })
    return rows


def summarize_ablation(rows):
    summary = {}
    for mode in MEMORY_MODES:
        mode_rows = [r for r in rows if r["memory_mode"] == mode]
        shifts = [r["confidence_shift"] for r in mode_rows if r["confidence_shift"] is not None]
        diverged = [r["diverged"] for r in mode_rows if r["diverged"] is not None]
        summary[mode] = {
            "num_cases": len(mode_rows),
            "avg_confidence_shift": float(np.mean(shifts)) if shifts else None,
            "pct_verdict_changed": (sum(diverged) / len(diverged) * 100) if diverged else None,
            "total_llm_calls": sum(r["llm_calls"] for r in mode_rows),
        }
    return summary


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------

def render_ablation_markdown(summary):
    lines = ["| memory config | avg confidence shift | % verdict changed | LLM calls |",
             "|---|---|---|---|"]
    for mode, s in summary.items():
        shift = f"{s['avg_confidence_shift']:+.3f}" if s["avg_confidence_shift"] is not None else "—"
        pct = f"{s['pct_verdict_changed']:.0f}%" if s["pct_verdict_changed"] is not None else "—"
        lines.append(f"| {mode} | {shift} | {pct} | {s['total_llm_calls']} |")
    return "\n".join(lines)


def plot_ablation(summary, out_path):
    import matplotlib.pyplot as plt

    modes = list(summary.keys())
    shifts = [summary[m]["avg_confidence_shift"] or 0.0 for m in modes]
    pct_changed = [summary[m]["pct_verdict_changed"] or 0.0 for m in modes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.bar(modes, shifts, color="steelblue")
    ax1.set_title("Avg confidence shift")
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.tick_params(axis="x", rotation=20)

    ax2.bar(modes, pct_changed, color="darkorange")
    ax2.set_title("% cases with verdict change")
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
