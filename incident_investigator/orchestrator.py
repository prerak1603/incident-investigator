import json
import time
from pathlib import Path

from incident_investigator.ingestion import ingest_video
from incident_investigator.features import extract_window_features
from incident_investigator.segments import merge_segments
from incident_investigator.cases import bundle_cases
from incident_investigator.memory import init_db, seed_demo_data, attach_memory, ShortTermMemory
from incident_investigator.agent import evaluate_case, reflect_case
from incident_investigator.report import build_report, render_markdown

LOG_DIR = Path("logs")
REPORT_DIR = Path("reports")


def _serialize_features(windows, features):
    return [
        {
            "start": round(w.start, 2),
            "end": round(w.end, 2),
            "class_scores": {k: round(v, 4) for k, v in f.class_scores.items()},
            "person_count": f.person_count,
            "motion_magnitude": round(f.motion_magnitude, 3),
        }
        for w, f in zip(windows, features)
    ]


def _serialize_segment(seg):
    return {
        "label": seg.label,
        "start": round(seg.start, 2),
        "end": round(seg.end, 2),
        "confidence": round(seg.confidence, 3),
        "peak_person_count": seg.peak_person_count,
        "avg_motion_magnitude": round(seg.avg_motion_magnitude, 3),
    }


def _serialize_case(case, case_index):
    return {
        "case_index": case_index,
        "start": round(case.start, 2),
        "end": round(case.end, 2),
        "classes": case.classes,
        "camera_id": case.camera_id,
        "historical_fp_rate": case.historical_fp_rate,
        "related_cases_count": len(case.related_cases),
        "segments": [_serialize_segment(s) for s in case.segments],
    }


def _serialize_verdict(verdict, case_index):
    return {
        "case_index": case_index,
        "needs_review": verdict.needs_review,
        "error": verdict.error,
        "segments": [
            {
                "class": sv.segment_class,
                "revised_confidence": sv.revised_confidence,
                "corroboration_status": sv.corroboration_status,
                "reasoning": sv.reasoning,
            }
            for sv in verdict.segments
        ],
    }


def _serialize_reflection(reflection, case_index):
    return {
        "case_index": case_index,
        "needs_review": reflection.needs_review,
        "error": reflection.error,
        "segments": [
            {
                "class": rs.segment_class,
                "pre_confidence": rs.pre_confidence,
                "pre_status": rs.pre_status,
                "final_confidence": rs.final_confidence,
                "final_verdict": rs.final_verdict,
                "reflection_note": rs.reflection_note,
            }
            for rs in reflection.segments
        ],
    }


def _write_log(video_id, log):
    LOG_DIR.mkdir(exist_ok=True)
    path = LOG_DIR / f"{video_id}.json"
    path.write_text(json.dumps(log, indent=2))
    return path


def run_pipeline(path, camera_id="cam_01", reference_time=None, conn=None, short_term=None):
    video_id = Path(path).stem
    pipeline_start = time.perf_counter()
    timing = {}
    errors = []

    t0 = time.perf_counter()
    result = ingest_video(path)
    timing["ingestion"] = round(time.perf_counter() - t0, 4)
    warnings = [{"stage": "ingestion", "message": w} for w in result.warnings]

    if result.error:
        errors.append({"stage": "ingestion", "message": result.error})
        log = {
            "video_id": video_id,
            "source_path": path,
            "camera_id": camera_id,
            "timing": {**timing, "total": round(time.perf_counter() - pipeline_start, 4)},
            "llm_calls": {"stage5_calls": 0, "stage6_calls": 0, "total": 0},
            "errors": errors,
            "warnings": warnings,
            "stage1_ingestion": {
                "fps": result.fps,
                "duration": result.duration,
                "frame_count": result.frame_count,
                "window_count": 0,
                "warnings": result.warnings,
            },
        }
        _write_log(video_id, log)
        return log

    t0 = time.perf_counter()
    features = [extract_window_features(w) for w in result.windows]
    timing["features"] = round(time.perf_counter() - t0, 4)

    t0 = time.perf_counter()
    segments = merge_segments(result.windows, features, result.duration)
    timing["segments"] = round(time.perf_counter() - t0, 4)

    t0 = time.perf_counter()
    if conn is None:
        conn = init_db()
        seed_demo_data(conn)
    if short_term is None:
        short_term = ShortTermMemory()
    cases = bundle_cases(segments, camera_id=camera_id)
    attach_memory(cases, conn, short_term, reference_time=reference_time)
    timing["cases_memory"] = round(time.perf_counter() - t0, 4)

    stage5_calls = 0
    t0 = time.perf_counter()
    verdicts = []
    for case in cases:
        verdict = evaluate_case(case)
        verdicts.append(verdict)
        stage5_calls += 1
        if verdict.needs_review:
            errors.append({"stage": "agent_reasoning", "message": verdict.error})
    timing["agent_reasoning"] = round(time.perf_counter() - t0, 4)

    stage6_calls = 0
    t0 = time.perf_counter()
    reflections = []
    for case, verdict in zip(cases, verdicts):
        reflection = reflect_case(case, verdict)
        reflections.append(reflection)
        stage6_calls += 1
        if reflection.needs_review and not verdict.needs_review:
            errors.append({"stage": "reflection", "message": reflection.error})
    timing["reflection"] = round(time.perf_counter() - t0, 4)

    t0 = time.perf_counter()
    report = build_report(
        video_id=video_id,
        source_path=path,
        camera_id=camera_id,
        duration=result.duration,
        segments=segments,
        cases=cases,
        verdicts=verdicts,
        reflections=reflections,
    )
    REPORT_DIR.mkdir(exist_ok=True)
    report_path = REPORT_DIR / f"{video_id}.json"
    report_path.write_text(json.dumps(report, indent=2))
    (REPORT_DIR / f"{video_id}.md").write_text(render_markdown(report))
    timing["report"] = round(time.perf_counter() - t0, 4)
    timing["total"] = round(time.perf_counter() - pipeline_start, 4)

    log = {
        "video_id": video_id,
        "source_path": path,
        "camera_id": camera_id,
        "timing": timing,
        "llm_calls": {
            "stage5_calls": stage5_calls,
            "stage6_calls": stage6_calls,
            "total": stage5_calls + stage6_calls,
        },
        "errors": errors,
        "warnings": warnings,
        "stage1_ingestion": {
            "fps": round(result.fps, 2),
            "duration": round(result.duration, 2),
            "frame_count": result.frame_count,
            "window_count": len(result.windows),
            "warnings": result.warnings,
        },
        "stage2_features": _serialize_features(result.windows, features),
        "stage3_segments": [_serialize_segment(s) for s in segments],
        "stage4_cases": [_serialize_case(c, i) for i, c in enumerate(cases)],
        "stage5_verdicts": [_serialize_verdict(v, i) for i, v in enumerate(verdicts)],
        "stage6_reflections": [_serialize_reflection(r, i) for i, r in enumerate(reflections)],
        "stage7_report_path": str(report_path),
    }

    _write_log(video_id, log)
    return log
