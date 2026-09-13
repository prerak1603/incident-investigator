def _evidence_summary(segment, case):
    fp_rate = case.historical_fp_rate.get(segment.label)
    fp_str = "no history" if fp_rate is None else f"{fp_rate:.2f}"
    return (
        f"peak person count {segment.peak_person_count}, "
        f"avg motion magnitude {segment.avg_motion_magnitude:.2f}; "
        f"historical false-positive rate at this camera/hour: {fp_str}"
    )


def _reflection_changed(rs):
    # confidence can stay put while the verdict still meaningfully changes —
    # needs_human_review has no Stage 5 equivalent at all, so it always counts
    if rs.final_verdict == "needs_human_review":
        return True
    return abs(rs.final_confidence - rs.pre_confidence) > 1e-6


def _build_incident(case_index, segment, case, verdict, reflection):
    incident = {
        "case_index": case_index,
        "class": segment.label,
        "start_time": round(segment.start, 2),
        "end_time": round(segment.end, 2),
        "raw_confidence": round(segment.confidence, 3),
        "evidence_summary": _evidence_summary(segment, case),
        "citation": f"{case.camera_id} @ {segment.start:.1f}s-{segment.end:.1f}s",
    }

    if verdict is None or verdict.needs_review:
        incident.update(
            revised_confidence=None,
            final_confidence=None,
            final_verdict="pipeline_error",
            reflection_note=verdict.error if verdict else "stage 5 was not run for this case",
        )
        return incident

    sv = next((s for s in verdict.segments if s.segment_class == segment.label), None)
    if sv is None:
        incident.update(
            revised_confidence=None,
            final_confidence=None,
            final_verdict="pipeline_error",
            reflection_note=f"stage 5 response did not include a verdict for class '{segment.label}'",
        )
        return incident

    incident["revised_confidence"] = round(sv.revised_confidence, 3)

    if reflection is None or reflection.needs_review:
        incident.update(
            final_confidence=round(sv.revised_confidence, 3),
            final_verdict="pipeline_error",
            reflection_note=reflection.error if reflection else "stage 6 was not run for this case",
        )
        return incident

    rs = next((s for s in reflection.segments if s.segment_class == segment.label), None)
    if rs is None:
        incident.update(
            final_confidence=round(sv.revised_confidence, 3),
            final_verdict="pipeline_error",
            reflection_note=f"stage 6 response did not include a verdict for class '{segment.label}'",
        )
        return incident

    incident["final_confidence"] = round(rs.final_confidence, 3)
    incident["final_verdict"] = rs.final_verdict
    incident["reflection_note"] = rs.reflection_note if _reflection_changed(rs) else None
    return incident


def build_report(video_id, source_path, camera_id, duration, segments, cases, verdicts, reflections):
    incidents = []
    for case_index, case in enumerate(cases):
        verdict = verdicts[case_index]
        reflection = reflections[case_index]
        for segment in case.segments:
            incidents.append(_build_incident(case_index, segment, case, verdict, reflection))

    incidents.sort(key=lambda i: i["start_time"])

    idle_segments = [s for s in segments if s.label == "idle"]
    idle_duration = sum(s.end - s.start for s in idle_segments)

    summary = {
        "total_incidents": len(incidents),
        "confirmed": sum(1 for i in incidents if i["final_verdict"] == "confirmed"),
        "downgraded": sum(1 for i in incidents if i["final_verdict"] == "downgraded"),
        "needs_human_review": sum(1 for i in incidents if i["final_verdict"] == "needs_human_review"),
        "pipeline_errors": sum(1 for i in incidents if i["final_verdict"] == "pipeline_error"),
        "idle_segment_count": len(idle_segments),
        "idle_duration": round(idle_duration, 2),
        "idle_fraction": round(idle_duration / duration, 3) if duration > 0 else 0.0,
    }

    return {
        "video_id": video_id,
        "source_path": source_path,
        "camera_id": camera_id,
        "duration": round(duration, 2),
        "incidents": incidents,
        "summary": summary,
    }


def _fmt(value):
    return "—" if value is None else f"{value:.2f}"


def render_markdown(report):
    lines = [f"# Incident Report — {report['video_id']}", ""]
    lines.append(f"Camera: {report['camera_id']}  ")
    lines.append(f"Source: `{report['source_path']}`  ")
    lines.append(f"Duration: {report['duration']:.2f}s")
    lines.append("")

    s = report["summary"]
    error_clause = f", {s['pipeline_errors']} pipeline error(s)" if s["pipeline_errors"] else ""
    lines.append(
        f"{s['total_incidents']} incident(s) detected: {s['confirmed']} confirmed, "
        f"{s['downgraded']} downgraded, {s['needs_human_review']} needs human review{error_clause}. "
        f"Idle time: {s['idle_duration']:.2f}s ({s['idle_fraction'] * 100:.1f}% of footage)."
    )
    lines.append("")

    if not report["incidents"]:
        lines.append("No incidents detected — footage classified as idle/background throughout.")
        return "\n".join(lines)

    for incident in report["incidents"]:
        title = incident["class"].replace("_", " ").title()
        lines.append(f"## {title} — {incident['citation']}")
        lines.append("")
        lines.append("| raw | revised | final | verdict |")
        lines.append("|---|---|---|---|")
        lines.append(
            f"| {_fmt(incident['raw_confidence'])} | {_fmt(incident['revised_confidence'])} | "
            f"{_fmt(incident['final_confidence'])} | {incident['final_verdict']} |"
        )
        lines.append("")
        lines.append(incident["evidence_summary"])
        if incident.get("reflection_note"):
            lines.append("")
            lines.append(f"> {incident['reflection_note']}")
        lines.append("")

    return "\n".join(lines)
