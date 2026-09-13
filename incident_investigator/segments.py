from dataclasses import dataclass

from incident_investigator.features import CLASSES

ACTIVE_THRESHOLD = 0.5
MAX_GAP_TO_BRIDGE = 1.0
MIN_SEGMENT_DURATION = 1.5

ACTIVITY_CLASSES = [c for c in CLASSES if c != "normal"]


@dataclass
class Segment:
    label: str
    start: float
    end: float
    confidence: float
    peak_person_count: int
    avg_motion_magnitude: float


def _windows_to_raw_segments(windows, features, label):
    segments = []
    current = None
    for window, feats in zip(windows, features):
        score = feats.class_scores.get(label, 0.0)
        if score >= ACTIVE_THRESHOLD:
            if current is None:
                current = {
                    "start": window.start,
                    "end": window.end,
                    "scores": [score],
                    "persons": [feats.person_count],
                    "motion": [feats.motion_magnitude],
                }
            else:
                current["end"] = window.end
                current["scores"].append(score)
                current["persons"].append(feats.person_count)
                current["motion"].append(feats.motion_magnitude)
        elif current is not None:
            segments.append(current)
            current = None
    if current is not None:
        segments.append(current)
    return segments


def _bridge_gaps(raw_segments):
    if not raw_segments:
        return []
    bridged = [raw_segments[0]]
    for seg in raw_segments[1:]:
        last = bridged[-1]
        if seg["start"] - last["end"] <= MAX_GAP_TO_BRIDGE:
            last["end"] = seg["end"]
            last["scores"].extend(seg["scores"])
            last["persons"].extend(seg["persons"])
            last["motion"].extend(seg["motion"])
        else:
            bridged.append(seg)
    return bridged


def _finalize(raw_segments, label):
    segments = []
    for seg in raw_segments:
        if seg["end"] - seg["start"] < MIN_SEGMENT_DURATION:
            continue
        segments.append(Segment(
            label=label,
            start=seg["start"],
            end=seg["end"],
            confidence=sum(seg["scores"]) / len(seg["scores"]),
            peak_person_count=max(seg["persons"]),
            avg_motion_magnitude=sum(seg["motion"]) / len(seg["motion"]),
        ))
    return segments


def _idle_gaps(windows, segments, video_duration):
    if not windows:
        return []
    active_ranges = sorted((s.start, s.end) for s in segments)
    merged = []
    for start, end in active_ranges:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    gaps = []
    cursor = 0.0
    for start, end in merged:
        if start - cursor >= MIN_SEGMENT_DURATION:
            gaps.append(Segment("idle", cursor, start, 1.0, 0, 0.0))
        cursor = max(cursor, end)
    if video_duration - cursor >= MIN_SEGMENT_DURATION:
        gaps.append(Segment("idle", cursor, video_duration, 1.0, 0, 0.0))
    return gaps


def merge_segments(windows, features, video_duration):
    segments = []
    for label in ACTIVITY_CLASSES:
        raw = _windows_to_raw_segments(windows, features, label)
        raw = _bridge_gaps(raw)
        segments.extend(_finalize(raw, label))

    segments.sort(key=lambda s: s.start)
    segments.extend(_idle_gaps(windows, segments, video_duration))
    segments.sort(key=lambda s: s.start)
    return segments
