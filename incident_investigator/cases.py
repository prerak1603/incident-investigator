from dataclasses import dataclass, field

CASE_PROXIMITY = 20.0


@dataclass
class Case:
    segments: list
    start: float
    end: float
    classes: list
    camera_id: str
    historical_fp_rate: dict = field(default_factory=dict)
    related_cases: list = field(default_factory=list)


def bundle_cases(segments, camera_id="cam_01"):
    incidents = sorted((s for s in segments if s.label != "idle"), key=lambda s: s.start)
    if not incidents:
        return []

    groups = [[incidents[0]]]
    for seg in incidents[1:]:
        group_end = max(s.end for s in groups[-1])
        if seg.start - group_end <= CASE_PROXIMITY:
            groups[-1].append(seg)
        else:
            groups.append([seg])

    return [
        Case(
            segments=group,
            start=min(s.start for s in group),
            end=max(s.end for s in group),
            classes=sorted({s.label for s in group}),
            camera_id=camera_id,
        )
        for group in groups
    ]
