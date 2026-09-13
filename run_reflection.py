import sys
from datetime import datetime

from incident_investigator.ingestion import ingest_video
from incident_investigator.features import extract_window_features
from incident_investigator.segments import merge_segments, Segment
from incident_investigator.cases import bundle_cases, Case
from incident_investigator.memory import init_db, seed_demo_data, attach_memory, ShortTermMemory
from incident_investigator.agent import evaluate_case, reflect_case, SegmentVerdict, CaseVerdict

DEFAULT_CLIPS = [
    "data/raw/Fighting/two_boys_fighting_10505848.mp4",
    "data/raw/Vandalism/man_vandalizing_glass_7120813.mp4",
    "data/raw/Arson/drone_fire_and_smoke_11584957.mp4",
    "data/raw/Normal/pedestrians_crossing_antwerp_14413746.mp4",
]

REFERENCE_TIME = datetime(2026, 9, 12, 22, 0, 0)


def print_reflection(verdict, reflection):
    if reflection.needs_review:
        print(f"    NEEDS REVIEW: {reflection.error}")
        if reflection.raw_response:
            print(f"    raw response: {reflection.raw_response[:300]}")
        return

    for seg in reflection.segments:
        changed = abs(seg.final_confidence - seg.pre_confidence) > 1e-6
        marker = "CHANGED" if changed else "confirmed"
        print(f"    {seg.segment_class}: pre={seg.pre_confidence:.2f}/{seg.pre_status} -> "
              f"final={seg.final_confidence:.2f}/{seg.final_verdict}  [{marker}]")
        print(f"      reflection_note: {seg.reflection_note}")


def run_real_clips(paths):
    conn = init_db()
    seed_demo_data(conn)

    for path in paths:
        result = ingest_video(path)
        print(f"\n=== {path} ===")
        if result.error:
            print(f"error: {result.error}")
            continue

        short_term = ShortTermMemory()
        features = [extract_window_features(w) for w in result.windows]
        segments = merge_segments(result.windows, features, result.duration)
        cases = bundle_cases(segments, camera_id="cam_01")
        attach_memory(cases, conn, short_term, reference_time=REFERENCE_TIME)

        if not cases:
            print("  no incident cases")
            continue

        for case in cases:
            print(f"\n  case [{case.start:.1f}-{case.end:.1f}s] classes={case.classes}")
            verdict = evaluate_case(case)
            if verdict.needs_review:
                print(f"    stage 5 NEEDS REVIEW: {verdict.error}")
                continue
            reflection = reflect_case(case, verdict)
            print_reflection(verdict, reflection)


def run_synthetic_case():
    # hand-built, not from real footage: stage 5 is deliberately given a verdict
    # whose reasoning contradicts its own confidence/status, to check reflection
    # actually catches that rather than just rubber-stamping every case
    print("\n=== synthetic case: constructed to test the reflection catch ===")

    segment = Segment(
        label="fighting",
        start=10.0,
        end=14.0,
        confidence=0.88,
        peak_person_count=1,
        avg_motion_magnitude=15.0,
    )
    case = Case(
        segments=[segment],
        start=10.0,
        end=14.0,
        classes=["fighting"],
        camera_id="cam_99",
        historical_fp_rate={"fighting": 0.7},
        related_cases=[],
    )
    verdict = CaseVerdict(
        case=case,
        segments=[SegmentVerdict(
            segment_class="fighting",
            revised_confidence=0.93,
            corroboration_status="insufficient_evidence",
            reasoning=(
                "Only one person was ever detected in frame, which is inconsistent with a "
                "fighting event that by definition requires at least two participants. There "
                "is no corroborating evidence and the historical false positive rate for this "
                "camera/class/hour is 0.70. The evidence here is genuinely insufficient to "
                "confirm this is a real fighting incident."
            ),
        )],
        raw_response="",
    )

    print(f"\n  case [{case.start:.1f}-{case.end:.1f}s] classes={case.classes}")
    print(f"    stage 5 (hand-built): revised_confidence={verdict.segments[0].revised_confidence:.2f} "
          f"status={verdict.segments[0].corroboration_status}")
    reflection = reflect_case(case, verdict)
    print_reflection(verdict, reflection)


def main():
    paths = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_CLIPS
    run_real_clips(paths)
    run_synthetic_case()


if __name__ == "__main__":
    main()
