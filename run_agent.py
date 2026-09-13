import sys
from datetime import datetime

from incident_investigator.ingestion import ingest_video
from incident_investigator.features import extract_window_features
from incident_investigator.segments import merge_segments
from incident_investigator.cases import bundle_cases
from incident_investigator.memory import init_db, seed_demo_data, attach_memory, ShortTermMemory
from incident_investigator.agent import evaluate_case

DEFAULT_CLIPS = [
    "data/raw/Fighting/two_boys_fighting_10505848.mp4",
    "data/raw/Vandalism/man_vandalizing_glass_7120813.mp4",
    "data/raw/Arson/drone_fire_and_smoke_11584957.mp4",
    "data/raw/Normal/pedestrians_crossing_antwerp_14413746.mp4",
]

REFERENCE_TIME = datetime(2026, 9, 12, 22, 0, 0)


def main():
    paths = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_CLIPS

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
                print(f"    NEEDS REVIEW: {verdict.error}")
                if verdict.raw_response:
                    print(f"    raw response: {verdict.raw_response[:300]}")
                continue

            for seg in verdict.segments:
                print(f"    {seg.segment_class}: revised_confidence={seg.revised_confidence:.2f} "
                      f"status={seg.corroboration_status}")
                print(f"      reasoning: {seg.reasoning}")


if __name__ == "__main__":
    main()
