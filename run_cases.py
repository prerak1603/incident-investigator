import sys
from datetime import datetime

from incident_investigator.ingestion import ingest_video
from incident_investigator.features import extract_window_features
from incident_investigator.segments import merge_segments
from incident_investigator.cases import bundle_cases
from incident_investigator.memory import init_db, seed_demo_data, attach_memory, ShortTermMemory

DEFAULT_CLIPS = [
    "data/raw/Fighting/two_boys_fighting_10505848.mp4",
    "data/raw/Vandalism/man_vandalizing_glass_7120813.mp4",
    "data/raw/Arson/drone_fire_and_smoke_11584957.mp4",
    "data/raw/Normal/pedestrians_crossing_antwerp_14413746.mp4",
]

# pinned to 22:00 so the seeded rows in memory.py actually line up with a case
# here — a real deployment would pass the clip's actual capture time instead
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

        # fresh short-term memory per clip: these are 4 unrelated videos, not one
        # continuous camera session, so nothing should carry over between them
        short_term = ShortTermMemory()
        features = [extract_window_features(w) for w in result.windows]
        segments = merge_segments(result.windows, features, result.duration)
        cases = bundle_cases(segments, camera_id="cam_01")
        attach_memory(cases, conn, short_term, reference_time=REFERENCE_TIME)

        if not cases:
            print("  no incident cases")
            continue

        for case in cases:
            print(f"  case [{case.start:.1f}-{case.end:.1f}s] classes={case.classes} "
                  f"fp_rate={case.historical_fp_rate} related_cases={len(case.related_cases)}")


if __name__ == "__main__":
    main()
