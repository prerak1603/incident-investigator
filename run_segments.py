import sys

from incident_investigator.ingestion import ingest_video
from incident_investigator.features import extract_window_features
from incident_investigator.segments import merge_segments

DEFAULT_CLIPS = [
    "data/raw/Fighting/two_boys_fighting_10505848.mp4",
    "data/raw/Vandalism/man_vandalizing_glass_7120813.mp4",
    "data/raw/Arson/drone_fire_and_smoke_11584957.mp4",
    "data/raw/Normal/pedestrians_crossing_antwerp_14413746.mp4",
]


def main():
    paths = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_CLIPS

    for path in paths:
        result = ingest_video(path)
        print(f"\n=== {path} ===")
        if result.error:
            print(f"error: {result.error}")
            continue

        features = [extract_window_features(w) for w in result.windows]
        segments = merge_segments(result.windows, features, result.duration)

        for seg in segments:
            print(f"  {seg.label:10s} [{seg.start:5.1f}-{seg.end:5.1f}s]  conf={seg.confidence:.2f}  "
                  f"peak_persons={seg.peak_person_count}  avg_motion={seg.avg_motion_magnitude:.2f}")


if __name__ == "__main__":
    main()
