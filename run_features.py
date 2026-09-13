import sys

from incident_investigator.ingestion import ingest_video
from incident_investigator.features import extract_window_features

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

        for i, window in enumerate(result.windows):
            features = extract_window_features(window)
            scores = {k: round(v, 3) for k, v in features.class_scores.items()}
            print(f"  window {i} [{window.start:.1f}-{window.end:.1f}s]  scores={scores}  "
                  f"persons={features.person_count}  motion={features.motion_magnitude:.2f}")


if __name__ == "__main__":
    main()
