import json
import sys
from datetime import datetime
from pathlib import Path

from incident_investigator.ingestion import ingest_video
from incident_investigator.features import extract_window_features
from incident_investigator.segments import merge_segments
from incident_investigator.cases import bundle_cases
from incident_investigator.memory import init_db, seed_demo_data, attach_memory, ShortTermMemory
from incident_investigator.agent import evaluate_case, reflect_case
from incident_investigator.report import build_report, render_markdown

DEFAULT_CLIPS = [
    "data/raw/Fighting/two_boys_fighting_10505848.mp4",
    "data/raw/Vandalism/man_vandalizing_glass_7120813.mp4",
    "data/raw/Arson/drone_fire_and_smoke_11584957.mp4",
    "data/raw/Normal/pedestrians_crossing_antwerp_14413746.mp4",
]

REFERENCE_TIME = datetime(2026, 9, 12, 22, 0, 0)
OUTPUT_DIR = Path("reports")


def main():
    paths = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_CLIPS
    OUTPUT_DIR.mkdir(exist_ok=True)

    conn = init_db()
    seed_demo_data(conn)

    for path in paths:
        result = ingest_video(path)
        video_id = Path(path).stem
        print(f"\n=== {path} ===")
        if result.error:
            print(f"error: {result.error}")
            continue

        short_term = ShortTermMemory()
        features = [extract_window_features(w) for w in result.windows]
        segments = merge_segments(result.windows, features, result.duration)
        cases = bundle_cases(segments, camera_id="cam_01")
        attach_memory(cases, conn, short_term, reference_time=REFERENCE_TIME)

        verdicts = []
        reflections = []
        for case in cases:
            verdict = evaluate_case(case)
            reflections.append(reflect_case(case, verdict))
            verdicts.append(verdict)

        report = build_report(
            video_id=video_id,
            source_path=path,
            camera_id="cam_01",
            duration=result.duration,
            segments=segments,
            cases=cases,
            verdicts=verdicts,
            reflections=reflections,
        )

        json_path = OUTPUT_DIR / f"{video_id}.json"
        md_path = OUTPUT_DIR / f"{video_id}.md"
        json_path.write_text(json.dumps(report, indent=2))
        md_path.write_text(render_markdown(report))

        print(f"wrote {json_path} and {md_path}")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
