import sys
from datetime import datetime

from incident_investigator.orchestrator import run_pipeline

DEFAULT_CLIP = "data/raw/Fighting/two_boys_fighting_10505848.mp4"
REFERENCE_TIME = datetime(2026, 9, 12, 22, 0, 0)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CLIP
    log = run_pipeline(path, camera_id="cam_01", reference_time=REFERENCE_TIME)

    print(f"video_id: {log['video_id']}")
    print(f"timing: {log['timing']}")
    print(f"llm_calls: {log['llm_calls']}")
    print(f"errors: {log['errors']}")
    print(f"warnings: {log['warnings']}")
    print(f"stage1_ingestion: {log['stage1_ingestion']}")
    print(f"stage2_features: {len(log.get('stage2_features', []))} windows logged "
          f"(showing first) -> {log['stage2_features'][0] if log.get('stage2_features') else None}")
    print(f"stage3_segments: {log.get('stage3_segments')}")
    print(f"stage4_cases: {log.get('stage4_cases')}")
    print(f"stage5_verdicts: {log.get('stage5_verdicts')}")
    print(f"stage6_reflections: {log.get('stage6_reflections')}")
    print(f"stage7_report_path: {log.get('stage7_report_path')}")


if __name__ == "__main__":
    main()
