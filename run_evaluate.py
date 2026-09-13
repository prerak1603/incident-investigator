import json
import sys
from datetime import datetime
from pathlib import Path

from incident_investigator.evaluate import (
    load_ground_truth, run_cv_evaluation, run_memory_ablation,
    summarize_ablation, render_ablation_markdown, plot_ablation,
)

DEFAULT_CLIPS = [
    "data/raw/Fighting/two_boys_fighting_10505848.mp4",
    "data/raw/Vandalism/man_vandalizing_glass_7120813.mp4",
    "data/raw/Arson/drone_fire_and_smoke_11584957.mp4",
    "data/raw/Normal/pedestrians_crossing_antwerp_14413746.mp4",
]

REFERENCE_TIME = datetime(2026, 9, 12, 22, 0, 0)
OUTPUT_DIR = Path("evaluation")


def main():
    paths = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_CLIPS
    OUTPUT_DIR.mkdir(exist_ok=True)

    ground_truth = load_ground_truth()
    cv_results = run_cv_evaluation(paths, ground_truth)

    ablation_rows = run_memory_ablation(paths, reference_time=REFERENCE_TIME)
    ablation_summary = summarize_ablation(ablation_rows)

    results = {
        "cv_evaluation": cv_results,
        "agent_evaluation": {
            "rows": ablation_rows,
            "summary": ablation_summary,
        },
    }

    (OUTPUT_DIR / "evaluation_results.json").write_text(json.dumps(results, indent=2))

    markdown = "# Evaluation Results\n\n"
    markdown += "## CV evaluation (Stage 3 segments vs ground truth)\n\n"
    for thresh, data in cv_results["map_by_iou_threshold"].items():
        markdown += f"- mAP@{thresh}: {data['mAP']}\n"
    markdown += "\n## Memory ablation\n\n"
    markdown += render_ablation_markdown(ablation_summary)
    markdown += "\n"
    (OUTPUT_DIR / "evaluation_summary.md").write_text(markdown)

    plot_ablation(ablation_summary, OUTPUT_DIR / "memory_ablation.png")

    print(json.dumps(results, indent=2))
    print(f"\nwrote {OUTPUT_DIR}/evaluation_results.json, evaluation_summary.md, memory_ablation.png")


if __name__ == "__main__":
    main()
