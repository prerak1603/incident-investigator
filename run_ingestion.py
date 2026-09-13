import sys

from incident_investigator.ingestion import ingest_video

DEFAULT_CLIP = "data/raw/Fighting/two_boys_fighting_10505848.mp4"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CLIP
    result = ingest_video(path)

    print(f"video: {result.video_path}")
    print(f"fps: {result.fps:.2f}")
    print(f"duration: {result.duration:.2f}s")
    print(f"frames sampled: {result.frame_count}")
    print(f"windows: {len(result.windows)}")

    for w in result.warnings:
        print(f"warning: {w}")

    if result.error:
        print(f"error: {result.error}")
        return

    for i, window in enumerate(result.windows):
        timestamps = [round(f.timestamp, 2) for f in window.frames]
        print(f"  window {i}: [{window.start:.2f}, {window.end:.2f}) frames={len(window.frames)} timestamps={timestamps}")


if __name__ == "__main__":
    main()
