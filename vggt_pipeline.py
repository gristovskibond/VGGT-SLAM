import sys
from pathlib import Path

from batch_vggt_pipeline import run_studiox_scan_pipeline

if __name__ == "__main__":
    
    if len(sys.argv) < 1:
        print(
            "Usage: python3 vggt_pipeline.py <video_path>",
            file=sys.stderr,
        )
        sys.exit(1)
    
    video_path = Path(sys.argv[1])
    
    try:
        result = run_studiox_scan_pipeline(video_path, "output")
        print(result, flush=True)
    except Exception as e:
        print(f"Failed: {e}", file=sys.stderr, flush=True)
