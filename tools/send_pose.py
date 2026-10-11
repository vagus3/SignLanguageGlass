#!/usr/bin/env python
"""Send a saved pose clip without camera/microphone/OLED; mock is clearly identified."""
import argparse
import json
from pathlib import Path
import sys
import uuid
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cloud_sign import CloudClient
from src.pose_sequence import PoseSequence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pose")
    args = parser.parse_args()
    payload = {**PoseSequence.load(args.pose).to_dict(),
               "session_id": uuid.uuid4().hex, "segment_id": uuid.uuid4().hex}
    print(json.dumps(CloudClient().translate(payload), ensure_ascii=False, indent=2))
