"""Collect complete pose sentences without saving large videos.

python -m src.collect_continuous --glosses 이름 무엇 --text '이름이 뭐예요?'
SPACE starts/ends capture; c cancels; q quits. Empty --glosses records a negative.
"""
import argparse
import json
from pathlib import Path
import time
import uuid
import config as C
from src.pose_sequence import (VERSION, VERSION_3D, JOINT_NAMES, from_mediapipe,
                               world_from_mediapipe)
from src.cloud_sign import SegmentRecorder


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(C.DATA_DIR / "continuous"))
    p.add_argument("--glosses", nargs="*", required=True)
    p.add_argument("--text", default="")
    p.add_argument("--signer", default=C.SIGNER_ID)
    p.add_argument("--session", default=C.SESSION_ID)
    p.add_argument("--camera", type=int, default=C.CAM_INDEX)
    p.add_argument("--with-3d", action="store_true",
                   help="also record Holistic 3D points (pose3d-sequence-v2 dataset)")
    args = p.parse_args()
    if any(not v.strip() for v in (args.signer, args.session)):
        p.error("signer/session must not be empty")
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    version = VERSION_3D if args.with_3d else VERSION
    manifest = (json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists()
                else {"version": version, "joint_names": JOINT_NAMES, "clips": []})
    if manifest["version"] != version:
        p.error("Existing dataset uses a different pose contract")
    import cv2
    from src.camera import CameraReader
    from src.landmarks import make_holistic
    cap = CameraReader(camera_index=args.camera)
    if not cap.ok():
        raise SystemExit("Cannot open camera")
    recorder = SegmentRecorder()
    holistic = make_holistic(C.MODEL_COMPLEXITY)
    started = None
    print("SPACE: start/end; c: cancel; q: quit. Label:", args.glosses)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            captured = time.monotonic()
            h, w = frame.shape[:2]
            res = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            points, mask = from_mediapipe(res)
            world = world_from_mediapipe(res, (w, h)) if args.with_3d else (None, None)
            recorder.add(points, mask, captured, (w, h), *world)
            status = "REC" if recorder.active else ("CANCELLED: " + recorder.error
                                                      if recorder.error else "READY")
            cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 255), 2)
            cv2.imshow("continuous pose collector", frame)
            key = cv2.waitKey(1) & 0xff
            if key == ord("q"):
                break
            if key == ord("c"):
                recorder.cancel()
            if key == ord(" "):
                if recorder.active:
                    try:
                        sequence, _ = recorder.finish()
                    except ValueError as e:
                        print("Not saved:", e)
                        continue
                    identity = uuid.uuid4().hex
                    filename = identity + ".npz"
                    sequence.save(root / filename)
                    manifest["clips"].append({
                        "id": identity, "file": filename, "signer_id": args.signer,
                        "session_id": args.session, "group_id": identity, "continuous": True,
                        "glosses": args.glosses, "text": args.text, "negative": not args.glosses,
                        "capture_start": started,
                        "duration_seconds": float(sequence.timestamps[-1]),
                        "signer_expression_available": False})
                    temporary = root / "manifest.tmp"
                    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
                    temporary.replace(manifest_path)
                    print("Saved", filename)
                else:
                    recorder.start()
                    started = time.time()
    finally:
        cap.release()
        holistic.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
