"""Conversion preserves actual time, explicit camera size and text-only exclusions."""
import _path  # noqa
import sys
try:
    import torch  # noqa
except ImportError:
    sys.exit(77)
import json
import tempfile
from pathlib import Path
import numpy as np
from tools.prepare_continuous import prepare
from src.pose_sequence import PoseSequence, VERSION_3D
from src.continuous_sign import load_dataset

with tempfile.TemporaryDirectory() as td:
    root=Path(td)
    clips=[]
    for index in range(2):
        folder=root/f"input{index}"
        folder.mkdir()
        for frame in [2,3,5]:
            hand=np.tile([100,200,1],(21,1))
            person={"hand_left_keypoints_2d":hand.ravel().tolist(),
                    "hand_right_keypoints_2d":hand.ravel().tolist()}
            (folder/f"clip_{frame:012d}_keypoints.json").write_text(json.dumps({"people":person}))
        clips.append({"id":str(index),"keypoints_dir":folder.name,"signer_id":"s",
                      "session_id":"day1","group_id":str(index),
                      "glosses":["A"] if index==0 else None,"text":"reference"})
    annotation=root/"annotations.json"
    source={"image_size":[640,480],"fps":30,"clips":clips}
    annotation.write_text(json.dumps(source))
    report=prepare(annotation,root/"out")
    assert len(report["excluded_from_ctc"])==1
    s=PoseSequence.load(root/"out/clip_000000.npz")
    assert np.allclose(s.timestamps,[0,1/30,3/30]) and len(s.timestamps)==3
    rows,skipped=load_dataset(root/"out/manifest.json")
    assert len(rows)==1 and len(skipped)==1
    try:
        prepare(annotation,root/"out")
        raise AssertionError("Existing output overwritten")
    except ValueError:
        pass
    try:
        prepare(annotation,root/"out3d",with_world=True)
        raise AssertionError("2D-only keypoints became a 3D dataset")
    except ValueError:
        pass
    for path in root.glob("input*/*_keypoints.json"):
        person=json.loads(path.read_text())["people"]
        person["hand_left_keypoints_3d"]=np.tile([.1,.2,2,1],(21,1)).ravel().tolist()
        path.write_text(json.dumps({"people":person}))
    report=prepare(annotation,root/"out3d",with_world=True)
    assert report["version"]==VERSION_3D
    rows,_=load_dataset(root/"out3d/manifest.json")
    assert rows[0]["sequence"].world_mask[:,:21].all() and not rows[0]["sequence"].world_mask[:,21:].any()
    del source["image_size"]
    annotation.write_text(json.dumps(source))
    try:
        prepare(annotation,root/"bad")
        raise AssertionError("Image size silently inferred")
    except ValueError:
        pass
    assert not (root/"bad").exists(),"Partial output must not become a dataset"
print("[OK] explicit calibration, irregular timing, missing gloss report, output protection")
