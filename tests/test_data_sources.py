"""Data sources travel manifest -> checkpoint -> ONNX sidecar -> server /health (AI Hub terms)."""
import _path  # noqa: F401
import json
import sys
import tempfile
from pathlib import Path
import numpy as np

from src.data_sources import SOURCES, attribution, manifest_sources, set_sources, source_entry

assert source_entry("aihub-sign-video")["url"] == "https://aihub.or.kr/aidata/7965"
assert "한국지능정보사회진흥원" in SOURCES["aihub-sign-video"]["attribution"]
assert all(not s["redistribution"] for s in SOURCES.values()), "no source may be passed on"
try:
    source_entry("somewhere")
    raise AssertionError("Unknown source accepted")
except ValueError:
    pass
both = [source_entry("aihub-sign-video"), source_entry("team"), source_entry("aihub-sign-video")]
assert attribution(both) == [SOURCES["aihub-sign-video"]["attribution"]], "deduplicated, team silent"
assert manifest_sources({}) == [] and manifest_sources({"sources": both}) == both
try:
    manifest_sources({"sources": ["aihub"]})
    raise AssertionError("Malformed sources accepted")
except ValueError:
    pass

# Datasets converted before sources existed can be marked afterwards (python -m src.data_sources).
import subprocess
with tempfile.TemporaryDirectory() as td:
    old = Path(td) / "manifest.json"
    old.write_text(json.dumps({"version": "pose2d-sequence-v1", "clips": [{"id": "a"}]}),
                   encoding="utf-8")
    recorded = set_sources(old, ["aihub-sign-video", "aihub-sign-video", "team"])
    assert [x["id"] for x in recorded] == ["aihub-sign-video", "team"], "deduplicated, in order"
    again = json.loads(old.read_text(encoding="utf-8"))
    assert again["clips"] == [{"id": "a"}] and manifest_sources(again) == recorded
    assert not list(Path(td).glob("*.tmp")), "written atomically, no leftovers"
    try:
        set_sources(old, ["nowhere"])
        raise AssertionError("Unknown source recorded")
    except ValueError:
        pass
    done = subprocess.run([sys.executable, "-m", "src.data_sources", str(old), "--source", "team"],
                          capture_output=True, text=True)
    assert done.returncode == 0 and "팀 자체 수집" in done.stdout, done.stderr
    assert [x["id"] for x in manifest_sources(json.loads(old.read_text(encoding="utf-8")))] == ["team"]

try:
    import onnx  # noqa: F401
    import torch  # noqa: F401
    from fastapi.testclient import TestClient
except ImportError:
    print("[OK] data sources (training path skipped: torch/onnx/fastapi missing)")
    sys.exit(77)
from src.continuous_sign import Recognizer
from src.pose_sequence import PoseSequence
from src.sign_server import DeviceModel, MockTranslator, TranslationService, create_app
from src.train_continuous import dataset_sources, prepare_training, train_one
from tools.export_ctc_onnx import export

rng = np.random.default_rng(2)
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    clips = []
    for i in range(6):
        PoseSequence(np.arange(12) / 30, rng.uniform(.1, .9, (12, 49, 2)), np.ones((12, 49), bool),
                     (640, 480)).save(td / f"c{i}.npz")
        clips.append({"id": f"c{i}", "file": f"c{i}.npz", "glosses": ["A", "B"], "continuous": True,
                      "signer_id": f"s{i}", "session_id": "d", "group_id": f"g{i}"})
    (td / "manifest.json").write_text(json.dumps({
        "version": "pose2d-sequence-v1", "sources": [source_entry("aihub-sign-video")],
        "clips": clips}), encoding="utf-8")
    # Without sources, training still works but says the attribution will be missing.
    import contextlib
    import io
    bare = td / "bare.json"
    bare.write_text(json.dumps({"clips": []}), encoding="utf-8")
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        assert dataset_sources(bare) == []
    assert "출처" in printed.getvalue() and "python -m src.data_sources" in printed.getvalue()
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        dataset_sources(td / "manifest.json")
    assert printed.getvalue() == "", "no warning when sources are recorded"
    _, splits, labels = prepare_training(td / "manifest.json")
    train_one(splits, labels, "2d", td / "model", epochs=1,
              data_sources=dataset_sources(td / "manifest.json"))
    recognizer = Recognizer(td / "model" / "ctc.pt")
    assert recognizer.metadata["data_sources"][0]["id"] == "aihub-sign-video"
    export(td / "model" / "ctc.pt", td / "ctc.onnx")
    sidecar = json.loads((td / "ctc.json").read_text(encoding="utf-8"))
    assert sidecar["metadata"]["data_sources"][0]["id"] == "aihub-sign-video"
    service = TranslationService(recognizer, MockTranslator(), {"A": "가", "B": "나"},
                                 device_model=DeviceModel(td / "ctc.onnx"))
    health = TestClient(create_app(service, token="")).get("/health").json()
    assert health["data_sources"] == [SOURCES["aihub-sign-video"]["attribution"]], health
    assert TranslationService().data_sources() == []
print("[OK] data sources: AI Hub attribution recorded in manifests, checkpoints, ONNX "
      "sidecars and /health")
