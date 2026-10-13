"""Path B: device CTC (ONNX) -> /v1/sign/compose -> same decisions as server recognition."""
import _path  # noqa: F401
import io
import sys
import tempfile
import time
import urllib.error
from pathlib import Path
from unittest.mock import patch
import numpy as np

try:
    import onnx  # noqa: F401
    import torch
    from fastapi.testclient import TestClient
except ImportError:
    sys.exit(77)
from src import clock
from src.cloud_sign import CloudClient, CloudWorker, GlossClient
from src.continuous_sign import CtcRecognizer, Recognizer, save_checkpoint
from src.ctc_runtime import GLOSS_VERSION, OnnxRecognizer
from src.latency import LatencyTracker
from src.pose_sequence import PoseSequence
from src.sign_server import DeviceModel, MockTranslator, TranslationService, create_app
from src.translation import Glossary
from tools.export_ctc_onnx import export

labels = ["<blank>", "G0", "G1", "G2"]
glossary = Glossary.from_mapping({"G0": "병원", "G1": "안녕하세요", "G2": "어디"})
rng = np.random.default_rng(5)
sequence = PoseSequence(np.arange(40) / 30, rng.uniform(.1, .9, (40, 49, 2)),
                        np.ones((40, 49), bool), (1280, 720))


def checkpoint(path, calibration=None):
    model = CtcRecognizer(len(labels), hidden=16)
    with torch.no_grad():   # always "G1": a deterministic, nonempty prediction
        model.output.weight.zero_()
        model.output.bias.copy_(torch.tensor([0., 0., 12., 0.]))
    save_checkpoint(path, model, labels, calibration=calibration)


def routed(client):
    def urlopen(request, timeout):
        response = client.post(request.full_url.split("8765")[1], content=request.data,
                               headers=dict(request.header_items()))
        if response.status_code >= 400:
            raise urllib.error.HTTPError(request.full_url, response.status_code, "error", {}, None)
        return io.BytesIO(response.content)
    return urlopen


with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    calibration = {"enabled": True, "threshold": 0.5, "negative_seconds": 600,
                   "reviewer": "fixture", "mapping_hash": glossary.policy_hash(),
                   "language_model": MockTranslator.model}
    checkpoint(td / "ctc.pt", calibration)
    export(td / "ctc.pt", td / "ctc.onnx")
    device_model = DeviceModel(td / "ctc.onnx")
    assert DeviceModel(td / "ctc.json").metadata == device_model.metadata
    service = TranslationService(Recognizer(td / "ctc.pt"), MockTranslator(), glossary,
                                 device_model=device_model)
    api = TestClient(create_app(service, token=""))
    health = api.get("/health").json()
    assert health["device_model_ready"] and health["device_auto_speech"] and health["auto_speech"]

    # The same segment through A (server PyTorch) and B (device ONNX) decides the same.
    with patch("urllib.request.urlopen", side_effect=routed(api)):
        a = CloudClient()
        b = GlossClient(OnnxRecognizer(td / "ctc.onnx"))
        result_a = a.translate(a.payload(sequence, "s-a", "seg"))
        stats = LatencyTracker()
        body = b.payload(sequence, "s-b", "seg", stats)
        assert body["version"] == GLOSS_VERSION and body["glosses"] == ["G1"]
        assert body["recognizer"] == device_model.metadata["model_id"]
        assert len(b.encode(body)[0]) < 400 and "Content-Encoding" not in b.encode(body)[1]
        assert stats.details()["device_recognize"]["count"] == 1
        result_b = b.translate(body)
        for key in ("status", "glosses", "text", "source", "auto_speech"):
            assert result_a[key] == result_b[key], key
        assert result_b["status"] == "ok" and result_b["auto_speech"] and not result_b["mock"]
        assert "recognize" in result_a["timings_ms"] and "recognize" not in result_b["timings_ms"]

        # The worker builds the body (recognition) off the caller's thread.
        worker = CloudWorker(b, stats=stats)
        worker.start()
        assert worker.submit(sequence, clock.now())
        end, received = time.monotonic() + 5, None
        while received is None and time.monotonic() < end:
            received = worker.poll()
            time.sleep(.01)
        worker.stop()
        worker.join(1)
        assert received[0]["status"] == "ok" and stats.details()["device_recognize"]["count"] == 2

    def post(**change):
        base = {"version": GLOSS_VERSION, "session_id": "direct", "segment_id": "x",
                "glosses": ["G1"], "score": .9, "recognizer": device_model.metadata["model_id"]}
        return api.post("/v1/sign/compose", json={**base, **change})

    assert post(segment_id="low", score=.3).json()["status"] == "uncertain"   # threshold .5
    assert post(segment_id="none", glosses=[], score=0).json()["status"] == "no_sign"
    assert post(segment_id="other", recognizer="0123456789abcdef").status_code == 409
    assert post(segment_id="vocab", glosses=["G9"]).status_code == 422
    for bad in ({"score": 1.5}, {"version": "pose2d-sequence-v1"}, {"glosses": [""]},
                {"glosses": ["G1"] * 65}, {"extra": 1}):
        assert post(segment_id="bad", **bad).status_code == 422, bad
    first = post(segment_id="dup").json()
    assert post(segment_id="dup").json() == first
    assert post(segment_id="dup", glosses=["G0"]).status_code == 409

    # A server without the device model answers unavailable; mock results never speak.
    bare = TestClient(create_app(TranslationService(None, MockTranslator(), glossary), token=""))
    response = bare.post("/v1/sign/compose", json={
        "version": GLOSS_VERSION, "session_id": "s", "segment_id": "x", "glosses": ["G1"],
        "score": .9, "recognizer": "abc"})
    assert response.status_code == 200 and response.json()["status"] == "unavailable"
    mock = TranslationService(None, MockTranslator(), glossary, mock=True, device_model=device_model)
    assert not mock.device_speech_enabled
    try:
        GlossClient(type("NoId", (), {"metadata": {}})())
        raise AssertionError("A device model without model_id was accepted")
    except ValueError:
        pass
print("[OK] gloss API: device ONNX and server PyTorch decide alike, threshold/vocabulary/"
      "model checks, duplicates, worker-thread recognition, tiny uncompressed bodies")
