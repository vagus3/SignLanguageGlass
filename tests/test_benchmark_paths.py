"""A/B placement benchmark: both paths measured on the same clips, server and model."""
import _path  # noqa: F401
import io
import json
import sys
import tempfile
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
from src.cloud_sign import CloudClient, GlossClient
from src.continuous_sign import CtcRecognizer, Recognizer, save_checkpoint
from src.ctc_runtime import OnnxRecognizer
from src.pose_sequence import PoseSequence
from src.sign_server import DeviceModel, MockTranslator, TranslationService, create_app
from tools.benchmark_paths import benchmark, load_clips, make_dummy_model, table
from tools.export_ctc_onnx import export

labels = ["<blank>", "G0", "G1"]
rng = np.random.default_rng(8)
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    model = CtcRecognizer(len(labels), hidden=16)
    with torch.no_grad():
        model.output.weight.zero_()
        model.output.bias.copy_(torch.tensor([0., 0., 12.]))
    save_checkpoint(td / "ctc.pt", model, labels)
    export(td / "ctc.pt", td / "ctc.onnx")
    clips = []
    for i, n in enumerate((40, 90, 150)):
        PoseSequence(np.arange(n) / 30, rng.uniform(.1, .9, (n, 49, 2)), np.ones((n, 49), bool),
                     (1280, 720)).save(td / f"clip_{i}.npz")
        clips.append({"id": f"c{i}", "file": f"clip_{i}.npz"})
    (td / "manifest.json").write_text(json.dumps({"clips": clips}), encoding="utf-8")

    # Same as the server's --mock-language: real recognizers, mock composer, never spoken.
    service = TranslationService(Recognizer(td / "ctc.pt"), MockTranslator(),
                                 {"G0": "병원", "G1": "안녕하세요"}, mock=True,
                                 device_model=DeviceModel(td / "ctc.json"))
    api = TestClient(create_app(service, token=""))

    def urlopen(request, timeout):
        response = api.post(request.full_url.split("8765")[1], content=request.data,
                            headers=dict(request.header_items()))
        if response.status_code >= 400:
            raise urllib.error.HTTPError(request.full_url, response.status_code, "x", {}, None)
        return io.BytesIO(response.content)

    with patch("urllib.request.urlopen", side_effect=urlopen):
        report = benchmark(load_clips(td / "manifest.json"),
                           {"A": CloudClient(), "B": GlossClient(OnnxRecognizer(td / "ctc.onnx"))},
                           warmup=1)
    a, b = report["paths"]["A"], report["paths"]["B"]
    assert a["segments"] == b["segments"] == 3
    assert a["statuses"] == b["statuses"] == {"ok": 3}, (a["statuses"], b["statuses"])
    assert report["gloss_agreement"] == {"same": 3, "compared": 3}
    # Pose bodies are kilobytes even gzipped; gloss bodies are a few hundred bytes.
    assert a["sent_bytes"]["p50"] > 20 * b["sent_bytes"]["p50"]
    assert all(r["server_ms"].get("recognize") is not None for r in report["records"]["A"])
    assert all("recognize" not in r["server_ms"] for r in report["records"]["B"])
    assert all(r["status"] != "error" for rs in report["records"].values() for r in rs)
    text = table(report)
    assert "A/B glosses identical on 3/3" in text
    assert load_clips(td / "manifest.json", limit=2)[1][0] == "c1"
    dummy = make_dummy_model(td / "manifest.json", td / "dummy")
    assert DeviceModel(dummy / "ctc.json").labels == ["<blank>", "G0"]
    assert OnnxRecognizer(dummy / "ctc.onnx").predict(load_clips(td / "manifest.json")[0][1])[0] == ["G0"]
    try:
        make_dummy_model(td / "manifest.json", td / "dummy")
        raise AssertionError("Overwrote an existing dummy model")
    except ValueError:
        pass
print("[OK] A/B benchmark: same clips/server/model, statuses, sizes, server vs device timing")
