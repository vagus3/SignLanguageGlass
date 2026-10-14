"""Integrated SPACE -> capture -> API -> HUD path without opening real hardware."""
import _path  # noqa
import sys
try:
    from fastapi.testclient import TestClient
except ImportError:
    sys.exit(77)
import contextlib
import io
import os
import time
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
import main
from src.sign_server import create_app, TranslationService, MockRecognizer, MockTranslator

service=TranslationService(MockRecognizer(),MockTranslator(),{"안녕하세요":"안녕하세요"},mock=True)
api=TestClient(create_app(service,token=""))
camera_state={"released":False,"closed":False,"phase":0,"started":None,"calls":0,
              "frames":0,"mark":0}
# Steps wait for processed frames and printed results, not fixed delays: a slow CI machine
# only takes longer. LIMIT still ends a broken flow.
LIMIT=30.0
RESULT=("[cloud 번역]","[cloud 미리보기]","[cloud] 요청 실패")
def answered():
    return any(marker in output.getvalue() for marker in RESULT)
class Camera:
    def __init__(self,**kwargs):
        pass
    def ok(self):
        return True
    def read(self):
        time.sleep(.01)
        return True,np.zeros((120,160,3),np.uint8)
    def release(self):
        camera_state["released"]=True
class Holistic:
    def process(self,rgb):
        camera_state["frames"]+=1
        hand=NS(landmark=[NS(x=.2+i*.01,y=.3+i*.01) for i in range(21)])
        return NS(left_hand_landmarks=hand,right_hand_landmarks=hand,
                  pose_landmarks=None,face_landmarks=None)
    def close(self):
        camera_state["closed"]=True
def wait_key(delay):
    """SPACE, record at least 8 processed frames, SPACE, quit once the result is printed."""
    phase=camera_state["phase"]
    if phase==0:
        camera_state.update(phase=1,started=time.monotonic(),mark=camera_state["frames"])
        return ord(" ")
    if phase==1 and camera_state["frames"]-camera_state["mark"]>=8:
        camera_state["phase"]=2
        return ord(" ")
    if phase==2 and answered() or time.monotonic()-camera_state["started"]>LIMIT:
        return ord("q")
    return -1
def urlopen(request,timeout):
    camera_state["calls"]+=1
    r=api.post("/v1/sign/translate",content=request.data,headers=dict(request.header_items()))
    assert r.status_code==200
    return io.BytesIO(r.content)
output=io.StringIO()
with patch.object(sys,"argv",["main.py","--sign-backend","cloud","--display","null",
                             "--no-stt","--no-tts","--fixed-caption"]), \
        patch("src.camera.CameraReader",Camera), \
        patch("src.landmarks.make_holistic",lambda _:Holistic()), \
        patch("cv2.imshow"), patch("cv2.destroyAllWindows"), patch("cv2.waitKey",wait_key), \
        patch("urllib.request.urlopen",side_effect=urlopen), \
        patch.dict(os.environ,{"SIGN_SERVER_URL":"http://127.0.0.1:8765","SIGN_SERVER_TOKEN":""}), \
        contextlib.redirect_stdout(output):
    main.main()
assert camera_state["calls"]==1
assert "[cloud 번역] 안녕하세요." in output.getvalue()
assert "텍스트만 표시" in output.getvalue()
assert camera_state["released"] and camera_state["closed"]

# --auto-segment: hands move for 1.3 s then leave the view; no SPACE is pressed.
# Mid-segment the system clock is pulled back an hour; capture timing must not care.
# Times count from the first processed frame, so a slow start-up cannot skip the motion.
began = time.monotonic()
first = {"t": None}
real_time = time.time
def corrected_wall_clock():
    return real_time() - (3600 if first["t"] and time.monotonic() - first["t"] > .9 else 0)
versions = []
class MovingHolistic(Holistic):
    def process(self,rgb):
        camera_state["frames"]+=1
        first["t"]=first["t"] or time.monotonic()
        t=time.monotonic()-first["t"]
        if not .3<t<1.6:
            return NS(left_hand_landmarks=None,right_hand_landmarks=None,
                      pose_landmarks=None,pose_world_landmarks=None,face_landmarks=None)
        dx=.1*np.sin(2*np.pi*1.5*t)
        hand=NS(landmark=[NS(x=.3+dx+i*.01,y=.3+i*.01,z=0) for i in range(21)])
        return NS(left_hand_landmarks=hand,right_hand_landmarks=None,
                  pose_landmarks=None,pose_world_landmarks=None,face_landmarks=None)
def quit_later(delay):
    return ord("q") if answered() or time.monotonic()-began>LIMIT else -1
def urlopen_3d(request,timeout):
    import gzip
    import json
    versions.append(json.loads(gzip.decompress(request.data))["version"])
    return urlopen(request,timeout)
camera_state.update(calls=0)
output=io.StringIO()
with patch.object(sys,"argv",["main.py","--sign-backend","cloud","--display","null",
                             "--no-stt","--no-tts","--fixed-caption","--auto-segment",
                             "--pose-3d"]), \
        patch("src.camera.CameraReader",Camera), \
        patch("src.landmarks.make_holistic",lambda _:MovingHolistic()), \
        patch("cv2.imshow"), patch("cv2.destroyAllWindows"), patch("cv2.waitKey",quit_later), \
        patch("urllib.request.urlopen",side_effect=urlopen_3d), \
        patch("time.time",corrected_wall_clock), \
        patch.dict(os.environ,{"SIGN_SERVER_URL":"http://127.0.0.1:8765","SIGN_SERVER_TOKEN":""}), \
        contextlib.redirect_stdout(output):
    main.main()
assert camera_state["calls"]==1, output.getvalue()
assert versions==["pose3d-sequence-v2"]
assert "[cloud 번역] 안녕하세요." in output.getvalue()

# Sentence generation fails on the server: the result is a preview, never a spoken translation.
class Unreachable:
    model="fixture-model"
    def translate(self,glosses,mapping):
        raise TimeoutError("upstream timed out")
api=TestClient(create_app(TranslationService(MockRecognizer(),Unreachable(),
                                             {"안녕하세요":"안녕하세요"}),token=""))
camera_state.update(phase=0,calls=0)
output=io.StringIO()
with patch.object(sys,"argv",["main.py","--sign-backend","cloud","--display","null",
                             "--no-stt","--no-tts","--fixed-caption"]), \
        patch("src.camera.CameraReader",Camera), \
        patch("src.landmarks.make_holistic",lambda _:Holistic()), \
        patch("cv2.imshow"), patch("cv2.destroyAllWindows"), patch("cv2.waitKey",wait_key), \
        patch("urllib.request.urlopen",side_effect=urlopen), \
        patch.dict(os.environ,{"SIGN_SERVER_URL":"http://127.0.0.1:8765","SIGN_SERVER_TOKEN":""}), \
        contextlib.redirect_stdout(output):
    main.main()
assert camera_state["calls"]==1, output.getvalue()
assert "[cloud 미리보기] 안녕하세요 (words)" in output.getvalue(), output.getvalue()
assert "[cloud 번역]" not in output.getvalue()
# Path B: --recognize device runs the exported ONNX here and posts only glosses; --stt picks the
# speech engine. Needs torch/onnx to export a model; the rest of this test does not.
try:
    import onnx  # noqa: F401
    import torch
except ImportError:
    torch = None
if torch is None:
    print("torch/onnx 가 없어 기기 인식 경로(B) 통합 확인은 건너뜁니다")
else:
    import json
    import tempfile
    from pathlib import Path
    from src.continuous_sign import CtcRecognizer, save_checkpoint
    from src.sign_server import DeviceModel
    from tools.export_ctc_onnx import export
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        model = CtcRecognizer(2, hidden=16)
        with torch.no_grad():                    # always recognizes "G0"
            model.output.weight.zero_()
            model.output.bias.copy_(torch.tensor([0., 12.]))
        save_checkpoint(td / "ctc.pt", model, ["<blank>", "G0"])
        export(td / "ctc.pt", td / "ctc.onnx")
        api = TestClient(create_app(TranslationService(None, MockTranslator(), {"G0": "안녕하세요"},
                                                       mock=True,
                                                       device_model=DeviceModel(td / "ctc.json")),
                                    token=""))
        sent, engines = [], []

        def urlopen_device(request, timeout):
            path = request.full_url.split("8765")[1]
            sent.append((path, request.headers.get("Content-encoding"), json.loads(request.data)))
            r = api.post(path, content=request.data, headers=dict(request.header_items()))
            assert r.status_code == 200, r.text
            return io.BytesIO(r.content)

        class Stt:
            def __init__(self, out, gate=None, backend=None, caption_events=False):
                assert caption_events, "Main uses utterance-aware caption events"
                engines.append(backend)

            def start(self):
                pass

            def stop(self):
                pass

            def join(self, timeout=None):
                pass

        camera_state.update(phase=0, calls=0)
        output = io.StringIO()
        with patch.object(sys, "argv", ["main.py", "--sign-backend", "cloud", "--display", "null",
                                        "--no-tts", "--fixed-caption", "--stt", "vosk",
                                        "--recognize", "device", "--onnx", str(td / "ctc.onnx")]), \
                patch("src.camera.CameraReader", Camera), \
                patch("src.landmarks.make_holistic", lambda _: Holistic()), \
                patch("src.stt_worker.SttWorker", Stt), \
                patch("cv2.imshow"), patch("cv2.destroyAllWindows"), patch("cv2.waitKey", wait_key), \
                patch("urllib.request.urlopen", side_effect=urlopen_device), \
                patch.dict(os.environ, {"SIGN_SERVER_URL": "http://127.0.0.1:8765",
                                        "SIGN_SERVER_TOKEN": ""}), \
                contextlib.redirect_stdout(output):
            main.main()
        assert engines == ["vosk"], engines
        assert len(sent) == 1, output.getvalue()
        path, encoding, body = sent[0]
        assert path == "/v1/sign/compose" and encoding is None, "glosses travel small and plain"
        assert body["glosses"] == ["G0"] and "coordinates" not in body
        assert body["recognizer"] == DeviceModel(td / "ctc.json").metadata["model_id"]
        assert "[cloud 번역] 안녕하세요." in output.getvalue(), output.getvalue()
        # Wrong combinations stop before any camera opens.
        for argv in (["--recognize", "device"],
                     ["--recognize", "device", "--onnx", str(td / "ctc.onnx"), "--pose-3d"]):
            with patch.object(sys, "argv", ["main.py", "--sign-backend", "cloud", "--display",
                                            "null", "--no-stt", "--no-tts", *argv]), \
                    contextlib.redirect_stderr(io.StringIO()):
                try:
                    main.main()
                    raise AssertionError(f"accepted {argv}")
                except SystemExit as error:
                    assert error.code == 2, argv
print("[OK] integrated cloud main capture/API/preview, auto segmentation with 3D across a clock "
      "correction, fallback preview, device recognition (B) with --stt, cleanup")
