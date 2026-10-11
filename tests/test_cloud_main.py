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
camera_state={"released":False,"closed":False,"phase":0,"started":None,"calls":0}
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
        hand=NS(landmark=[NS(x=.2+i*.01,y=.3+i*.01) for i in range(21)])
        return NS(left_hand_landmarks=hand,right_hand_landmarks=hand,
                  pose_landmarks=None,face_landmarks=None)
    def close(self):
        camera_state["closed"]=True
def wait_key(delay):
    phase=camera_state["phase"]
    if phase==0:
        camera_state["phase"]=1
        camera_state["started"]=time.monotonic()
        return ord(" ")
    elapsed=time.monotonic()-camera_state["started"]
    if phase==1 and elapsed>.15:
        camera_state["phase"]=2
        return ord(" ")
    if phase==2 and elapsed>.6:
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
print("[OK] integrated cloud main capture/API/preview, mock speech block and resource cleanup")
