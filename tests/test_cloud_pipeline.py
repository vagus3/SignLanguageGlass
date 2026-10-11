"""ASGI API + actual HTTP client serialization + vLLM protocol + worker failure flow."""
import _path  # noqa
import sys
try:
    from fastapi.testclient import TestClient
except ImportError:
    sys.exit(77)
from concurrent.futures import ThreadPoolExecutor
from fastapi import HTTPException
import hashlib
import io
import json
import threading
import time
from unittest.mock import patch
import urllib.error
import numpy as np
from src.pose_sequence import PoseSequence
from src.sign_server import (create_app, TranslationService, MockRecognizer, MockTranslator,
                             TranslationRequest, VllmTranslator)
from src.cloud_sign import CloudClient, CloudWorker, SegmentRecorder, validate_response

s=PoseSequence([0,.1,.2],np.full((3,49,2),.4),np.ones((3,49),bool),(640,480))
payload={**s.to_dict(),"session_id":"fixture","segment_id":"first"}
service=TranslationService(MockRecognizer(),MockTranslator(),{"안녕하세요":"안녕하세요"},mock=True)
client=TestClient(create_app(service,token="fixture-token"))
assert client.get("/health").status_code==401
headers={"Authorization":"Bearer fixture-token"}
assert client.get("/health",headers=headers).json()["mock"]

def asgi_urlopen(request, timeout):
    response=client.post("/v1/sign/translate",content=request.data,headers=dict(request.header_items()))
    if response.status_code>=400:
        raise urllib.error.HTTPError(request.full_url,response.status_code,"error",{},None)
    return io.BytesIO(response.content)

with patch("urllib.request.urlopen",side_effect=asgi_urlopen):
    outcome=CloudClient(token="fixture-token").translate(payload)
assert outcome["status"]=="ok" and outcome["mock"] and not outcome["auto_speech"]
assert client.post("/v1/sign/translate",json=payload,headers=headers).json()==outcome
bad={**payload,"timestamps":[0,0,.2]}
assert client.post("/v1/sign/translate",json=bad,headers=headers).status_code==422
changed={**payload,"coordinates":np.full((3,49,2),.5).tolist()}
assert client.post("/v1/sign/translate",json=changed,headers=headers).status_code==409
empty={**payload,"segment_id":"empty","mask":np.zeros((3,49),bool).tolist()}
assert client.post("/v1/sign/translate",json=empty,headers=headers).json()["status"]=="no_sign"
assert client.post("/v1/sign/translate",content=b" "* (4*1024*1024+1),headers=headers).status_code==413

count=[0]
def retry_then_success(request,timeout):
    count[0]+=1
    assert json.loads(request.data)["segment_id"]=="first"
    if count[0]==1:
        raise urllib.error.URLError("fixture disconnected")
    return io.BytesIO(json.dumps(outcome).encode())
with patch("urllib.request.urlopen",side_effect=retry_then_success):
    assert CloudClient().translate(payload)==outcome and count[0]==2
try:
    validate_response({**outcome,"segment_id":"other"},payload)
    raise AssertionError("Mismatched result accepted")
except ValueError:
    pass
try:
    validate_response({**outcome,"auto_speech":True},payload)
    raise AssertionError("Mock result enabled speech")
except ValueError:
    pass

lm=VllmTranslator(url="http://127.0.0.1:8000/v1",api_key="fixture-key")
def language_urlopen(request,timeout):
    body=json.loads(request.data)
    assert "structured_outputs" in body and body["temperature"]==0
    assert request.headers["Authorization"]=="Bearer fixture-key"
    output=json.dumps({"text":"안녕하세요.","source_indices":[0]},ensure_ascii=False)
    return io.BytesIO(json.dumps({"choices":[{"message":{"content":output}}]}).encode())
with patch("urllib.request.urlopen",side_effect=language_urlopen):
    assert lm.translate(["hello"],{"hello":"안녕하세요"})=="안녕하세요."
def invented_number(request,timeout):
    output=json.dumps({"text":"2명입니다.","source_indices":[0]},ensure_ascii=False)
    return io.BytesIO(json.dumps({"choices":[{"message":{"content":output}}]}).encode())
with patch("urllib.request.urlopen",side_effect=invented_number):
    try:
        lm.translate(["hello"],{"hello":"안녕하세요"})
        raise AssertionError("Invented number was accepted")
    except ValueError:
        pass
mapping={"안녕하세요":"안녕하세요"}
policy_recognizer=MockRecognizer()
policy_recognizer.calibration={
    "enabled":True,"mapping_hash":hashlib.sha256(json.dumps(
        mapping,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
    "language_model":lm.model}
policy_service=TranslationService(policy_recognizer,lm,mapping)
assert policy_service.speech_enabled
policy_service.mapping={"안녕하세요":"다른 의미"}
assert not policy_service.speech_enabled
policy_service.mapping=mapping
lm.model="changed-model"
assert not policy_service.speech_enabled
try:
    validate_response({**outcome,"timings_ms":None},payload)
    raise AssertionError("Malformed timing data passed")
except ValueError:
    pass

class CountingRecognizer(MockRecognizer):
    def __init__(self):
        self.count=0
    def predict(self,sequence):
        self.count+=1
        time.sleep(.05)
        return super().predict(sequence)
r=CountingRecognizer()
concurrent=TranslationService(r,MockTranslator(),{"안녕하세요":"안녕하세요"},mock=True)
req=TranslationRequest(**payload)
with ThreadPoolExecutor(max_workers=3) as pool:
    outcomes=list(pool.map(lambda _:concurrent.process(req),range(3)))
assert r.count==1 and outcomes[0]==outcomes[1]==outcomes[2]
class SlowRecognizer(MockRecognizer):
    def predict(self,sequence):
        time.sleep(.5)
        return super().predict(sequence)
bounded=TranslationService(SlowRecognizer(),MockTranslator(),{"안녕하세요":"안녕하세요"},
                           mock=True,max_waiters=1)
def attempt(_):
    try:
        bounded.process(req)
        return 200
    except HTTPException as e:
        return e.status_code
with ThreadPoolExecutor(max_workers=3) as pool:
    codes=sorted(pool.map(attempt,range(3)))
assert codes==[200,200,429], f"Duplicate waiters must be bounded: {codes}"
assert bounded.waiting==0
unmapped=TranslationService(MockRecognizer(),MockTranslator(),{})
assert unmapped.process(req)["status"]=="uncertain"

class SlowClient:
    timeout=.5
    def translate(self,body):
        time.sleep(.05)
        return {**outcome,"segment_id":body["segment_id"],"session_id":body["session_id"]}
worker=CloudWorker(SlowClient())
worker.start()
assert worker.submit(s,time.time()) and not worker.submit(s,time.time())
worker.cancel()
end=time.monotonic()+2
received=None
while received is None and time.monotonic()<end:
    received=worker.poll()
    time.sleep(.01)
assert received[0]["status"]=="expired" and not received[0]["auto_speech"]
assert not worker.busy
worker.stop()
worker.join(1)
assert not worker.is_alive()
class ExpiredClient(SlowClient):
    timeout=.01
worker=CloudWorker(ExpiredClient())
worker.start()
worker.submit(s,time.time())
time.sleep(.1)
assert worker.poll()[0]["status"]=="expired"
worker.stop()
worker.join(1)

recorder=SegmentRecorder(max_seconds=.01)
recorder.start()
time.sleep(.02)
recorder.add(s.coordinates[0],s.mask[0],0,(640,480))
assert recorder.error=="too_long" and not recorder.active
recorder.start()
recorder.add(s.coordinates[0],s.mask[0],0,(640,480))
recorder.add(s.coordinates[0],s.mask[0],.1,(1280,720))
assert recorder.error=="camera_discontinuity"
print("[OK] API/auth/schema, client retry, vLLM protocol, duplicate/cancel/expiry and overflow")
