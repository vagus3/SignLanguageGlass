"""ASGI API + actual HTTP client serialization + vLLM protocol + worker failure flow."""
import _path  # noqa
import sys
try:
    from fastapi.testclient import TestClient
except ImportError:
    sys.exit(77)
from concurrent.futures import ThreadPoolExecutor
from fastapi import HTTPException
import gzip
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
                             TranslationRequest, VllmTranslator, MAX_BODY_BYTES)
from src.cloud_sign import CloudClient, CloudWorker, SegmentRecorder, validate_response
from src import clock
from src.translation import Glossary

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
# Duplicate detection compares processed content, not bytes: key order, spacing and
# 0 vs 0.0 are the same request; so are values under a false mask.
jsonh={**headers,"Content-Type":"application/json"}
reordered=json.dumps(dict(reversed(list({**payload,"timestamps":[0,.1,.2]}.items()))),indent=1)
assert reordered!=json.dumps(payload)
assert client.post("/v1/sign/translate",content=reordered,headers=jsonh).json()==outcome
masked={**payload,"segment_id":"masked","mask":[[j!=48 for j in range(49)]]*3}
hidden=json.loads(json.dumps(masked))
hidden["coordinates"][1][48]=[.9,.9]
assert client.post("/v1/sign/translate",json=masked,headers=headers).status_code==200
assert client.post("/v1/sign/translate",json=hidden,headers=headers).status_code==200
assert (TranslationRequest.model_validate(masked).fingerprint()
        ==TranslationRequest.model_validate(hidden).fingerprint())
visible=json.loads(json.dumps(masked))
visible["coordinates"][1][0]=[.9,.9]
assert client.post("/v1/sign/translate",json=visible,headers=headers).status_code==409
nan_body=json.dumps(payload).replace("0.4","NaN",1)
assert '"coordinates":[[[NaN,' in nan_body.replace(" ",""), "a real NaN token, not broken JSON"
for invalid in (nan_body,"{not json",json.dumps({**payload,"mask":[[1]*49]*3}),
                json.dumps({**payload,"timestamps":[0,.2,.1]})):
    response=client.post("/v1/sign/translate",content=invalid,headers=jsonh)
    assert response.status_code==422 and response.json()["detail"], invalid[:30]
    # Errors say where and why, never echo the (possibly megabyte) input back.
    assert all(set(e)=={"type","loc","msg"} for e in response.json()["detail"])
    assert len(response.content)<2000
assert "TranslationRequest" in json.dumps(client.get("/openapi.json",headers=headers).json())
empty={**payload,"segment_id":"empty","mask":np.zeros((3,49),bool).tolist()}
assert client.post("/v1/sign/translate",json=empty,headers=headers).json()["status"]=="no_sign"
assert client.post("/v1/sign/translate",content=b" "*(MAX_BODY_BYTES+1),headers=headers).status_code==413
# gzip bodies: decoded with the same size limit; plain JSON keeps working.
plain=json.dumps({**payload,"segment_id":"gzip"}).encode()
packed=gzip.compress(plain,mtime=0)
gz={**headers,"Content-Type":"application/json","Content-Encoding":"gzip"}
first=client.post("/v1/sign/translate",content=packed,headers=gz)
assert first.status_code==200 and len(packed)<len(plain)
again=client.post("/v1/sign/translate",content=plain,
                  headers={**headers,"Content-Type":"application/json"})
assert again.json()==first.json(),"Plain and gzip bodies of one segment are the same request"
for broken in (b"\x1f\x8bnot gzip",packed[:-4],packed+b"trailing"):
    assert client.post("/v1/sign/translate",content=broken,headers=gz).status_code==400
bomb=gzip.compress(b" "*(MAX_BODY_BYTES+1),mtime=0)
assert len(bomb)<MAX_BODY_BYTES//100
assert client.post("/v1/sign/translate",content=bomb,headers=gz).status_code==413
assert client.post("/v1/sign/translate",content=plain,
                   headers={**headers,"Content-Encoding":"br"}).status_code==415

count=[0]
sent=[]
def retry_then_success(request,timeout):
    count[0]+=1
    sent.append(request.data)
    assert request.get_header("Content-encoding")=="gzip"
    assert json.loads(gzip.decompress(request.data))["segment_id"]=="first"
    if count[0]==1:
        raise urllib.error.URLError("fixture disconnected")
    return io.BytesIO(json.dumps(outcome).encode())
with patch("urllib.request.urlopen",side_effect=retry_then_success):
    assert CloudClient().translate(payload)==outcome and count[0]==2
assert sent[0]==sent[1],"A retry resends the identical compressed bytes"
def plain_only(request,timeout):
    assert request.get_header("Content-encoding") is None
    return io.BytesIO(json.dumps(outcome).encode())
with patch("urllib.request.urlopen",side_effect=plain_only):
    assert CloudClient(compress=False).translate(payload)==outcome
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
assert worker.submit(s,clock.now()) and not worker.submit(s,clock.now())
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
from src.cloud_sign import describe
assert describe(urllib.error.HTTPError("u",409,"x",{},None))=="HTTP 409"
assert describe(TimeoutError())=="시간 초과" and describe(ValueError())=="ValueError"
assert describe(urllib.error.URLError("refused")).startswith("연결 실패")
worker=CloudWorker(ExpiredClient())
worker.start()
worker.submit(s,clock.now())
time.sleep(.1)
assert worker.poll()[0]["status"]=="expired"
worker.stop()
worker.join(1)

class Unreachable:
    model="fixture-model"
    def translate(self,glosses,mapping):
        raise TimeoutError("upstream timed out")
class TwoGlosses(MockRecognizer):
    def predict(self,sequence):
        return ["안녕하세요","병원"],0.99
glossary=Glossary({"안녕하세요":"안녕하세요","병원":"병원"},
                  sentences=[{"glosses":["안녕하세요"],"text":"안녕하세요!","reviewed_by":"fixture"}])
def ask(service,segment):
    return service.process(TranslationRequest(**{**payload,"segment_id":segment}))
reviewed=ask(TranslationService(MockRecognizer(),Unreachable(),glossary),"reviewed")
assert (reviewed["status"],reviewed["text"],reviewed["source"])==("ok","안녕하세요!","reviewed")
calibrated=TwoGlosses()
calibrated.calibration={"enabled":True,"mapping_hash":glossary.policy_hash(),
                        "language_model":Unreachable.model}
speaking=TranslationService(calibrated,Unreachable(),glossary)
assert speaking.speech_enabled
fallback=ask(speaking,"fallback")
assert (fallback["status"],fallback["source"],fallback["text"])==("fallback","words","안녕하세요 병원")
assert fallback["auto_speech"] is False,"A fallback preview is never spoken, even when calibrated"
assert fallback["issues"]==["generation:TimeoutError"]
assert "upstream" not in json.dumps(fallback),"Upstream error messages stay on the server"
assert ask(speaking,"fallback")==fallback,"Fallback is a cached terminal result"
fallback_payload={**payload,"segment_id":"fallback"}
assert validate_response(fallback,fallback_payload)
for unsafe in ({**fallback,"auto_speech":True},{**fallback,"text":" "},
               {**fallback,"source":"language_model"},{**fallback,"text":"가"*1001}):
    try:
        validate_response(unsafe,fallback_payload)
        raise AssertionError(f"Unsafe fallback accepted: {unsafe['source']}")
    except ValueError:
        pass

recorder=SegmentRecorder(max_seconds=.01)
recorder.start()
time.sleep(.02)
recorder.add(s.coordinates[0],s.mask[0],0,(640,480))
assert recorder.error=="too_long" and not recorder.active
recorder.start()
recorder.add(s.coordinates[0],s.mask[0],0,(640,480))
recorder.add(s.coordinates[0],s.mask[0],.1,(1280,720))
assert recorder.error=="camera_discontinuity"
print("[OK] API/auth/schema, client retry, vLLM protocol, reviewed sentences, preview-only "
      "fallback, duplicate/cancel/expiry and overflow")
