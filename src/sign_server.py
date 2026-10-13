"""Local/cloud API: pose CTC recognition followed by sentence composition.

Run: python -m src.sign_server --mock
Real: python -m src.sign_server --checkpoint models/continuous/ctc.pt --mapping glossary.json

POST /v1/sign/translate takes pose coordinates and recognizes on the server (path A).
POST /v1/sign/compose takes glosses a device recognized itself (path B); start the
server with --device-model ctc.json (the ONNX sidecar) so it knows that model's
labels, calibration and identity.

--mapping takes glossary.json (meanings, kinds, reviewed sentences) or a plain
{gloss: meaning} mapping.json. See src/translation.py for the composition order.
"""
import argparse
from collections import OrderedDict
import hashlib
import json
import logging
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Annotated, Literal
import zlib

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import (BaseModel, ConfigDict, Field, PrivateAttr, StrictBool, StrictInt,
                      ValidationError, model_validator)
import numpy as np
from src.ctc_runtime import GLOSS_VERSION, ONNX_FORMAT, check_calibration, check_contract
from src.pose_sequence import PoseSequence, VERSION, MAX_FRAMES
from src.translation import (DEFAULT_MODEL, Glossary, MockTranslator,  # noqa: F401
                             VllmTranslator, compose)
# A full-length 3D segment serializes to about 5 MB of JSON. The limit applies to
# the received bytes and again to a gzip body after decompression.
MAX_BODY_BYTES = 8 * 1024 * 1024


class _TooLarge(ValueError):
    pass


def _gunzip(data, limit):
    """Decompress one gzip member without ever producing more than limit bytes."""
    stream = zlib.decompressobj(16 + zlib.MAX_WBITS)
    out = stream.decompress(data, limit + 1)
    if len(out) > limit or stream.unconsumed_tail:
        raise _TooLarge("Decompressed request too large")
    if not stream.eof or stream.unused_data:
        raise zlib.error("Truncated or trailing gzip data")
    return out


class TranslationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    version: str
    session_id: str = Field(min_length=1, max_length=128, pattern=r"^[\w.-]+$")
    segment_id: str = Field(min_length=1, max_length=128, pattern=r"^[\w.-]+$")
    timestamps: list[float] = Field(min_length=2, max_length=MAX_FRAMES)
    coordinates: list[list[list[float]]] = Field(min_length=2, max_length=MAX_FRAMES)
    mask: list[list[StrictBool]] = Field(min_length=2, max_length=MAX_FRAMES)
    image_size: tuple[StrictInt, StrictInt]
    world: list[list[list[float]]] | None = Field(default=None, min_length=2,
                                                  max_length=MAX_FRAMES)
    world_mask: list[list[StrictBool]] | None = Field(default=None, min_length=2,
                                                      max_length=MAX_FRAMES)
    _sequence: PoseSequence = PrivateAttr(default=None)

    @model_validator(mode="after")
    def validate_pose(self):
        sequence = PoseSequence(self.timestamps, self.coordinates, self.mask,
                                self.image_size, self.version, self.world,
                                self.world_mask).validate()
        if sequence.version != self.version:
            raise ValueError("version does not match the presence of world coordinates")
        self._sequence = sequence
        return self

    def sequence(self):
        """The validated arrays, built once while the request was validated."""
        return self._sequence

    def fingerprint(self):
        """Identity of what the server would process, for duplicate detection.

        Hashes the validated arrays instead of re-serializing the JSON: whitespace,
        key order, gzip and 1 vs 1.0 do not change it, while any processed value does.
        Masked-out points are zeroed and coordinates are float32, as in recognition.
        """
        s = self.sequence()
        digest = hashlib.sha256(json.dumps([self.version, self.session_id, self.segment_id,
                                            list(s.image_size)]).encode())
        for array in (s.timestamps, s.coordinates, s.mask, s.world, s.world_mask):
            if array is None:
                digest.update(b"none")
            else:
                digest.update(f"{array.dtype}{array.shape}".encode())
                digest.update(np.ascontiguousarray(array).tobytes())
        return digest.hexdigest()


_ID = {"min_length": 1, "max_length": 128, "pattern": r"^[\w.-]+$"}


class GlossRequest(BaseModel):
    """Path B (REVIEW_AND_PLAN 3.3): the device ran CTC itself and sends only glosses."""
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    version: Literal[GLOSS_VERSION]
    session_id: str = Field(**_ID)
    segment_id: str = Field(**_ID)
    glosses: list[Annotated[str, Field(min_length=1, max_length=64)]] = Field(max_length=64)
    score: float = Field(ge=0, le=1)
    recognizer: str = Field(**_ID)    # model_id of the device model

    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.model_dump(), ensure_ascii=False,
                                         sort_keys=True).encode()).hexdigest()


class DeviceModel:
    """Labels, calibration and identity of the recognizer that runs on devices.

    Read from the ONNX sidecar of tools/export_ctc_onnx.py, so glosses from the
    device pass the same threshold and speech gate as server recognition.
    """

    def __init__(self, path):
        spec = json.loads(Path(path).with_suffix(".json").read_text(encoding="utf-8"))
        if spec.get("format") != ONNX_FORMAT:
            raise ValueError("Not a ctc-onnx-v1 sidecar")
        check_contract(spec["feature_set"], spec["version"], spec["feature_dim"], spec["labels"])
        self.labels = spec["labels"]
        self.calibration = spec.get("calibration") or {}
        check_calibration(self.calibration)
        self.metadata = spec.get("metadata") or {}
        if not self.metadata.get("model_id"):
            raise ValueError("Device model sidecar has no model_id")


class MockRecognizer:
    labels = ["<blank>", "안녕하세요"]
    calibration = {}
    metadata = {"mock": True}

    def predict(self, sequence):
        return ["안녕하세요"], 0.99


class MockDeviceModel:
    labels = MockRecognizer.labels
    calibration = {}
    metadata = {"mock": True, "model_id": "mock"}


class TranslationService:
    def __init__(self, recognizer=None, translator=None, mapping=None, *, mock=False,
                 ttl=300, cache_size=1024, max_waiters=8, clock=time.monotonic,
                 device_model=None):
        self.recognizer = recognizer
        self.device_model = device_model
        self.translator = translator
        self.glossary = Glossary.from_mapping(mapping)
        self.mock = mock
        self.clock, self.ttl, self.cache_size = clock, ttl, cache_size
        self.lock = threading.Lock()
        self.cache = OrderedDict()
        self.inflight = {}
        self.sessions = set()
        # Duplicate retries block a worker thread while waiting; bound them too.
        self.max_waiters, self.waiting = max_waiters, 0

    @property
    def mapping(self):
        return self.glossary.meanings

    @mapping.setter
    def mapping(self, value):
        self.glossary = Glossary.from_mapping(value)

    def _speech_allowed(self, recognizer):
        if self.mock or recognizer is None or self.translator is None:
            return False
        policy = recognizer.calibration
        return bool(policy.get("enabled") is True
                    and policy.get("mapping_hash") == self.glossary.policy_hash()
                    and policy.get("language_model") == self.translator.model)

    @property
    def speech_enabled(self):
        return self._speech_allowed(self.recognizer)

    @property
    def device_speech_enabled(self):
        return self._speech_allowed(self.device_model)

    def data_sources(self):
        """Attribution of the data the loaded models were trained on (AI Hub terms)."""
        from src.data_sources import attribution
        sources = [source for model in (self.recognizer, self.device_model)
                   for source in ((getattr(model, "metadata", None) or {}).get("data_sources")
                                  or [])]
        return attribution(sources)

    def check_device(self, request):
        """Refuse glosses from a model other than the one calibrated here."""
        if self.device_model is None:
            return  # answered "unavailable", like a server without a recognizer
        if request.recognizer != self.device_model.metadata["model_id"]:
            raise HTTPException(409, "Recognizer differs from the server's device model")
        if set(request.glosses) - set(self.device_model.labels[1:]):
            raise HTTPException(422, "Glosses outside the device model vocabulary")

    def process(self, request, received_at=None):
        received_at = self.clock() if received_at is None else received_at
        fingerprint = request.fingerprint()
        key = (request.session_id, request.segment_id)
        owner = False
        with self.lock:
            now = self.clock()
            for old in list(self.cache):
                if now - self.cache[old][0] >= self.ttl:
                    del self.cache[old]
            if key in self.cache:
                _, digest, result = self.cache[key]
                if digest != fingerprint:
                    raise HTTPException(409, "Segment ID reused with different data")
                return result.copy()
            if key in self.inflight:
                digest, event = self.inflight[key]
                if digest != fingerprint:
                    raise HTTPException(409, "Segment ID reused with different data")
                if self.waiting >= self.max_waiters:
                    raise HTTPException(429, "Too many duplicate requests; retry later")
                self.waiting += 1
            else:
                if request.session_id in self.sessions or len(self.inflight) >= 8:
                    raise HTTPException(429, "A request is already processing; retry later")
                event = threading.Event()
                self.inflight[key] = (fingerprint, event)
                self.sessions.add(request.session_id)
                owner = True
        if not owner:
            try:
                finished = event.wait(30)
            finally:
                with self.lock:
                    self.waiting -= 1
            if not finished:
                raise HTTPException(503, "Original request still processing")
            with self.lock:
                if key in self.cache:
                    return self.cache[key][2].copy()
            raise HTTPException(503, "Original request failed")
        try:
            result = self._translate(request, received_at)
            with self.lock:
                # Terminal results are cached, fallback included. Unavailable may recover.
                if result["status"] != "unavailable":
                    self.cache[key] = (self.clock(), fingerprint, result.copy())
                    while len(self.cache) > self.cache_size:
                        self.cache.popitem(last=False)
            return result
        finally:
            with self.lock:
                self.inflight.pop(key, None)
                self.sessions.discard(request.session_id)
                event.set()

    def _translate(self, request, received_at=None):
        start = self.clock()
        device = isinstance(request, GlossRequest)
        recognizer = self.device_model if device else self.recognizer
        result = {"version": request.version, "session_id": request.session_id,
                  "segment_id": request.segment_id, "status": "unavailable",
                  "glosses": [], "text": "", "source": "", "issues": [], "auto_speech": False,
                  "mock": self.mock, "model_version": {
                      "recognizer": (getattr(recognizer, "metadata", {}) or {}).get(
                          "model_id", "mock" if self.mock else "unavailable"),
                      "language": getattr(self.translator, "model", None)},
                  "timings_ms": {"queue_wait": max(0.0, (
                      start - (received_at if received_at is not None else start)) * 1000)}}
        if not device and not request.sequence().mask[:, :42].any():
            result["status"] = "no_sign"
            return result
        if recognizer is None or self.translator is None:
            return result
        try:
            if device:
                glosses, score = list(request.glosses), request.score
            else:
                begin = self.clock()
                glosses, score = self.recognizer.predict(request.sequence())
                result["timings_ms"]["recognize"] = (self.clock() - begin) * 1000
            self._decide(result, glosses, score, recognizer)
        except Exception as error:
            # Do not return upstream URLs, keys or exception messages to clients.
            logging.getLogger(__name__).warning("Translation failed (%s)", type(error).__name__)
            result.update(status="unavailable", text="", source="", issues=[], auto_speech=False)
        result["timings_ms"]["server_total"] = (self.clock() - start) * 1000
        return result

    def _decide(self, result, glosses, score, recognizer):
        """Shared by both paths: threshold, vocabulary, composition and the speech gate."""
        result["glosses"] = glosses
        threshold = recognizer.calibration.get("threshold")
        if not glosses:
            result["status"] = "no_sign"
        elif self.glossary.unknown(glosses) or (threshold is not None and score < threshold):
            result["status"] = "uncertain"
        else:
            begin = self.clock()
            composition = compose(glosses, self.glossary, self.translator)
            result["timings_ms"]["language"] = (self.clock() - begin) * 1000
            result.update(status=composition.status, text=composition.text,
                          source=composition.source, issues=list(composition.issues))
            if composition.issues:
                logging.getLogger(__name__).warning("Sentence fallback (%s)",
                                                    ", ".join(composition.issues))
            # Reviewed sentences get no exemption: the recognizer calibration still gates
            # speech. Fallback previews are never spoken automatically.
            result["auto_speech"] = (composition.status == "ok"
                                     and self._speech_allowed(recognizer))


def create_app(service=None, token=None):
    service = service or TranslationService()
    token = os.environ.get("SIGN_SERVER_TOKEN", "") if token is None else token
    app = FastAPI(title="signglass continuous translation")
    app.state.service = service

    @app.middleware("http")
    async def access(request: Request, call_next):
        request.state.received_at = service.clock()
        if token and not secrets.compare_digest(
                request.headers.get("authorization", ""), "Bearer " + token):
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
        if request.method == "POST":
            chunks, total = [], 0
            async for chunk in request.stream():
                total += len(chunk)
                if total > MAX_BODY_BYTES:
                    return JSONResponse({"detail": "Request too large"}, status_code=413)
                chunks.append(chunk)
            body = b"".join(chunks)
            encoding = request.headers.get("content-encoding", "identity").strip().lower()
            if encoding == "gzip":
                try:
                    body = _gunzip(body, MAX_BODY_BYTES)
                except _TooLarge:
                    return JSONResponse({"detail": "Request too large"}, status_code=413)
                except zlib.error:
                    return JSONResponse({"detail": "Malformed gzip body"}, status_code=400)
            elif encoding not in ("", "identity"):
                return JSONResponse({"detail": "Unsupported content encoding"}, status_code=415)
            request._body = body
        return await call_next(request)

    @app.get("/health")
    def health():
        return {"version": VERSION, "mock": service.mock,
                "recognizer_ready": service.recognizer is not None,
                "device_model_ready": service.device_model is not None,
                "language_configured": service.translator is not None,
                "auto_speech": service.speech_enabled,
                "device_auto_speech": service.device_speech_enabled,
                "data_sources": service.data_sources()}

    def parse(model, body):
        # pydantic's own JSON parser is about twice as fast as json.loads + validation
        # on these nested coordinate lists, and the arrays are built only once.
        try:
            return model.model_validate_json(body)
        except ValidationError as error:
            # Where and why only. The offending input can be the whole multi-megabyte
            # request, or a NaN that a JSON response cannot carry; a bad mask alone can
            # yield 44,100 errors, so a few are enough to locate the problem.
            errors = error.errors(include_url=False)
            detail = [{k: e[k] for k in ("type", "loc", "msg")} for e in errors[:10]]
            if len(errors) > 10:
                detail.append({"type": "more_errors", "loc": [],
                               "msg": f"{len(errors) - 10} more errors"})
            raise RequestValidationError(detail) from None

    def body_schema(model):
        return {"requestBody": {"required": True, "content": {
            "application/json": {"schema": model.model_json_schema()}}}}

    @app.post("/v1/sign/translate", openapi_extra=body_schema(TranslationRequest))
    async def translate(http_request: Request):
        request = parse(TranslationRequest, await http_request.body())
        # Recognition and the language model block; keep them off the event loop.
        return await run_in_threadpool(service.process, request,
                                       http_request.state.received_at)

    @app.post("/v1/sign/compose", openapi_extra=body_schema(GlossRequest))
    async def compose_glosses(http_request: Request):
        request = parse(GlossRequest, await http_request.body())
        service.check_device(request)
        return await run_in_threadpool(service.process, request,
                                       http_request.state.received_at)

    return app


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mock", action="store_true")
    p.add_argument("--checkpoint")
    p.add_argument("--mapping", help="glossary.json or a JSON object mapping gloss ID to meaning")
    p.add_argument("--device-model", help="ONNX sidecar (ctc.json) of the model devices run, "
                                          "for /v1/sign/compose")
    p.add_argument("--mock-language", action="store_true",
                   help="real recognizers with the mock composer: latency measurement only, "
                        "results are marked mock and never spoken")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()
    if args.host not in ("127.0.0.1", "localhost", "::1") and not os.environ.get("SIGN_SERVER_TOKEN"):
        p.error("SIGN_SERVER_TOKEN is required when binding a non-loopback address")
    if args.mock:
        service = TranslationService(MockRecognizer(), MockTranslator(),
                                     {"안녕하세요": "안녕하세요"}, mock=True,
                                     device_model=MockDeviceModel())
    else:
        if not (args.checkpoint or args.device_model) or not args.mapping:
            p.error("--mapping and --checkpoint and/or --device-model are required "
                    "outside mock mode")
        try:
            glossary = Glossary.load(args.mapping)
            device_model = DeviceModel(args.device_model) if args.device_model else None
        except (OSError, KeyError, ValueError) as error:
            p.error(str(error))
        recognizer = None
        if args.checkpoint:
            from src.continuous_sign import Recognizer
            recognizer = Recognizer(args.checkpoint)
        translator = MockTranslator() if args.mock_language else VllmTranslator()
        service = TranslationService(recognizer, translator, glossary, mock=args.mock_language,
                                     device_model=device_model)
    import uvicorn
    uvicorn.run(create_app(service), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
