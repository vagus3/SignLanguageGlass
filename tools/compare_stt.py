#!/usr/bin/env python
"""Compare STT engines on the same recordings: CER, compute and caption timing.

python tools/compare_stt.py --tts-cache                      # synthetic TTS phrases
python tools/compare_stt.py --wavs data/raw/stt_samples/team --refs refs.txt \
    --engines vosk sherpa:models/sherpa-ko-2024-06-16 sherpa:models/sherpa-ko-174m-chunk32

refs: "file.wav<space>text" lines (sherpa test_wavs/trans.txt style) or a JSON
{"file.wav": "text"}. Audio is 16 kHz mono and is streamed in AUDIO_BLOCK chunks
exactly as SttWorker feeds the microphone, followed by trailing silence:
  cer          character errors / reference characters, spaces and punctuation removed
  rtf          decoding time / audio time (below 1 keeps up with live speech)
  first_ms     audio time until the first partial caption
  final_ms     audio time from the end of speech to the final caption, by the
               engine's own endpointing (None: only the end-of-input flush gave it)
Synthetic speech is not a human-speech accuracy result; decide with real recordings
from the demo microphone and room.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src.continuous_sign import edit_distance
from src.stt_worker import BACKENDS, open_decoder

TRAILING_SILENCE = 1.5
_NOISE = re.compile(r"[\s.,!?~…·'\"“”‘’()\[\]-]+")


def normalize(text):
    return _NOISE.sub("", text)


def cer(reference, hypothesis):
    ref, hyp = normalize(reference), normalize(hypothesis)
    return edit_distance(list(ref), list(hyp)), len(ref)


def load_refs(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        return dict(json.loads(text))
    refs = {}
    for line in text.splitlines():
        if line.strip():
            name, _, sentence = line.strip().partition(" ")
            refs[name] = sentence.strip()
    return refs


def tts_cache_refs():
    from src.build_tts_cache import cached_file, phrases
    return [(cached_file(text), text) for text in phrases() if cached_file(text)]


def read_pcm(path):
    data, rate = sf.read(path, dtype="int16", always_2d=True)
    if rate != C.AUDIO_RATE:
        raise ValueError(f"{path}: {rate} Hz; convert to {C.AUDIO_RATE} Hz mono first")
    return data[:, 0].tobytes()


def transcribe(decoder, pcm, rate=None):
    """Stream one recording; returns text, decode seconds and caption timings (audio ms)."""
    rate = rate or C.AUDIO_RATE
    block = C.AUDIO_BLOCK * 2
    speech_ms = len(pcm) / 2 / rate * 1000
    stream = pcm + bytes(int(rate * TRAILING_SILENCE) * 2)
    finals, first_ms, final_ms, busy = [], None, None, 0.0
    decoder.reset()
    for i in range(0, len(stream), block):
        start = time.perf_counter()
        events = decoder.feed(stream[i:i + block])
        busy += time.perf_counter() - start
        heard_ms = (i + len(stream[i:i + block])) / 2 / rate * 1000
        for kind, text in events:
            if first_ms is None:
                first_ms = heard_ms
            if kind == "final":
                finals.append(text)
                final_ms = heard_ms - speech_ms
    start = time.perf_counter()
    rest = [text for kind, text in decoder.flush() if kind == "final"]
    busy += time.perf_counter() - start
    return " ".join(finals + rest), busy, first_ms, (final_ms if finals else None)


def compare(items, engines):
    """items: [(path, reference)]; engines: [(label, backend, model_dir)]."""
    report = {"files": len(items), "engines": {}}
    audio = [(path, ref, read_pcm(path)) for path, ref in items]
    total_s = sum(len(pcm) / 2 / C.AUDIO_RATE for _, _, pcm in audio)
    for label, backend, model_dir in engines:
        decoder = open_decoder(backend, model_dir)
        rows, edits, chars, busy = [], 0, 0, 0.0
        for path, ref, pcm in audio:
            text, seconds, first_ms, final_ms = transcribe(decoder, pcm)
            e, n = cer(ref, text)
            edits, chars, busy = edits + e, chars + n, busy + seconds
            rows.append({"file": Path(path).name, "reference": ref, "recognized": text,
                         "cer": e / max(n, 1), "first_ms": first_ms, "final_ms": final_ms})
        finals = [r["final_ms"] for r in rows if r["final_ms"] is not None]
        firsts = [r["first_ms"] for r in rows if r["first_ms"] is not None]
        report["engines"][label] = {
            "cer": edits / max(chars, 1), "rtf": busy / max(total_s, 1e-9),
            "first_ms_p50": float(np.median(firsts)) if firsts else None,
            "final_ms_p50": float(np.median(finals)) if finals else None,
            "endpointed": len(finals), "rows": rows}
    return report


def table(report):
    lines = [f"{'engine':<34} {'CER':>6} {'RTF':>6} {'first p50':>10} {'final p50':>10} "
             f"{'endpointed':>10}"]
    for label, e in report["engines"].items():
        first = f"{e['first_ms_p50']:.0f}ms" if e["first_ms_p50"] is not None else "-"
        final = f"{e['final_ms_p50']:.0f}ms" if e["final_ms_p50"] is not None else "-"
        lines.append(f"{label:<34} {e['cer']:>6.1%} {e['rtf']:>6.3f} {first:>10} {final:>10} "
                     f"{e['endpointed']:>4}/{report['files']}")
    return "\n".join(lines)


def parse_engine(spec):
    backend, _, model_dir = spec.partition(":")
    if backend not in BACKENDS:
        raise argparse.ArgumentTypeError(f"engine must be {BACKENDS}[:model_dir]")
    return spec, backend, (model_dir or None)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--tts-cache", action="store_true", help="synthetic phrases (assets)")
    source.add_argument("--wavs", help="directory of 16 kHz mono recordings (needs --refs)")
    p.add_argument("--refs", help="trans.txt-style lines or JSON {file: text}")
    p.add_argument("--engines", nargs="+", type=parse_engine,
                   default=[parse_engine("vosk"), parse_engine("sherpa")])
    p.add_argument("--out", help="JSON report with every transcript")
    p.add_argument("-v", "--verbose", action="store_true", help="print every transcript")
    args = p.parse_args()
    if args.wavs and not args.refs:
        p.error("--wavs needs --refs")
    if args.tts_cache:
        items = tts_cache_refs()
    else:
        refs = load_refs(args.refs)
        items = [(Path(args.wavs) / name, text) for name, text in sorted(refs.items())]
        missing = [str(path) for path, _ in items if not path.exists()]
        if missing:
            p.error(f"missing recordings: {missing[:3]}")
    if not items:
        p.error("no recordings")
    try:
        report = compare(items, args.engines)
    except (OSError, ValueError) as error:
        p.error(str(error))
    report["synthetic"] = bool(args.tts_cache)
    print(table(report))
    if args.verbose:
        for label, e in report["engines"].items():
            for r in e["rows"]:
                if r["cer"]:
                    print(f"  [{label}] {r['reference']}  ->  {r['recognized']}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                  encoding="utf-8")


if __name__ == "__main__":
    main()
