#!/usr/bin/env python
"""Placement benchmark (REVIEW_AND_PLAN 3.3): server recognition (A) vs device recognition (B).

python tools/benchmark_paths.py data/processed/aihub_sample_extractors/mediapipe/manifest.json \
    --onnx models/continuous/ctc.onnx --url https://server.example --out reports/ab.json

Every clip goes through both paths against the same server and model:
  A  pose JSON (gzip) -> POST /v1/sign/translate: server CTC + sentence composition
  B  ONNX CTC on this machine -> glosses -> POST /v1/sign/compose: server composition
The server needs --checkpoint and --device-model from the same export. Weights do
not change latency, so a randomly initialized export measures placement before a
trained model exists. Run it on the demo laptop against the real server: the
numbers here are only as good as the machine and network they were taken on.

Before training, make a real-size, randomly initialized model for both sides:
python tools/benchmark_paths.py MANIFEST --make-dummy models/bench   (needs torch)
"""
import argparse
import json
import sys
import time
import uuid
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import clock
from src.cloud_sign import CloudClient, GlossClient
from src.pose_sequence import PoseSequence


def load_clips(manifest, limit=None):
    path = Path(manifest)
    rows = json.loads(path.read_text(encoding="utf-8"))["clips"][:limit]
    if not rows:
        raise ValueError("Manifest has no clips")
    return [(row["id"], PoseSequence.load(path.parent / row["file"])) for row in rows]


def make_dummy_model(manifest, out):
    """Real-size CTC model with random weights (and its ONNX + mapping) for latency only.

    The output layer is biased to one gloss so every request reaches sentence
    composition; recognition cost does not depend on the weights.
    """
    import torch
    from src.continuous_sign import CtcRecognizer, save_checkpoint
    from tools.export_ctc_onnx import export
    out = Path(out)
    if out.exists():
        raise ValueError(f"{out} exists; choose a new directory")
    rows = json.loads(Path(manifest).read_text(encoding="utf-8"))["clips"]
    labels = ["<blank>"] + sorted({g for r in rows for g in (r.get("glosses") or [])} or {"G0"})
    torch.manual_seed(0)
    model = CtcRecognizer(len(labels))
    with torch.no_grad():
        model.output.bias[1] += 30.0
    out.mkdir(parents=True)
    save_checkpoint(out / "ctc.pt", model, labels,
                    metadata={"purpose": "latency benchmark only, random weights"})
    export(out / "ctc.pt", out / "ctc.onnx")
    (out / "mapping.json").write_text(json.dumps({g: "안녕하세요" for g in labels[1:]},
                                                 ensure_ascii=False), encoding="utf-8")
    return out


def peak_rss_mb():
    try:
        import resource
    except ImportError:          # Windows: no resource module
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2 ** 20 if sys.platform == "darwin" else peak / 2 ** 10   # bytes vs KiB


def measure(client, clips, *, warmup=2):
    """One record per clip; the first `warmup` requests (cold sessions) are not kept."""
    session = f"bench-{uuid.uuid4().hex[:12]}"
    records, cpu = [], 0.0
    for i, (clip_id, sequence) in enumerate(clips[:warmup] + clips):
        segment = f"s{i}-{uuid.uuid4().hex[:8]}"
        cpu_start, t0 = time.process_time(), clock.now()
        payload = client.payload(sequence, session, segment)
        t1 = clock.now()
        try:
            result, error = client.translate(payload), None
        except Exception as e:  # noqa: BLE001 - the benchmark reports every failure
            result, error = {}, type(e).__name__
        t2 = clock.now()
        if i < warmup:
            continue
        cpu += time.process_time() - cpu_start
        body, _ = client.encode(payload)
        records.append({
            "clip": clip_id, "frames": len(sequence.timestamps),
            "status": result.get("status", "error"), "error": error,
            "glosses": result.get("glosses"),
            "prepare_ms": (t1 - t0) * 1000, "roundtrip_ms": (t2 - t1) * 1000,
            "total_ms": (t2 - t0) * 1000,
            "json_bytes": len(json.dumps(payload, allow_nan=False, separators=(",", ":"))),
            "sent_bytes": len(body), "server_ms": result.get("timings_ms", {})})
    return records, cpu


def _stats(values):
    a = np.asarray([v for v in values if v is not None], float)
    if not a.size:
        return None
    return {"p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
            "max": float(a.max()), "mean": float(a.mean())}


def summarize(records, cpu):
    statuses = {}
    for r in records:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    return {"segments": len(records), "statuses": statuses,
            "prepare_ms": _stats(r["prepare_ms"] for r in records),
            "roundtrip_ms": _stats(r["roundtrip_ms"] for r in records),
            "total_ms": _stats(r["total_ms"] for r in records),
            "server_total_ms": _stats(r["server_ms"].get("server_total") for r in records),
            "json_bytes": _stats(r["json_bytes"] for r in records),
            "sent_bytes": _stats(r["sent_bytes"] for r in records),
            "client_cpu_ms_per_segment": cpu * 1000 / max(len(records), 1)}


def benchmark(clips, clients, *, warmup=2):
    report = {"clips": len(clips), "paths": {}, "records": {}}
    for name, client in clients.items():
        records, cpu = measure(client, clips, warmup=warmup)
        report["paths"][name] = summarize(records, cpu)
        report["records"][name] = records
    if {"A", "B"} <= set(clients):
        pairs = list(zip(report["records"]["A"], report["records"]["B"]))
        same = sum(a["glosses"] == b["glosses"] for a, b in pairs if a["error"] is None
                   and b["error"] is None)
        report["gloss_agreement"] = {"same": same, "compared": len(pairs)}
    report["peak_rss_mb"] = peak_rss_mb()
    return report


def table(report):
    lines = [f"{'path':<4} {'n':>4} {'prepare p50/p95':>17} {'roundtrip p50/p95':>19} "
             f"{'total p50/p95':>15} {'sent KB p50':>11}  statuses"]
    for name, s in report["paths"].items():
        def pair(key):
            v = s[key]
            return f"{v['p50']:.1f}/{v['p95']:.1f}" if v else "-"
        sent = s["sent_bytes"]["p50"] / 1024 if s["sent_bytes"] else 0
        lines.append(f"{name:<4} {s['segments']:>4} {pair('prepare_ms'):>17} "
                     f"{pair('roundtrip_ms'):>19} {pair('total_ms'):>15} {sent:>11.1f}  "
                     f"{s['statuses']}")
    if "gloss_agreement" in report:
        g = report["gloss_agreement"]
        lines.append(f"A/B glosses identical on {g['same']}/{g['compared']} clips")
    if report.get("peak_rss_mb"):
        lines.append(f"peak RSS {report['peak_rss_mb']:.0f} MB (whole process)")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("manifest")
    p.add_argument("--make-dummy", metavar="DIR",
                   help="only write a random-weight ctc.pt/ctc.onnx/mapping.json for measuring")
    p.add_argument("--onnx", help="device model for path B (tools/export_ctc_onnx.py)")
    p.add_argument("--url", help="server URL (default: SIGN_SERVER_URL)")
    p.add_argument("--paths", nargs="+", choices=["A", "B"], default=["A", "B"])
    p.add_argument("--limit", type=int)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--out", help="JSON report with every record")
    args = p.parse_args()
    if args.make_dummy:
        try:
            out = make_dummy_model(args.manifest, args.make_dummy)
        except (OSError, ValueError) as error:
            p.error(str(error))
        print(f"{out}/ctc.pt, ctc.onnx, mapping.json written. Server:\n"
              f"  python -m src.sign_server --checkpoint {out}/ctc.pt --device-model "
              f"{out}/ctc.json --mapping {out}/mapping.json --mock-language\n"
              f"Then: python tools/benchmark_paths.py {args.manifest} --onnx {out}/ctc.onnx "
              "--url <server>")
        return
    if "B" in args.paths and not args.onnx:
        p.error("path B needs --onnx")
    if args.out and Path(args.out).exists():
        p.error("Output exists; choose a new path")
    try:
        clips = load_clips(args.manifest, args.limit)
        clients = {}
        if "A" in args.paths:
            clients["A"] = CloudClient(args.url)
        if "B" in args.paths:
            from src.ctc_runtime import OnnxRecognizer
            clients["B"] = GlossClient(OnnxRecognizer(args.onnx), args.url)
    except (OSError, ValueError) as error:
        p.error(str(error))
    report = benchmark(clips, clients, warmup=args.warmup)
    print(table(report))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                  encoding="utf-8")


if __name__ == "__main__":
    main()
