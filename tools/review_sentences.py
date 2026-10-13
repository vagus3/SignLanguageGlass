#!/usr/bin/env python
"""Semantic review sheet: what the system would say for each gloss sequence, judged by people.

Files go under reports/ (ignored by Git): review sheets carry reviewer names.

1) python tools/review_sentences.py draft --out reports/glossary.json
       glossary from config.VOCAB (meaning = the Korean word; edit before review)
2) python tools/review_sentences.py export --glossary reports/glossary.json \
       --out reports/review_sheet.csv
       runs the server's composition (reviewed sentence, else the language model at
       VLLM_BASE_URL/VLLM_MODEL) for config.PHRASE_RULES patterns, the glossary's
       reviewed sentences and --sequences; opens in Excel (UTF-8 with BOM). It stops
       without a sheet when the language model is unreachable or serves another model
3) reviewers fill meaning_preserved (Y/N), corrected_text, reviewer, note; Excel may
   save as "CSV UTF-8" or plain "CSV" (CP949), both are read
4) python tools/review_sentences.py import reports/review_sheet.csv \
       --glossary reports/glossary.json --out-glossary reports/glossary.reviewed.json \
       --out-review reports/review.json

Import turns approved and corrected rows into reviewed sentences (translation
memory: exact glosses skip the language model) and writes gloss-level approvals
for calibration (tools/evaluate_continuous.py --semantic-review). Changing the
sentences changes Glossary.policy_hash, so speech calibration must be redone.
"""
import argparse
import csv
import datetime
import io
import json
import sys
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src.translation import Glossary, MockTranslator, compose

COLUMNS = ["id", "glosses", "meanings", "text", "source", "issues", "model", "rule_text",
           "meaning_preserved", "corrected_text", "reviewer", "reviewed_on", "note"]
YES = {"y", "yes", "o", "true", "1", "예", "네", "ㅇ"}
NO = {"n", "no", "x", "false", "0", "아니오", "아니요", "ㄴ"}


def draft_glossary(vocab=None):
    words = [w for w in (vocab or C.VOCAB) if w != "NEUTRAL"]
    return Glossary({w: w for w in words})


def default_sequences(glossary, extra=()):
    """PHRASE_RULES patterns, reviewed sentences and extra sequences, deduplicated in order."""
    seen, out = set(), []
    for glosses in ([list(p) for p, _ in C.PHRASE_RULES]
                    + [s["glosses"] for s in glossary.sentences.values()] + [list(e) for e in extra]):
        key = tuple(glosses)
        if key not in seen:
            seen.add(key)
            out.append(list(glosses))
    return out


def export_rows(glossary, sequences, translator):
    rules = {tuple(p): text for p, text in C.PHRASE_RULES}
    rows = []
    for i, glosses in enumerate(sequences, 1):
        unknown = glossary.unknown(glosses)
        composition = compose(glosses, glossary, translator)
        rows.append({
            "id": f"r{i:03d}", "glosses": json.dumps(glosses, ensure_ascii=False),
            "meanings": " / ".join(glossary.meanings.get(g, "?") for g in glosses),
            "text": composition.text, "source": composition.source or composition.status,
            "issues": ", ".join([f"unknown:{g}" for g in unknown] + list(composition.issues)),
            "model": getattr(translator, "model", ""), "rule_text": rules.get(tuple(glosses), ""),
            "meaning_preserved": "", "corrected_text": "", "reviewer": "", "reviewed_on": "",
            "note": ""})
    return rows


def check_language_model(translator):
    """Stop before writing a sheet when the language model cannot answer.

    Without this, every row falls back to the word list and reviewers get a sheet
    with nothing to approve.
    """
    try:
        served = translator.served_models()
    except urllib.error.HTTPError as error:
        raise ValueError(f"언어 모델 서버 {translator.url} 가 HTTP {error.code} 로 거절했습니다. "
                         "VLLM_API_KEY 를 확인하세요.") from None
    except (OSError, ValueError) as error:      # URLError, timeouts, malformed replies
        raise ValueError(f"언어 모델 서버 {translator.url} 에 연결할 수 없습니다 "
                         f"({type(error).__name__}). vLLM 을 켜고 VLLM_BASE_URL 을 확인하세요. "
                         "형식만 보려면 --mock 을 쓰세요.") from None
    if translator.model not in served:
        raise ValueError(f"서버에 {translator.model} 모델이 없습니다(제공: {', '.join(served) or '없음'}). "
                         "VLLM_MODEL 을 시연에 쓸 모델로 맞추세요.")


def generation_failures(rows):
    """Rows whose sentence could not be generated (connection, timeout, invalid output)."""
    return [r["id"] for r in rows if "generation:" in r["issues"]]


def write_sheet(rows, path):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def read_sheet(path):
    """Reads UTF-8 (with or without BOM) and CP949, the default of Korean Excel's 'CSV' save."""
    data = Path(path).read_bytes()
    for encoding in ("utf-8-sig", "cp949"):
        try:
            text = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"{path}: 글자 인코딩을 읽을 수 없습니다. 엑셀에서 "
                         "'CSV UTF-8(쉼표로 분리)' 형식으로 다시 저장하세요.")
    if encoding == "cp949":
        print(f"{path}: 엑셀 기본 형식(CP949)으로 저장된 파일을 읽었습니다.")
    rows = list(csv.DictReader(io.StringIO(text, newline="")))
    missing = set(COLUMNS) - set(rows[0] if rows else COLUMNS)
    if missing:
        raise ValueError(f"{path}: 검수표 열 이름이 맞지 않습니다(빠진 열: {sorted(missing)}). "
                         "내보낸 파일의 첫 줄을 고치지 말고, 엑셀에서 'CSV UTF-8' 또는 "
                         "'CSV(쉼표로 분리)'로 저장하세요.")
    return rows


def judgement(value):
    v = (value or "").strip().lower()
    if v in YES:
        return True
    if v in NO:
        return False
    if v:
        raise ValueError(f"meaning_preserved 칸은 Y 또는 N 으로 적어 주세요 (지금 값: {value!r})")
    return None


def apply_review(glossary, rows, *, reviewer=None, today=None):
    """New glossary, gloss-level review and counts. Raises on incomplete or unsafe rows."""
    today = today or datetime.date.today().isoformat()
    sentences = {tuple(s["glosses"]): dict(s) for s in glossary.sentences.values()}
    review, counts, names = [], {"approved": 0, "corrected": 0, "rejected": 0, "removed": 0,
                                 "skipped": 0}, set()
    for row in rows:
        glosses = json.loads(row["glosses"])
        key, ok = tuple(glosses), judgement(row["meaning_preserved"])
        corrected = (row["corrected_text"] or "").strip()
        if ok is None and not corrected:
            counts["skipped"] += 1
            continue
        name = (row["reviewer"] or reviewer or "").strip()
        if not name:
            raise ValueError(f"{row['id']}: reviewer(검수자) 칸이 비어 있습니다")
        if glossary.unknown(glosses):
            raise ValueError(f"{row['id']}: 단어 {glossary.unknown(glosses)} 가 용어집(glossary)에 "
                             "없습니다. 먼저 뜻을 추가하세요")
        names.add(name)
        when = (row["reviewed_on"] or today).strip()
        if corrected:
            text, counts["corrected"] = corrected, counts["corrected"] + 1
        elif ok:
            if row["model"] == MockTranslator.model and row["source"] == "language_model":
                raise ValueError(f"{row['id']}: 모의(mock) 결과는 승인할 수 없습니다. 실제 언어 "
                                 "모델로 다시 내보내세요")
            if not row["text"].strip() or row["source"] not in ("reviewed", "language_model"):
                raise ValueError(f"{row['id']}: 말할 문장이 없는 행(source {row['source']!r})은 "
                                 "승인할 수 없습니다. corrected_text 칸에 문장을 적어 주세요")
            text, counts["approved"] = row["text"].strip(), counts["approved"] + 1
        else:
            counts["rejected"] += 1
            review.append({"glosses": glosses, "text": row["text"], "meaning_preserved": False,
                           "reviewed_by": name, "reviewed_on": when, "note": row["note"]})
            if key in sentences:
                del sentences[key]
                counts["removed"] += 1
            continue
        sentences[key] = {"glosses": glosses, "text": text, "reviewed_by": name,
                          "reviewed_on": when, **({"note": row["note"]} if row["note"] else {})}
        review.append({"glosses": glosses, "text": text, "meaning_preserved": True,
                       "reviewed_by": name, "reviewed_on": when})
    updated = Glossary(glossary.meanings, kinds=glossary.kinds, aihub=glossary.aihub,
                       sentences=list(sentences.values()))
    return updated, {"reviewer": ", ".join(sorted(names)), "sequences": review}, counts


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("draft", help="glossary.json from config.VOCAB")
    d.add_argument("--out", required=True)
    e = sub.add_parser("export", help="review sheet with the system's sentences")
    e.add_argument("--glossary", required=True)
    e.add_argument("--sequences", help='JSON list of gloss lists, e.g. [["이름","무엇"]]')
    e.add_argument("--mock", action="store_true", help="mock language model (format check only)")
    e.add_argument("--out", required=True)
    i = sub.add_parser("import", help="apply a filled sheet")
    i.add_argument("sheet")
    i.add_argument("--glossary", required=True)
    i.add_argument("--reviewer", help="for rows that leave reviewer empty")
    i.add_argument("--out-glossary", required=True)
    i.add_argument("--out-review", required=True)
    args = p.parse_args()

    outputs = ([args.out] if args.command in ("draft", "export")
               else [args.out_glossary, args.out_review])
    if any(Path(o).exists() for o in outputs):
        p.error("Output exists; choose new paths")
    try:
        if args.command == "draft":
            glossary = draft_glossary()
            Path(args.out).write_text(json.dumps(glossary.to_dict(), ensure_ascii=False,
                                                 indent=2), encoding="utf-8")
            print(f"{args.out}: 단어 {len(glossary.meanings)}개. 단어마다 'ko'(뜻)를 확인하고, "
                  "숫자·부정·사람·장소 단어에는 kind(number/negation/person/place)를 넣으세요.")
        elif args.command == "export":
            glossary = Glossary.load(args.glossary)
            extra = json.loads(Path(args.sequences).read_text(encoding="utf-8")) if args.sequences else ()
            if isinstance(extra, dict):
                extra = extra.get("sequences", [])
            if args.mock:
                translator = MockTranslator()
            else:
                from src.translation import VllmTranslator
                translator = VllmTranslator()
                check_language_model(translator)
            rows = export_rows(glossary, default_sequences(glossary, extra), translator)
            failed = generation_failures(rows)
            if failed:
                raise ValueError(f"{len(failed)}개 행에서 문장 생성이 실패했습니다"
                                 f"({', '.join(failed[:5])}). 서버 상태를 확인하고 다시 내보내세요. "
                                 "검수표는 만들지 않았습니다.")
            write_sheet(rows, args.out)
            failed = sum(r["source"] not in ("reviewed", "language_model") for r in rows)
            print(f"{args.out}: {len(rows)}행 (말할 문장이 없는 행 {failed}개, issues 칸 확인). "
                  f"언어 모델: {translator.model}")
        else:
            glossary = Glossary.load(args.glossary)
            updated, review, counts = apply_review(glossary, read_sheet(args.sheet),
                                                   reviewer=args.reviewer)
            Path(args.out_glossary).write_text(json.dumps(updated.to_dict(), ensure_ascii=False,
                                                          indent=2), encoding="utf-8")
            Path(args.out_review).write_text(json.dumps(review, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
            print(f"승인 {counts['approved']}, 수정 {counts['corrected']}, 반려 {counts['rejected']}, "
                  f"삭제 {counts['removed']}, 빈 행 {counts['skipped']}. "
                  f"검수 문장 {len(updated.sentences)}개.")
            if updated.policy_hash() != glossary.policy_hash():
                print("검수 문장이 바뀌어 이전 자동 발화 보정과 맞지 않습니다. "
                      "새 용어집으로 보정을 다시 하세요.")
    except (OSError, ValueError) as error:
        p.error(str(error))


if __name__ == "__main__":
    main()
