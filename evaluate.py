"""Evaluate the RAG system on eval/eval_questions.json.

    python evaluate.py --retrieval-only    # embeddings + FAISS only, no LLM / API key needed
    python evaluate.py                     # full run: retrieval + generated answers
    python evaluate.py --judge             # full run + LLM-as-judge groundedness check
    python evaluate.py --ids Q01,Q14       # only some questions
    python evaluate.py --sleep 4           # pause between questions (free API tiers have rate limits)

What is measured
  Retrieval (automatic): is the evidence for the question inside the top-k chunks?
      Hit@k    1 if at least one gold evidence chunk is in the top k
      Recall@k share of the gold evidence items found in the top k (matters for multi-document questions)
      MRR      1 / rank of the first gold chunk (0 if not found)
  Answer correctness (automatic keyword check): every keyword group must appear in the answer;
      for questions that are NOT in the knowledge base the system must decline.
  Groundedness (automatic, rough): every number in the answer must appear in the context that was
      given to the LLM (or in the question). With --judge an LLM also judges the answer against the context.
  Citation accuracy (automatic): the passages the answer cites must contain the gold evidence.
The CSV has an empty `manual_correct` column so the answers can also be reviewed by hand.
"""
import argparse
import csv
import json
import re
import time
from collections import defaultdict

from src import config
from src.rag_pipeline import build_pipeline, build_retriever, format_context
from src.utils import norm_answer, norm_evidence

K_VALUES = (1, 3, config.TOP_K)


# ---- retrieval ------------------------------------------------------------------
def satisfies(chunk, gold: dict) -> bool:
    """A chunk satisfies a gold item if it is from an accepted document and holds all evidence strings."""
    text = norm_evidence(chunk.text)
    return chunk.file_name in gold["docs"] and all(norm_evidence(e) in text for e in gold["evidence"])


def retrieval_metrics(hits, gold: list[dict]) -> dict:
    ranks = []   # best rank at which each gold item was found (None = not found)
    for item in gold:
        ranks.append(next((h.rank for h in hits if satisfies(h.chunk, item)), None))
    found = [r for r in ranks if r]
    metrics = {f"hit@{k}": int(any(r <= k for r in found)) for k in K_VALUES}
    metrics.update({f"recall@{k}": sum(r <= k for r in found) / len(gold) for k in K_VALUES})
    metrics["mrr"] = 1 / min(found) if found else 0.0
    metrics["gold_ranks"] = ranks
    return metrics


# ---- answers ---------------------------------------------------------------------
def keywords_present(answer: str, groups: list[list[str]]) -> bool:
    text = norm_answer(answer)
    return all(any(norm_answer(alt) in text for alt in group) for group in groups)


NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def numbers_in(text: str) -> set[float]:
    text = re.sub(r"\[[\d,\s]+\]", " ", text)                # drop citation markers like [1]
    values = set()
    for token in NUMBER_RE.findall(text.replace("₹", "")):
        try:
            values.add(float(token.replace(",", "").rstrip(".")))
        except ValueError:
            pass
    return values


def numeric_groundedness(answer: str, context: str, question: str, allowed: list[str]) -> list[str]:
    """Numbers in the answer that appear neither in the context nor in the question."""
    known = numbers_in(context) | numbers_in(question) | numbers_in(" ".join(allowed))
    return sorted(str(n).rstrip("0").rstrip(".") for n in numbers_in(answer) - known)


JUDGE_PROMPT = """You check an answer against the context it was based on. Decide whether EVERY factual claim in the \
answer is supported by the context (simple arithmetic on context numbers counts as supported). \
Reply with JSON only: {"supported": true or false, "unsupported_claims": ["..."]}"""


def judge_groundedness(llm, answer: str, context: str):
    reply = llm.chat([{"role": "system", "content": JUDGE_PROMPT},
                      {"role": "user", "content": f"Context:\n{context}\n\nAnswer:\n{answer}"}], max_tokens=300)
    match = re.search(r"\{.*\}", reply, re.S)
    try:
        return bool(json.loads(match.group(0))["supported"]) if match else None
    except (ValueError, KeyError):
        return None


# ---- one question ------------------------------------------------------------------
def evaluate_question(q: dict, retriever, pipeline=None, judge: bool = False) -> dict:
    row = {"id": q["id"], "category": q["category"], "question": q["question"], "answerable": q["answerable"]}
    history = q.get("history", [])

    if pipeline is None:   # retrieval-only: use the hand-written standalone question for follow-ups
        query = q.get("standalone", q["question"])
        hits = retriever.search(query)
        row.update(standalone_question=query, top_score=round(hits[0].score, 4) if hits else 0.0)
        response = None
    else:
        started = time.perf_counter()
        response = pipeline.ask(q["question"], history)
        hits = response.retrieved
        row.update(standalone_question=response.standalone_question, top_score=round(response.top_score, 4),
                   latency_s=round(time.perf_counter() - started, 2))

    row["top_chunks"] = [f"{h.chunk.chunk_id}:{h.score:.2f}" for h in hits]
    if q["answerable"]:
        row.update(retrieval_metrics(hits, q["gold"]))
    else:
        row["rejected_at_retrieval"] = int(row["top_score"] < retriever.min_score)

    if response is not None:
        row["answer"] = response.answer
        row["refused"] = int(not response.answerable)
        row["cited_chunks"] = [h.chunk.chunk_id for _, h in response.sources]
        row["cited_pages"] = [f"{h.chunk.file_name} p{h.chunk.page_start}" for _, h in response.sources]
        if q["answerable"]:
            row["answer_correct"] = int(response.answerable and keywords_present(response.answer, q["answer_keywords"]))
            row["citation_hit"] = int(any(satisfies(h.chunk, g) for g in q["gold"] for _, h in response.sources))
            row["citation_full"] = int(all(any(satisfies(h.chunk, g) for _, h in response.sources) for g in q["gold"]))
            context = format_context([h for h in response.retrieved if h.score >= retriever.min_score])
            unsupported = numeric_groundedness(response.answer, context, q["question"], q.get("allowed_numbers", []))
            row["numbers_not_in_context"] = unsupported
            row["numeric_grounded"] = int(response.answerable and not unsupported)
            if judge and response.answerable:
                row["judge_supported"] = judge_groundedness(pipeline.llm, response.answer, context)
        else:
            row["answer_correct"] = int(not response.answerable)       # correct = it declined
            row["no_citation_shown"] = int(not response.sources)
    row["manual_correct"] = ""
    return row


# ---- summary -------------------------------------------------------------------------
def mean(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 3) if values else None


def summarize(rows: list[dict], full_run: bool) -> dict:
    def block(subset):
        ans = [r for r in subset if r["answerable"]]
        unans = [r for r in subset if not r["answerable"]]
        out = {"n": len(subset), "n_answerable": len(ans), "n_not_in_kb": len(unans)}
        for k in K_VALUES:
            out[f"hit@{k}"] = mean(r[f"hit@{k}"] for r in ans)
            out[f"recall@{k}"] = mean(r[f"recall@{k}"] for r in ans)
        out["mrr"] = mean(r["mrr"] for r in ans)
        out["rejected_at_retrieval"] = mean(r["rejected_at_retrieval"] for r in unans)
        if full_run:
            out["answer_correct_answerable"] = mean(r["answer_correct"] for r in ans)
            out["correctly_declined_not_in_kb"] = mean(r["answer_correct"] for r in unans)
            out["numeric_grounded"] = mean(r["numeric_grounded"] for r in ans)
            out["judge_supported"] = mean(1 if r.get("judge_supported") else 0 for r in ans if "judge_supported" in r)
            out["citation_hit"] = mean(r["citation_hit"] for r in ans)
            out["citation_full"] = mean(r["citation_full"] for r in ans)
            out["no_citation_when_declined"] = mean(r["no_citation_shown"] for r in unans)
            out["avg_latency_s"] = mean(r.get("latency_s") for r in subset)
        return out

    by_category = defaultdict(list)
    for r in rows:
        by_category[r["category"]].append(r)
    ans_scores = [r["top_score"] for r in rows if r["answerable"]]
    un_scores = [r["top_score"] for r in rows if not r["answerable"]]
    return {"overall": block(rows), "by_category": {c: block(v) for c, v in by_category.items()},
            "score_calibration": {"answerable_top1": ans_scores, "not_in_kb_top1": un_scores}}


def print_calibration(summary: dict) -> None:
    ans, un = summary["score_calibration"]["answerable_top1"], summary["score_calibration"]["not_in_kb_top1"]
    if not ans or not un:
        return
    print("\nSimilarity threshold calibration (top-1 cosine score of each question)")
    print(f"  answerable questions : min {min(ans):.2f}  mean {sum(ans)/len(ans):.2f}  max {max(ans):.2f}")
    print(f"  not-in-KB questions  : min {min(un):.2f}  mean {sum(un)/len(un):.2f}  max {max(un):.2f}")
    print(f"  current MIN_SIMILARITY = {config.MIN_SIMILARITY}")
    if min(ans) > max(un):
        print(f"  scores are separable: any threshold between {max(un):.2f} and {min(ans):.2f} works "
              f"(for example {(min(ans) + max(un)) / 2:.2f})")
    else:
        print("  scores overlap: no threshold separates the two groups perfectly. A lower threshold lets the LLM "
              "decide more often; a higher one rejects more genuine questions.")


def markdown_report(summary: dict, rows: list[dict], full_run: bool) -> str:
    lines = [f"# Evaluation results ({'full run' if full_run else 'retrieval only'})", "",
             f"- Embedding model: `{config.EMBEDDING_MODEL}`",
             f"- Top-k: {config.TOP_K}, MIN_SIMILARITY: {config.MIN_SIMILARITY}"]
    if full_run:
        lines.append(f"- LLM: `{config.LLM_MODEL}`")
    lines += ["- All numbers below are produced automatically by `evaluate.py`; see `results.csv` for each question.", ""]
    cols = [f"hit@{k}" for k in K_VALUES] + [f"recall@{K_VALUES[-1]}", "mrr", "rejected_at_retrieval"]
    if full_run:
        cols += ["answer_correct_answerable", "correctly_declined_not_in_kb", "numeric_grounded",
                 "citation_hit", "citation_full"]
    table = {"overall": summary["overall"], **summary["by_category"]}
    lines.append("| group | n | " + " | ".join(cols) + " |")
    lines.append("|---|---|" + "---|" * len(cols))
    for name, s in table.items():
        lines.append(f"| {name} | {s['n']} | " + " | ".join("-" if s.get(c) is None else str(s[c]) for c in cols) + " |")
    misses = [r for r in rows if r["answerable"] and r.get("hit@%d" % K_VALUES[-1]) == 0]
    if misses:
        lines += ["", f"Questions whose evidence was NOT in the top {K_VALUES[-1]}: " + ", ".join(r["id"] for r in misses)]
    return "\n".join(lines) + "\n"


def write_outputs(rows, summary, full_run: bool) -> None:
    config.EVAL_OUT_DIR.mkdir(exist_ok=True)
    prefix = "results" if full_run else "results_retrieval_only"
    (config.EVAL_OUT_DIR / f"{prefix}.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    fieldnames = list(dict.fromkeys(k for r in rows for k in r))
    with open(config.EVAL_OUT_DIR / f"{prefix}.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (config.EVAL_OUT_DIR / f"{prefix}_summary.md").write_text(markdown_report(summary, rows, full_run), encoding="utf-8")


def run_evaluation(questions, retriever, pipeline=None, judge=False, sleep=0.0, verbose=True):
    rows = []
    for q in questions:
        row = evaluate_question(q, retriever, pipeline, judge)
        rows.append(row)
        if verbose:
            if q["answerable"]:
                flag = f"hit@{K_VALUES[-1]}={row[f'hit@{K_VALUES[-1]}']}  mrr={row['mrr']:.2f}"
            else:
                flag = f"rejected_at_retrieval={row['rejected_at_retrieval']}"
            if pipeline is not None:
                flag += f"  correct={row['answer_correct']}"
            print(f"{q['id']} [{q['category']:<14}] top1={row['top_score']:.2f}  {flag}")
        if pipeline is not None and sleep:
            time.sleep(sleep)
    return rows, summarize(rows, full_run=pipeline is not None)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--retrieval-only", action="store_true", help="skip the LLM (no API key needed)")
    parser.add_argument("--judge", action="store_true", help="add an LLM-as-judge groundedness check")
    parser.add_argument("--ids", help="comma separated question ids, e.g. Q01,Q14")
    parser.add_argument("--sleep", type=float, default=0.0, help="seconds to wait between questions")
    args = parser.parse_args()

    questions = json.loads(config.EVAL_FILE.read_text(encoding="utf-8"))
    if args.ids:
        wanted = set(args.ids.split(","))
        questions = [q for q in questions if q["id"] in wanted]

    if args.retrieval_only:
        retriever, pipeline = build_retriever(), None
    else:
        pipeline = build_pipeline()
        retriever = pipeline.retriever

    rows, summary = run_evaluation(questions, retriever, pipeline, args.judge, args.sleep)
    write_outputs(rows, summary, full_run=pipeline is not None)
    print("\n" + markdown_report(summary, rows, full_run=pipeline is not None))
    print_calibration(summary)
    print(f"\nSaved to {config.EVAL_OUT_DIR}/ (csv, json, summary.md)")


if __name__ == "__main__":
    main()
