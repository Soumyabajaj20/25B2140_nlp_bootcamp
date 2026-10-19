import argparse
import re
import subprocess
import sys
import zipfile

import common as C

STOP = {"the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "is", "are", "was", "were", "with",
        "by", "as", "at", "that", "this", "it", "be", "from", "has", "have", "than", "can", "do", "does"}

PRED_FILES = ["t1_bm25.trec", "t2_general.trec", "t2_scientific.trec", "t3_hybrid.trec", "t3_rerank.trec",
              "t4_random.trec", "t4_hard.trec", "val_bm25.trec", "val_general.trec",
              "t5_verification.csv", "t5_val_verification.csv", "t5_val_errors.csv"]

def overlap(claim, doc_text):
    toks = [t for t in re.findall(r"\w+", claim.lower()) if t not in STOP]
    dt = set(re.findall(r"\w+", doc_text.lower()))
    return sum(t in dt for t in toks) / max(1, len(toks))


def pick_q1(val):
    pb = C.per_claim_metrics(C.read_trec(C.PRED / "val_bm25.trec"), val)
    pg = C.per_claim_metrics(C.read_trec(C.PRED / "val_general.trec"), val)
    diff = {cid: pb[cid]["ndcg10"] - pg[cid]["ndcg10"] for cid in pb}
    bm_w = sorted((c for c in diff if diff[c] > 0), key=lambda c: -diff[c])[:2]
    ge_w = sorted((c for c in diff if diff[c] < 0), key=lambda c: diff[c])[:2]
    for lst, key in ((bm_w, lambda c: -diff[c]), (ge_w, lambda c: diff[c])):
        if len(lst) < 2:
            print("WARNING: fewer than 2 claims where this system truly wins; filling with closest claims")
            lst.extend([c for c in sorted(diff, key=key) if c not in lst][:2 - len(lst)])
    corpus = C.load_corpus()
    by = {c["id"]: c for c in val}
    print("\n--- Q1 candidates (use these facts to write your explanation) ---")
    for title, ids in (("BM25 beats general", bm_w), ("general beats BM25", ge_w)):
        for cid in ids:
            c = by[cid]
            gd = list(c["gold"])[0]
            gtext = corpus["texts"][corpus["index"][gd]]
            print(f"[{title}] claim {cid}: {c['text']}\n    ndcg bm25={pb[cid]['ndcg10']:.2f} general={pg[cid]['ndcg10']:.2f} | "
                  f"gold doc {gd} '{corpus['titles'][corpus['index'][gd]][:80]}' | "
                  f"claim-term overlap with gold abstract {overlap(c['text'], gtext):.0%}")
    return bm_w, ge_w

def sync_results():
    """Fill val_ndcg10 fields that are None in results.json from outputs/registry.json."""
    reg = C.load_registry()
    res = C.load_results()
    fill = {
        "task1": {"val_ndcg10": "t1_bm25"},
        "task2": {"val_ndcg10_general": "t2_general", "val_ndcg10_scientific": "t2_scientific"},
        "task3": {"val_ndcg10_hybrid": "t3_hybrid", "val_ndcg10_rerank": "t3_rerank"},
        "task4": {"val_ndcg10_random_negatives": "t4_random", "val_ndcg10_hard_negatives": "t4_hard"},
    }
    for sec, keys in fill.items():
        for key, name in keys.items():
            if res.get(sec, {}).get(key) is None and name in reg:
                C.update_results(sec, {key: round(reg[name]["val_ndcg10"], 4)})
                print(f"restored {sec}.{key} = {reg[name]['val_ndcg10']:.4f} from registry")
    res = C.load_results()

    def walk(d, path=""):
        for k, v in d.items():
            if isinstance(v, dict):
                yield from walk(v, f"{path}{k}.")
            elif v is None or v == "" or (isinstance(v, list) and None in v):
                yield f"{path}{k}"
    empty = list(walk(res))
    if empty:
        print("WARNING: results.json still has empty fields:", ", ".join(empty))
        print("         (re-run the task that fills them, then run finalize.py again)")

def fmt(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) else "n/a"

def words(s):
    return len(s.split())

NEXT_STEP = {
    "retrieval_miss": "improve first-stage recall (e.g. fine-tune on train with more epochs or add a second scientific dense model to the fusion)",
    "negation": "fine-tune the NLI model on SciFact training claims, since zero-shot NLI mishandles negation",
    "numerical": "add a numeric-aware check or fine-tune NLI on training claims with number-heavy evidence",
    "entity_mismatch": "rerank candidates with a cross-encoder fine-tuned on SciFact before verification",
    "needs_multiple_docs": "aggregate evidence across the top documents instead of verifying each one independently",
    "other": "fine-tune the NLI model on the training claims",
}


def draft_report(res):
    m3 = C.get_metric("task3_summary", {})
    t5 = C._load_json(C.OUT / "task5_stats.json", {})
    logs = {k: C._load_json(C.OUT / f"task4_{k}_log.json", {}) for k in ("random", "hard")}
    q = {}
    q[1] = ("[EDIT] BM25 wins on claims {b} where the claim's rare technical terms appear verbatim in the gold "
            "abstract, which exact matching rewards while the dense model drifts to topically similar abstracts. "
            "general wins on {g} where the claim paraphrases the evidence with different wording, which BM25 "
            "cannot match lexically.").format(b=res["report_q1"]["bm25_wins"], g=res["report_q1"]["general_wins"])
    if m3:
        gh, gr = m3["gain_hybrid_over_best_single"], m3["gain_rerank_over_first_stage"]
        top, lat = ("the hybrid fusion", m3["latency_ms"]["hybrid_approx"]) if gh >= gr else \
            ("the cross-encoder rerank", m3["latency_ms"]["rerank_extra"])
        q[2] = (f"{top[0].upper() + top[1:]} gave the largest validation gain (hybrid {gh:+.3f}, rerank {gr:+.3f} nDCG@10). "
                f"Its cost is about {lat:.0f} ms per query (BM25 {m3['latency_ms']['bm25']:.0f} ms, dense "
                f"{m3['latency_ms']['dense']:.0f} ms, rerank of top-{m3['k']} adds {m3['latency_ms']['rerank_extra']:.0f} ms).")
    else:
        q[2] = "[EDIT] fill in after Task 3"
    if logs["random"] and logs["hard"]:
        vr, vh = res["task4"]["val_ndcg10_random_negatives"], res["task4"]["val_ndcg10_hard_negatives"]
        if vr is None:
            vr = logs["random"]["final_val"]["ndcg10"]
        if vh is None:
            vh = logs["hard"]["final_val"]["ndcg10"]
        hh = logs["hard"]["history"]
        gap = hh[-1]["train_ndcg10"] - hh[-1]["val_ndcg10"]
        q[3] = (f"[EDIT] Validation nDCG@10: random negatives {vr:.3f}, hard negatives {vh:.3f} "
                f"({'hard helped' if vh > vr else 'hard did not help'}). I tracked train-subset vs validation nDCG@10 "
                f"each epoch; the final train-val gap for hard negatives is {gap:.3f} and the best epoch was "
                f"{logs['hard']['best_epoch']} of {logs['hard']['args']['epochs']}. [Say whether val flattened/dropped while train kept rising.]")
    else:
        q[3] = "[EDIT] fill in after Task 4"
    if t5:
        q[4] = (f"{t5['categories'].get('retrieval_miss', 0)} of {t5['n_errors']} validation verification errors are "
                f"retrieval_miss, i.e. {t5['retrieval_miss_fraction']:.0%}. Full breakdown: {t5['categories']}.")
        top_cat = max((k for k in t5["categories"]), key=lambda k: t5["categories"][k], default="other")
        q[5] = (f"[EDIT] The most common error category is {top_cat}, so I would next {NEXT_STEP.get(top_cat, NEXT_STEP['other'])}. "
                f"[Explain why you expect it to help.]")
    else:
        q[4], q[5] = "[EDIT] fill in after Task 5", "[EDIT] fill in after Task 5"
    out = "# Report\n\n"
    titles = {1: "Q1. BM25 vs general dense (validation)", 2: "Q2. Largest Task 3 gain and its latency cost",
              3: "Q3. Hard negatives and overfitting", 4: "Q4. Fraction of verification errors that are retrieval_miss",
              5: "Q5. Single change to try next"}
    for i in range(1, 6):
        if words(q[i]) > 100:
            print(f"WARNING: Q{i} draft has {words(q[i])} words (> 100); shorten it")
        out += f"## {titles[i]}\n\n{q[i]}\n\n"
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite_report", action="store_true")
    a = ap.parse_args()
    val = C.load_claims("val")
    bm_w, ge_w = pick_q1(val)
    C.update_results("report_q1", {"bm25_wins": [int(x) for x in bm_w], "general_wins": [int(x) for x in ge_w]})
    sync_results()
    res = C.load_results()
    rp = C.ROOT / "report.md"
    if a.overwrite_report or not rp.exists():
        rp.write_text(draft_report(res), encoding="utf-8")
        print("\nWrote draft", rp, "- edit the [EDIT] parts, then re-run finalize.py")
    else:
        print("\nreport.md exists - kept as is")
    for w in re.findall(r"## (Q\d)[^\n]*\n\n(.*?)\n\n", rp.read_text(encoding="utf-8"), flags=re.S):
        if words(w[1]) > 100:
            print(f"WARNING: {w[0]} has {words(w[1])} words (> 100)")
    if "[EDIT]" in rp.read_text(encoding="utf-8"):
        print("NOTE: report.md still contains [EDIT] markers")

    # ---- zip
    zpath = C.ROOT / f"{C.ROLL_NO}_nlp_bootcamp.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted((C.ROOT / "code").glob("*")):
            if p.is_file() and p.suffix in (".py", ".md", ".txt"):
                z.write(p, f"code/{p.name}")
        for name in PRED_FILES:
            p = C.PRED / name
            if p.exists():
                z.write(p, f"predictions/{name}")
            else:
                print("MISSING prediction file:", name)
        z.write(C.ROOT / "results.json", "results.json")
        z.write(rp, "report.md")
    print("built", zpath)
    chk = C.ROOT / "check_format.py"
    if chk.exists():
        subprocess.run([sys.executable, str(chk), str(zpath)], cwd=str(C.ROOT))
    else:
        print("Copy check_format.py into", C.ROOT, "and run: python check_format.py", zpath.name)


if __name__ == "__main__":
    main()
