import argparse
import csv
import json
import re
import numpy as np
import common as C
import verify_lib as V

NEG_WORDS = {"no", "not", "never", "without", "lack", "lacks", "lacking", "fail", "fails", "failed",
             "absent", "absence", "unable", "cannot", "neither", "nor", "non", "none", "nothing"}


def has_negation(text):
    toks = re.findall(r"[a-z']+", text.lower())
    return any(t in NEG_WORDS or t.endswith("n't") for t in toks)


def has_number(text):
    return bool(re.search(r"\d", text))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nli", default="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
                    help="HF model name or local folder (e.g. cache/nli_ft)")
    ap.add_argument("--sent_model", default=None, help="sentence-selection bi-encoder (default: Task 2 general model)")
    ap.add_argument("--ncand", type=int, default=5)
    ap.add_argument("--nsent", type=int, default=3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--only_if_better", action="store_true")
    args = ap.parse_args()

    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    C.seed_everything()
    dev = C.device()
    corpus = C.load_corpus()
    val, ev = C.load_claims("val"), C.load_claims("eval")
    claims_by = {"val": val, "eval": ev}
    all_ids = [c["id"] for c in val + ev]
    assert len(all_ids) == len(set(all_ids)), "claim ids overlap between val and eval"
    nli_path, nli_label = C.resolve_model(args.nli)

    reg = C.load_registry()
    best_name = max(reg, key=lambda k: reg[k]["val_ndcg10"])
    print("retriever used:", best_name, "val ndcg@10 = %.4f" % reg[best_name]["val_ndcg10"])
    runs = {"val": C.read_trec(reg[best_name]["val_run"]), "eval": C.read_trec(reg[best_name]["eval_run"])}
    cand = {s: {c["id"]: [d for d, _ in runs[s][c["id"]][:args.ncand]] for c in claims_by[s]} for s in runs}

    res = C.load_results()
    sent_name = args.sent_model or res["task2"].get("general_model") or "BAAI/bge-base-en-v1.5"
    sm = SentenceTransformer(sent_name, device=dev)
    qp, dp = C.query_prefix(sent_name), C.doc_prefix(sent_name)
    pairs = [(c["id"], d) for s in claims_by for c in claims_by[s] for d in cand[s][c["id"]]]
    sel = V.select_sentences(sm, qp, dp, corpus, val + ev, pairs, args.nsent)
    del sm
    if dev == "cuda":
        torch.cuda.empty_cache()

    texts_in, meta = [], []  
    for s, cl in claims_by.items():
        for c in cl:
            for d in cand[s][c["id"]]:
                idx, chosen = sel[(c["id"], d)]
                if not chosen:
                    continue
                for t in chosen:
                    texts_in.append((t, c["text"]))
                    meta.append((s, c["id"], d, False))
                texts_in.append((" ".join(chosen), c["text"]))
                meta.append((s, c["id"], d, True))

    tok = AutoTokenizer.from_pretrained(nli_path)
    nli = AutoModelForSequenceClassification.from_pretrained(nli_path).to(dev).eval()
    id2l = {int(i): l.lower() for i, l in nli.config.id2label.items()}
    i_ent = next(i for i, l in id2l.items() if l.startswith("entail"))
    i_con = next(i for i, l in id2l.items() if l.startswith("contra"))
    i_neu = next(i for i, l in id2l.items() if l.startswith("neutral"))
    print(f"NLI {nli_label}: {len(texts_in)} pairs, labels {id2l}")
    order = np.argsort([len(p[0]) for p in texts_in])
    probs = np.zeros((len(texts_in), 3), dtype=np.float32)  # ent, con, neu
    with torch.no_grad():
        for s in range(0, len(order), args.bs):
            ids = order[s:s + args.bs]
            enc = tok([texts_in[i][0] for i in ids], [texts_in[i][1] for i in ids], truncation="only_first",
                      max_length=256, padding=True, return_tensors="pt").to(dev)
            with torch.autocast(device_type=dev, dtype=torch.float16, enabled=dev == "cuda"):
                logits = nli(**enc).logits
            p = torch.softmax(logits.float(), -1).cpu().numpy()
            probs[ids, 0], probs[ids, 1], probs[ids, 2] = p[:, i_ent], p[:, i_con], p[:, i_neu]

    feats = {}
    for (s, cid, d, is_cat), (pe, pc, pn) in zip(meta, probs):
        f = feats.setdefault((s, cid, d), [0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
        if is_cat:
            f[2], f[3], f[5] = float(pe), float(pc), float(pn)
        else:
            f[0], f[1], f[4] = max(f[0], float(pe)), max(f[1], float(pc)), min(f[4], float(pn))

    arr = V.build_arrays(val, cand["val"], feats, "val", args.ncand)
    grid = np.round(np.arange(0.10, 0.991, 0.05), 2)
    gates = [0.5, 0.7, 0.9, 1.01] 
    best = V.grid_search(arr, [1, 2, 3, args.ncand], grid, grid, gates, [1, 2, 3])
    _, _, _, mode, n, ts, tc, gn, maxk = best
    pred_val = V.predict(val, cand["val"], feats, "val", mode, n, ts, tc, gn, maxk)
    f, pr, rc = V.f1(pred_val, val)
    print(f"BEST val triple-F1 {f:.4f} (P {pr:.3f} R {rc:.3f}) | mode={mode} N={n} thr_support={ts} "
          f"thr_contradict={tc} neutral_gate={gn} max_docs={maxk}")

    prev = C._load_json(C.OUT / "task5_stats.json", {}).get("val_f1")
    if args.only_if_better and prev is not None and f <= prev:
        print(f"Not better than the stored result (val F1 {prev:.4f}); keeping previous predictions and results.")
        return

    pred_eval = V.predict(ev, cand["eval"], feats, "eval", mode, n, ts, tc, gn, maxk)

    def write_ver(path, pred):
        with open(path, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["claim_id", "doc_id", "label"])
            for cid in sorted(pred):
                for d, l in pred[cid]:
                    w.writerow([cid, d, l])

    write_ver(C.PRED / "t5_val_verification.csv", pred_val)
    write_ver(C.PRED / "t5_verification.csv", pred_eval)

    cats, rows = {}, []
    for c in val:
        gold, pred = dict(c["gold"]), dict(pred_val.get(c["id"], []))
        if set(gold.items()) == set(pred.items()):
            continue
        pool = set(cand["val"][c["id"]][:n])
        if not gold:
            cat = "other"                                  
        elif not (set(gold) & pool):
            cat = "retrieval_miss"                         
        elif len(gold) > 1 and set(gold) - set(pred):
            cat = "needs_multiple_docs"
        elif has_negation(c["text"]):
            cat = "negation"
        elif has_number(c["text"]):
            cat = "numerical"
        elif pred and not (set(pred) & set(gold)):
            cat = "entity_mismatch"                        
        else:
            cat = "other"
        rows.append((c["id"], cat))
        cats[cat] = cats.get(cat, 0) + 1
    with open(C.PRED / "t5_val_errors.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["claim_id", "category"])
        w.writerows(rows)
    n_err = len(rows)
    frac = cats.get("retrieval_miss", 0) / max(1, n_err)
    print(f"{n_err}/{len(val)} validation claims are wrong | categories: {cats} | retrieval_miss fraction {frac:.2f}")

    C.update_results("task5", {"retriever_used": best_name, "nli_model": nli_label,
                               "zero_shot_or_finetuned": "finetuned" if "fine-tuned" in nli_label else "zero-shot",
                               "val_f1": round(f, 4)})
    (C.OUT / "task5_stats.json").write_text(json.dumps(
        {"val_f1": f, "precision": pr, "recall": rc, "n_errors": n_err, "categories": cats,
         "retrieval_miss_fraction": frac,
         "config": {"mode": mode, "N": n, "thr_support": ts, "thr_contradict": tc, "neutral_gate": gn,
                    "max_docs": maxk}}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
