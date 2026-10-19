import argparse
import itertools
import time
import numpy as np
import common as C

def minmax(S):
    mn, mx = S.min(1, keepdims=True), S.max(1, keepdims=True)
    return (S - mn) / (mx - mn + 1e-9)


def zscore(S):
    return (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-9)


def rank_matrix(S):
    order = np.argsort(-S, axis=1, kind="stable")
    R = np.empty_like(order)
    R[np.arange(S.shape[0])[:, None], order] = np.arange(S.shape[1])[None, :]
    return R + 1


def build(spec, S):
    """spec = ('rrf', [tags], k) or ('interp', norm, {tag: weight})."""
    if spec[0] == "rrf":
        return sum(1.0 / (spec[2] + rank_matrix(S[t])) for t in spec[1]).astype(np.float32)
    _, norm, w = spec
    f = {"minmax": minmax, "zscore": zscore}[norm]
    return sum(wt * f(S[t]) for t, wt in w.items() if wt > 0).astype(np.float32)


def describe(spec):
    if spec[0] == "rrf":
        return f"RRF(k={spec[2]}) over {'+'.join(spec[1])}"
    return f"score interpolation ({spec[1]} norm) weights " + ", ".join(
        f"{t}={w:.2f}" for t, w in spec[2].items() if w > 0)


def candidate_specs(best_dense):
    specs = [("rrf", ["bm25", best_dense], 60), ("rrf", ["bm25", "general", "scientific"], 60)]
    for norm in ("minmax", "zscore"):
        for a in np.arange(0.0, 1.0001, 0.05):
            specs.append(("interp", norm, {"bm25": float(a), best_dense: float(1 - a)}))
        for wb, wg in itertools.product(np.arange(0, 1.0001, 0.1), repeat=2):
            ws = 1 - wb - wg
            if ws >= -1e-9:
                specs.append(("interp", norm, {"bm25": float(wb), "general": float(wg),
                                               "scientific": float(max(ws, 0))}))
    return specs

def rerank_run(first, ce, k):
    out = {}
    for cid, lst in first.items():
        sc = np.asarray(ce[cid][:k], dtype=float)
        order = np.argsort(-sc, kind="stable")
        new = [(lst[i][0], float(sc[i])) for i in order]
        low = new[-1][1]
        rest = [(d, low - 1.0 - 0.001 * j) for j, (d, _) in enumerate(lst[k:])]
        out[cid] = new + rest
    return out

def ce_scores(ce, claims, run, texts, index, k, bs=32):
    pairs, owner = [], []
    for c in claims:
        for d, _ in run[c["id"]][:k]:
            pairs.append((c["text"], texts[index[d]]))
            owner.append(c["id"])
    preds = ce.predict(pairs, batch_size=bs, show_progress_bar=True)
    out = {c["id"]: [] for c in claims}
    for cid, p in zip(owner, preds):
        out[cid].append(float(np.ravel(p)[-1]))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reranker", default="cross-encoder/ms-marco-MiniLM-L-12-v2")
    ap.add_argument("--ks", type=int, nargs="+", default=[10, 20, 30, 50])
    ap.add_argument("--only_if_better", action="store_true",
                    help="keep the stored rerank run/results unless this run improves validation nDCG@10")
    a = ap.parse_args()
    C.seed_everything()

    corpus = C.load_corpus()
    doc_ids, texts, index = corpus["doc_ids"], corpus["texts"], corpus["index"]
    val, ev = C.load_claims("val"), C.load_claims("eval")
    vids, eids = [c["id"] for c in val], [c["id"] for c in ev]
    tags = ["bm25", "general", "scientific"]
    SV = {t: C.load_scores(t, "val") for t in tags}
    SE = {t: C.load_scores(t, "eval") for t in tags}

    nd = {t: C.evaluate(C.scores_to_run(SV[t], vids, doc_ids), val)["ndcg10"] for t in tags}
    best_dense = max(("general", "scientific"), key=lambda t: nd[t])
    print("single systems (val ndcg@10):", {k: round(v, 4) for k, v in nd.items()}, "| best dense:", best_dense)

    best = None
    for spec in candidate_specs(best_dense):
        v = C.evaluate(C.scores_to_run(build(spec, SV), vids, doc_ids), val)["ndcg10"]
        if best is None or v > best[0]:
            best = (v, spec)
    hyb_nd, spec = best
    hybrid_method = describe(spec)
    print("BEST hybrid:", hybrid_method, "val ndcg@10 = %.4f" % hyb_nd)
    HV, HE = build(spec, SV), build(spec, SE)
    C.save_scores("hybrid", "val", HV)
    C.save_scores("hybrid", "eval", HE)
    rv = C.scores_to_run(HV, vids, doc_ids)
    re_ = C.scores_to_run(HE, eids, doc_ids)
    mh = C.save_system("t3_hybrid", re_, rv, "t3_hybrid.trec", C.RUNS / "val_hybrid.trec", val)

    cands = {"bm25": nd["bm25"], "general": nd["general"], "scientific": nd["scientific"], "hybrid": mh["ndcg10"]}
    first_name = max(cands, key=cands.get)
    FV = {"bm25": SV["bm25"], "general": SV["general"], "scientific": SV["scientific"], "hybrid": HV}[first_name]
    FE = {"bm25": SE["bm25"], "general": SE["general"], "scientific": SE["scientific"], "hybrid": HE}[first_name]
    fv = C.scores_to_run(FV, vids, doc_ids)
    fe = C.scores_to_run(FE, eids, doc_ids)
    print("first stage for reranking:", first_name, "(val ndcg@10 %.4f)" % cands[first_name])

    from sentence_transformers import CrossEncoder
    rr_path, rr_label = C.resolve_model(a.reranker)
    ce = CrossEncoder(rr_path, max_length=512, device=C.device())
    kmax = max(a.ks)
    cev = ce_scores(ce, val, fv, texts, index, kmax)
    best_k, best_nd = None, -1
    for k in a.ks:
        v = C.evaluate(rerank_run(fv, cev, k), val)["ndcg10"]
        print(f"  rerank top-{k}: val ndcg@10 = {v:.4f}")
        if v > best_nd:
            best_k, best_nd = k, v
    print("BEST k =", best_k)
    prev = C.load_registry().get("t3_rerank", {}).get("val_ndcg10")
    if a.only_if_better and prev is not None and best_nd <= prev:
        print(f"Not better than the stored rerank (val ndcg@10 {prev:.4f}); keeping previous rerank run and results.")
        return
    run_val = rerank_run(fv, cev, best_k)
    cee = ce_scores(ce, ev, fe, texts, index, best_k)
    run_eval = rerank_run(fe, cee, best_k)
    mr = C.save_system("t3_rerank", run_eval, run_val, "t3_rerank.trec", C.RUNS / "val_rerank.trec", val)

    lat = []
    for c in val[:50]:
        pairs = [(c["text"], texts[index[d]]) for d, _ in fv[c["id"]][:best_k]]
        t = C.now()
        ce.predict(pairs, batch_size=32, show_progress_bar=False)
        lat.append((C.now() - t) * 1000)
    rerank_lat = float(np.mean(lat[3:]))

    bm_lat = C.get_metric("bm25_latency_ms", 0.0)
    dense_lat = C.get_metric(f"dense_latency_ms_{best_dense}", 0.0)
    used = spec[1] if spec[0] == "rrf" else [t for t, w in spec[2].items() if w > 0]
    hyb_lat = sum(C.get_metric("bm25_latency_ms", 0.0) if t == "bm25" else C.get_metric(f"dense_latency_ms_{t}", 0.0)
                  for t in used)
    best_single = max(nd["bm25"], nd[best_dense])
    summary = {
        "val_ndcg10": {"bm25": nd["bm25"], "best_dense": nd[best_dense], "hybrid": mh["ndcg10"], "rerank": mr["ndcg10"]},
        "gain_hybrid_over_best_single": mh["ndcg10"] - best_single,
        "gain_rerank_over_first_stage": mr["ndcg10"] - cands[first_name],
        "latency_ms": {"bm25": bm_lat, "dense": dense_lat, "hybrid_approx": hyb_lat, "rerank_extra": rerank_lat},
        "first_stage": first_name, "best_dense": best_dense, "k": best_k, "hybrid_method": hybrid_method,
    }
    C.set_metric("task3_summary", summary)
    print("summary:", summary)
    C.update_results("task3", {
        "hybrid_method": hybrid_method, "reranker_model": rr_label, "rerank_top_k": best_k,
        "val_ndcg10_hybrid": round(mh["ndcg10"], 4), "val_ndcg10_rerank": round(mr["ndcg10"], 4),
        "rerank_query_latency_ms": round(rerank_lat, 2)})


if __name__ == "__main__":
    main()
