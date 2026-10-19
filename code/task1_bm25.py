import itertools
import json
import time
from collections import Counter
import numpy as np
import common as C
from bm25_lib import BM25Index

def explore(corpus, train, val, ev):
    L = []
    wl = [sum(len(s.split()) for s in ss) for ss in corpus["sents"]]
    L.append(f"docs: {len(corpus['doc_ids'])} | words/abstract mean {np.mean(wl):.0f} "
             f"median {np.median(wl):.0f} max {max(wl)} | sentences/abstract mean "
             f"{np.mean([len(s) for s in corpus['sents']]):.1f}")
    for name, cl in (("train", train), ("val", val), ("eval", ev)):
        n = len(cl)
        has = [c for c in cl if c["gold"]]
        labs = Counter(l for c in cl for l in c["gold"].values())
        multi = sum(len(c["gold"]) > 1 for c in cl)
        L.append(f"{name}: {n} claims | claim words mean {np.mean([len(c['text'].split()) for c in cl]):.1f} | "
                 f"with evidence {len(has)} | NEI {n - len(has)} | doc labels {dict(labs)} | multi-doc claims {multi}")
    L.append("sample claims: " + " || ".join(c["text"] for c in train[:3]))
    txt = "\n".join(L)
    print(txt)
    (C.OUT / "explore.txt").write_text(txt, encoding="utf-8")


def main():
    C.seed_everything()
    corpus = C.load_corpus()
    doc_ids, texts = corpus["doc_ids"], corpus["texts"]
    train, val, ev = (C.load_claims(s) for s in ("train", "val", "eval"))
    explore(corpus, train, val, ev)

    vq = [c["text"] for c in val]
    vids = [c["id"] for c in val]
    results = []
    best = None
    for stem, stop in itertools.product([False, True], [False, True]):
        idx = BM25Index(texts, stem=stem, stop=stop)
        for k1, b in itertools.product([0.4, 0.6, 0.9, 1.2, 1.5, 2.0], [0.3, 0.5, 0.75, 0.9, 1.0]):
            S = idx.score(vq, k1, b)
            nd = C.evaluate(C.scores_to_run(S, vids, doc_ids), val)["ndcg10"]
            results.append((nd, stem, stop, k1, b))
            if best is None or nd > best[0]:
                best = (nd, stem, stop, k1, b, idx)
    results.sort(reverse=True)
    print("\nTop 8 BM25 configs on validation (ndcg@10, stem, stop, k1, b):")
    for r in results[:8]:
        print("  %.4f stem=%s stop=%s k1=%.1f b=%.2f" % r)
    nd, stem, stop, k1, b, idx = best
    settings = f"k1={k1}, b={b}, stemming={stem}, stopwords_removed={stop}, tf-binary-query, lucene-idf"
    print("\nBEST:", settings, "val ndcg@10 = %.4f" % nd)

    W = idx.doc_weights(k1, b)
    mats = {}
    for split, cl in (("train", train), ("val", val), ("eval", ev)):
        mats[split] = idx.score([c["text"] for c in cl], W=W)
        C.save_scores("bm25", split, mats[split])

    run_val = C.scores_to_run(mats["val"], vids, doc_ids)
    run_eval = C.scores_to_run(mats["eval"], [c["id"] for c in ev], doc_ids)
    m = C.save_system("t1_bm25", run_eval, run_val, "t1_bm25.trec", C.PRED / "val_bm25.trec", val)
    print("val metrics:", {k: round(v, 4) for k, v in m.items()})

    lat = []
    for c in val[:100]:
        t = time.perf_counter()
        s = idx.score([c["text"]], W=W)
        C.scores_to_run(s, [c["id"]], doc_ids)
        lat.append((time.perf_counter() - t) * 1000)
    C.set_metric("bm25_latency_ms", float(np.mean(lat)))
    C.update_results("task1", {"bm25_settings": settings, "val_ndcg10": round(m["ndcg10"], 4)})
    (C.OUT / "bm25_best.json").write_text(json.dumps(
        {"k1": k1, "b": b, "stem": stem, "stop": stop}), encoding="utf-8")
    print("bm25 latency %.1f ms/query" % np.mean(lat))


if __name__ == "__main__":
    main()
