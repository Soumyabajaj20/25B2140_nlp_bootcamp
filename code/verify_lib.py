import itertools
import numpy as np

MODES = ("units", "concat", "both")

def select_sentences(sm, qp, dp, corpus, claims, pairs, nsent, batch=128):
    """For each (claim_id, doc_id) pick the nsent abstract sentences most similar to the claim.
    sm: SentenceTransformer. Returns {(cid, did): (sorted_sentence_indices, [sentence_texts])}."""
    index, sents = corpus["index"], corpus["sents"]
    docs = sorted({d for _, d in pairs})
    flat, offs = [], {}
    for d in docs:
        ss = sents[index[d]]
        offs[d] = (len(flat), len(flat) + len(ss))
        flat.extend(dp + s for s in ss)
    print(f"encoding {len(flat)} sentences from {len(docs)} abstracts")
    E = sm.encode(flat, batch_size=batch, convert_to_numpy=True, normalize_embeddings=True,
                  show_progress_bar=True)
    text_of = {c["id"]: c["text"] for c in claims}
    cids = sorted({c for c, _ in pairs})
    Q = sm.encode([qp + text_of[c] for c in cids], batch_size=batch, convert_to_numpy=True,
                  normalize_embeddings=True, show_progress_bar=False)
    qv = dict(zip(cids, Q))
    out = {}
    for cid, did in pairs:
        a, b = offs[did]
        ss = sents[index[did]]
        if b <= a:
            out[(cid, did)] = ([], [])
            continue
        sim = E[a:b] @ qv[cid]
        top = sorted(int(i) for i in np.argsort(-sim)[:nsent])  
        out[(cid, did)] = (top, [ss[i] for i in top])
    return out

def _triple(f, mode):
    if mode == "units":
        return f[0], f[1], f[4]
    if mode == "concat":
        return f[2], f[3], f[5]
    return max(f[0], f[2]), max(f[1], f[3]), min(f[4], f[5])


def predict(claims, cand, feats, split, mode, n, ts, tc, gn, maxk):
    """cand: {cid: [doc ids ranked]}. Accept a doc if its best class is SUPPORT/CONTRADICT with
    prob >= threshold and P(neutral) <= gn. Keep at most maxk docs, best first."""
    out = {}
    for c in claims:
        got = []
        for d in cand[c["id"]][:n]:
            f = feats.get((split, c["id"], d))
            if f is None:
                continue
            pe, pc, pn = _triple(f, mode)
            if pn > gn:
                continue
            if pe >= pc and pe >= ts:
                got.append((pe, d, "SUPPORT"))
            elif pc > pe and pc >= tc:
                got.append((pc, d, "CONTRADICT"))
        got.sort(key=lambda x: -x[0])  # stable: ties keep retrieval order
        out[c["id"]] = [(d, l) for _, d, l in got[:maxk]]
    return out

def f1(pred, claims):
    """Micro P/R/F1 over (claim, doc, label) triples. Returns (f1, precision, recall)."""
    tp = fp = fn = 0
    for c in claims:
        g, p = set(c["gold"].items()), set(pred.get(c["id"], []))
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
    pr, rc = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return (2 * pr * rc / (pr + rc) if pr + rc else 0.0), pr, rc

def build_arrays(claims, cand, feats, split, ncand):
    """Dense arrays (n_claims x ncand) of probabilities per mode, plus gold label codes."""
    nc = len(claims)
    arr = {m: [np.zeros((nc, ncand)), np.zeros((nc, ncand)), np.ones((nc, ncand))] for m in MODES}
    gold = np.zeros((nc, ncand), dtype=np.int8)
    total = 0
    for i, c in enumerate(claims):
        total += len(c["gold"])
        for r, d in enumerate(cand[c["id"]][:ncand]):
            lab = c["gold"].get(d)
            gold[i, r] = 1 if lab == "SUPPORT" else (2 if lab == "CONTRADICT" else 0)
            f = feats.get((split, c["id"], d))
            if f is None:
                continue
            for m in MODES:
                pe, pc, pn = _triple(f, m)
                arr[m][0][i, r], arr[m][1][i, r], arr[m][2][i, r] = pe, pc, pn
    arr["gold"], arr["total_gold"] = gold, total
    return arr

def grid_search(arr, ns, grid_s, grid_c, gates, maxks):
    """Exhaustive search maximising triple-level F1. Returns (f1, P, R, mode, n, ts, tc, gn, maxk)."""
    best = None
    gold, total = arr["gold"], max(1, arr["total_gold"])
    for mode in MODES:
        pe, pc, pn = arr[mode]
        s = np.maximum(pe, pc)
        lab = np.where(pe >= pc, 1, 2)
        match = (lab == gold) & (gold > 0)
        for n in ns:
            s_n, lab_n, pn_n, m_n = s[:, :n], lab[:, :n], pn[:, :n], match[:, :n]
            for ts, tc, gn in itertools.product(grid_s, grid_c, gates):
                thr = np.where(lab_n == 1, ts, tc)
                score = np.where((s_n >= thr) & (pn_n <= gn), s_n, -1.0)
                order = np.argsort(-score, axis=1, kind="stable")
                for k in maxks:
                    idx = order[:, :k]
                    taken = np.take_along_axis(score, idx, 1) >= 0
                    tp = int((np.take_along_axis(m_n, idx, 1) & taken).sum())
                    sel = int(taken.sum())
                    pr, rc = tp / max(1, sel), tp / total
                    f = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
                    if best is None or f > best[0]:
                        best = (f, pr, rc, mode, n, float(ts), float(tc), float(gn), k)
    return best
