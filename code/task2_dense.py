import argparse
import os
import time
import numpy as np
import common as C

def run_role(role, model_name):
    import torch
    from sentence_transformers import SentenceTransformer

    dev = C.device()
    print(f"\n=== {role}: {model_name} on {dev}")
    model = SentenceTransformer(model_name, device=dev)
    try:
        model.max_seq_length = 512
    except Exception:
        pass
    if dev == "cuda":
        model.half()
    qp, dp = C.query_prefix(model_name), C.doc_prefix(model_name)

    corpus = C.load_corpus()
    doc_ids = corpus["doc_ids"]
    train, val, ev = (C.load_claims(s) for s in ("train", "val", "eval"))

    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = C.now()
    D = model.encode([dp + t for t in corpus["texts"]], batch_size=64, convert_to_numpy=True,
                     normalize_embeddings=True, show_progress_bar=True).astype(np.float32)
    index_time = C.now() - t0
    if dev == "cuda":
        mem = torch.cuda.max_memory_allocated() / 2 ** 20
    else:
        import psutil
        mem = psutil.Process(os.getpid()).memory_info().rss / 2 ** 20
    np.save(C.CACHE / f"emb_{role}_docs.npy", D)

    mats = {}
    for split, cl in (("train", train), ("val", val), ("eval", ev)):
        Q = model.encode([qp + c["text"] for c in cl], batch_size=128, convert_to_numpy=True,
                         normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
        mats[split] = Q @ D.T
        C.save_scores(role, split, mats[split])

    def one(c):
        q = model.encode([qp + c["text"]], convert_to_numpy=True, normalize_embeddings=True,
                         show_progress_bar=False).astype(np.float32)
        s = D @ q[0]
        idx = np.argpartition(-s, 99)[:100]
        return idx[np.argsort(-s[idx])]

    for c in val[:5]:
        one(c) 
    lat = []
    for c in val[:100]:
        t = C.now()
        one(c)
        lat.append((C.now() - t) * 1000)
    lat = float(np.mean(lat))

    run_val = C.scores_to_run(mats["val"], [c["id"] for c in val], doc_ids)
    run_eval = C.scores_to_run(mats["eval"], [c["id"] for c in ev], doc_ids)
    val_path = C.PRED / "val_general.trec" if role == "general" else C.RUNS / "val_scientific.trec"
    m = C.save_system(f"t2_{role}", run_eval, run_val, f"t2_{role}.trec", val_path, val)
    print({k: round(v, 4) for k, v in m.items()},
          f"| index {index_time:.1f}s | latency {lat:.1f} ms | mem {mem:.0f} MB")

    C.update_results("task2", {
        f"{role}_model": model_name,
        f"val_ndcg10_{role}": round(m["ndcg10"], 4),
        "index_time_sec": {role: round(index_time, 2)},
        "query_latency_ms": {role: round(lat, 2)},
        "memory_mb": {role: round(mem, 1)},
    })
    C.set_metric(f"dense_latency_ms_{role}", lat)
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--general", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--scientific", default="pritamdeka/S-PubMedBert-MS-MARCO")
    ap.add_argument("--only", choices=["general", "scientific"], default=None)
    a = ap.parse_args()
    C.seed_everything()
    if a.only in (None, "general"):
        run_role("general", a.general)
    if a.only in (None, "scientific"):
        run_role("scientific", a.scientific)


if __name__ == "__main__":
    main()
