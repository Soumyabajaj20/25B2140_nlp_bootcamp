import argparse
import json
import math
import random
import numpy as np
import common as C

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["random", "hard"], required=True)
    ap.add_argument("--base", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=32, help="lower to 16 if you run out of GPU memory")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--scale", type=float, default=20.0)
    ap.add_argument("--hn_topk", type=int, default=30)
    ap.add_argument("--hn_skip", type=int, default=0, help="skip the top-N ranks when mining (false-negative guard)")
    ap.add_argument("--seed", type=int, default=C.SEED)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from sentence_transformers import SentenceTransformer

    C.seed_everything(args.seed)
    dev = C.device()
    use_amp = dev == "cuda"
    corpus = C.load_corpus()
    doc_ids, texts, index = corpus["doc_ids"], corpus["texts"], corpus["index"]
    train, val, ev = (C.load_claims(s) for s in ("train", "val", "eval"))
    qp, dp = C.query_prefix(args.base), C.doc_prefix(args.base)
    gold = {c["id"]: set(c["gold"]) for c in train}
    pairs = [(c["id"], c["text"], d) for c in train for d in c["gold"] if d in index]
    print(f"{len(pairs)} (claim, gold doc) training pairs from {sum(bool(c['gold']) for c in train)} claims")

    pool = {}
    if args.mode == "hard":
        mats = [C.load_scores(t, "train") for t in ("bm25", "general", "scientific")]
        for i, c in enumerate(train):
            seen, cand = set(), []
            for S in mats:
                top = np.argsort(-S[i])[: args.hn_topk]
                for j in top[args.hn_skip:]:
                    d = doc_ids[j]
                    if d not in gold[c["id"]] and d not in seen:
                        seen.add(d)
                        cand.append(d)
            pool[c["id"]] = cand
        print("avg hard-negative pool size: %.1f" % np.mean([len(v) for v in pool.values()]))

    def sample_neg(cid, rng):
        if args.mode == "hard" and pool.get(cid):
            return rng.choice(pool[cid])
        while True:
            d = rng.choice(doc_ids)
            if d not in gold[cid]:
                return d

    model = SentenceTransformer(args.base, device=dev)
    model.max_seq_length = args.max_len

    def embed(batch_texts):
        feats = model.tokenize(batch_texts)
        feats = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in feats.items()}
        return F.normalize(model(feats)["sentence_embedding"].float(), dim=-1)

    train_eval = [c for c in train if c["gold"]][:200]

    def evaluate_model():
        model.eval()
        model.max_seq_length = 512
        with torch.no_grad(), torch.autocast(device_type=dev, dtype=torch.float16, enabled=use_amp):
            D = model.encode([dp + t for t in texts], batch_size=64, convert_to_numpy=True,
                             normalize_embeddings=True, show_progress_bar=False).astype(np.float32)

            def q(cl):
                return model.encode([qp + c["text"] for c in cl], batch_size=128, convert_to_numpy=True,
                                    normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
            Qv, Qt = q(val), q(train_eval)
        model.max_seq_length = args.max_len
        model.train()
        v = C.evaluate(C.scores_to_run(Qv @ D.T, [c["id"] for c in val], doc_ids), val)["ndcg10"]
        t = C.evaluate(C.scores_to_run(Qt @ D.T, [c["id"] for c in train_eval], doc_ids), train_eval)["ndcg10"]
        return v, t, D

    def snapshot():
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.train()
    v0, t0, _ = evaluate_model()
    print(f"epoch 0 (zero-shot): val ndcg@10 {v0:.4f} | train-subset ndcg@10 {t0:.4f}")
    history = [{"epoch": 0, "train_loss": None, "val_ndcg10": v0, "train_ndcg10": t0}]
    best_v, best_ep, best_state = v0, 0, snapshot()

    steps_per_epoch = math.ceil(len(pairs) / args.bs)
    total, warm = steps_per_epoch * args.epochs, max(1, int(0.1 * steps_per_epoch * args.epochs))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for ep in range(1, args.epochs + 1):
        order = np.random.RandomState(args.seed + ep).permutation(len(pairs))
        neg_rng = random.Random(args.seed * 1000 + ep)
        losses = []
        for s in range(0, len(order), args.bs):
            b = [pairs[i] for i in order[s:s + args.bs]]
            if len(b) < 2:
                continue
            cids = [x[0] for x in b]
            pos = [x[2] for x in b]
            neg = [sample_neg(cid, neg_rng) for cid in cids]
            dl = pos + neg
            with torch.autocast(device_type=dev, dtype=torch.float16, enabled=use_amp):
                qe = embed([qp + x[1] for x in b])
                de = embed([dp + texts[index[d]] for d in dl])
            logits = qe @ de.T * args.scale
            mask = np.zeros(logits.shape, dtype=bool)
            for i, cid in enumerate(cids):
                for j, d in enumerate(dl):
                    if j != i and d in gold[cid]:
                        mask[i, j] = True
            logits = logits.masked_fill(torch.from_numpy(mask).to(dev), -1e4)
            loss = F.cross_entropy(logits, torch.arange(len(b), device=dev))
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
        v, t, _ = evaluate_model()
        history.append({"epoch": ep, "train_loss": float(np.mean(losses)), "val_ndcg10": v, "train_ndcg10": t})
        print(f"epoch {ep}: loss {np.mean(losses):.4f} | val ndcg@10 {v:.4f} | train-subset ndcg@10 {t:.4f}")
        if v > best_v:
            best_v, best_ep, best_state = v, ep, snapshot()

    model.load_state_dict(best_state)
    print(f"best epoch = {best_ep} (val ndcg@10 {best_v:.4f})")

    model.eval()
    model.max_seq_length = 512
    with torch.no_grad(), torch.autocast(device_type=dev, dtype=torch.float16, enabled=use_amp):
        D = model.encode([dp + t for t in texts], batch_size=64, convert_to_numpy=True,
                         normalize_embeddings=True, show_progress_bar=True).astype(np.float32)
        enc = lambda cl: model.encode([qp + c["text"] for c in cl], batch_size=128, convert_to_numpy=True,  # noqa: E731
                                      normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
        SV, SE = enc(val) @ D.T, enc(ev) @ D.T
    tag = f"ft_{args.mode}"
    C.save_scores(tag, "val", SV)
    C.save_scores(tag, "eval", SE)
    run_val = C.scores_to_run(SV, [c["id"] for c in val], doc_ids)
    run_eval = C.scores_to_run(SE, [c["id"] for c in ev], doc_ids)
    m = C.save_system(f"t4_{args.mode}", run_eval, run_val, f"t4_{args.mode}.trec",
                      C.RUNS / f"val_{tag}.trec", val)
    print("final val metrics:", {k: round(x, 4) for k, x in m.items()})

    key = "val_ndcg10_random_negatives" if args.mode == "random" else "val_ndcg10_hard_negatives"
    C.update_results("task4", {"base_model": args.base, key: round(m["ndcg10"], 4)})
    (C.OUT / f"task4_{args.mode}_log.json").write_text(json.dumps(
        {"args": vars(args), "history": history, "best_epoch": best_ep,
         "zero_shot_val_ndcg10": v0, "final_val": m}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
