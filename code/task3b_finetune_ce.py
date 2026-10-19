import argparse
import json
import random
import numpy as np
import common as C
from task3_hybrid_rerank import rank_matrix, rerank_run

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="cross-encoder/ms-marco-MiniLM-L-12-v2")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=4, help="groups per step (pairs per step = bs * group)")
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=384)
    ap.add_argument("--eval_k", type=int, default=20)
    ap.add_argument("--mine_top", type=int, default=30)
    ap.add_argument("--out", default=str(C.CACHE / "ce_ft"))
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--no_amp", action="store_true")
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    C.seed_everything(args.seed)
    dev = C.device()
    use_amp = dev == "cuda" and not args.no_amp
    corpus = C.load_corpus()
    doc_ids, texts, index = corpus["doc_ids"], corpus["texts"], corpus["index"]
    train, val = C.load_claims("train"), C.load_claims("val")

    fused = sum(1.0 / (60 + rank_matrix(C.load_scores(t, "train"))) for t in ("bm25", "general", "scientific"))
    pool = {}
    for i, c in enumerate(train):
        top = np.argsort(-fused[i])[:args.mine_top]
        pool[c["id"]] = [doc_ids[j] for j in top if doc_ids[j] not in c["gold"]]
    pos = [(c, d) for c in train for d in c["gold"] if d in index]
    print(f"{len(pos)} training groups (1 gold + {args.group - 1} hard negatives each)")

    HV = C.load_scores("hybrid", "val")
    first = C.scores_to_run(HV, [c["id"] for c in val], doc_ids)
    base_nd = C.evaluate(first, val)["ndcg10"]

    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(args.base, num_labels=1).float().to(dev)
    def score_pairs(batch):
        enc = tok([q for q, _ in batch], [d for _, d in batch], truncation="only_second",
                  max_length=args.max_len, padding=True, return_tensors="pt").to(dev)
        with torch.autocast(device_type=dev, dtype=torch.float16, enabled=use_amp):
            return model(**enc).logits[:, 0].float()

    def val_ndcg():
        model.eval()
        pairs, owner = [], []
        for c in val:
            for d, _ in first[c["id"]][:args.eval_k]:
                pairs.append((c["text"], texts[index[d]]))
                owner.append(c["id"])
        out = np.zeros(len(pairs))
        order = np.argsort([len(p[1]) for p in pairs])
        with torch.no_grad():
            for s in range(0, len(order), 64):
                idx = order[s:s + 64]
                out[idx] = score_pairs([pairs[i] for i in idx]).cpu().numpy()
        ce = {c["id"]: [] for c in val}
        for cid, sc in zip(owner, out):
            ce[cid].append(float(sc))
        model.train()
        return C.evaluate(rerank_run(first, ce, args.eval_k), val)["ndcg10"]

    def snapshot():
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.train()
    v0 = val_ndcg()
    print(f"hybrid first stage val ndcg@10 {base_nd:.4f}")
    print(f"epoch 0 (pretrained CE, rerank top-{args.eval_k}): val ndcg@10 {v0:.4f}")
    best_v, best_ep, best_state = v0, 0, snapshot()
    history = [{"epoch": 0, "train_loss": None, "val_ndcg10": v0}]

    steps = int(np.ceil(len(pos) / args.bs)) * args.epochs
    warm = max(1, int(0.1 * steps))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (steps - s) / max(1, steps - warm)))
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for ep in range(1, args.epochs + 1):
        rng = random.Random(args.seed * 100 + ep)
        perm = np.random.RandomState(args.seed + ep).permutation(len(pos))
        losses = []
        for s in range(0, len(perm), args.bs):
            batch = [pos[i] for i in perm[s:s + args.bs]]
            pairs = []
            for c, d in batch:
                negs = pool[c["id"]]
                chosen = rng.sample(negs, min(args.group - 1, len(negs)))
                while len(chosen) < args.group - 1:  # tiny pools: pad with random non-gold docs
                    r = rng.choice(doc_ids)
                    if r not in c["gold"]:
                        chosen.append(r)
                for dd in [d] + chosen:             # position 0 is the positive
                    pairs.append((c["text"], texts[index[dd]]))
            logits = score_pairs(pairs).view(len(batch), args.group)
            loss = F.cross_entropy(logits, torch.zeros(len(batch), dtype=torch.long, device=dev))
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
        v = val_ndcg()
        history.append({"epoch": ep, "train_loss": float(np.mean(losses)), "val_ndcg10": v})
        print(f"epoch {ep}: loss {np.mean(losses):.4f} | val ndcg@10 after rerank {v:.4f}")
        if v > best_v:
            best_v, best_ep, best_state = v, ep, snapshot()

    model.load_state_dict(best_state)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    with open(f"{args.out}/ft_meta.json", "w", encoding="utf-8") as f:
        json.dump({"base": args.base, "best_epoch": best_ep, "val_ndcg10_rerank": best_v,
                   "hybrid_val_ndcg10": base_nd, "history": history, "args": vars(args)}, f, indent=2)
    print(f"saved best epoch {best_ep} (val ndcg@10 {best_v:.4f} vs hybrid {base_nd:.4f}) to {args.out}")
    print("next: python code/task3_hybrid_rerank.py --reranker cache/ce_ft --ks 10 20 30 50 --only_if_better")


if __name__ == "__main__":
    main()
