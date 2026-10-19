import argparse
import json
import random
import numpy as np
import common as C
import verify_lib as V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
    ap.add_argument("--sent_model", default=None)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--nsent", type=int, default=3)
    ap.add_argument("--neg_per_claim", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--out", default=str(C.CACHE / "nli_ft"))
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--no_amp", action="store_true", help="disable fp16 autocast (use if the loss becomes NaN)")
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics import f1_score
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    C.seed_everything(args.seed)
    dev = C.device()
    use_amp = dev == "cuda" and not args.no_amp
    corpus = C.load_corpus()
    doc_ids, index, sents = corpus["doc_ids"], corpus["index"], corpus["sents"]
    train, val = C.load_claims("train"), C.load_claims("val")
    ids = [c["id"] for c in train + val]
    assert len(ids) == len(set(ids)), "claim ids overlap between train and val"
    rng = random.Random(args.seed)

    Sg = {"train": C.load_scores("general", "train"), "val": C.load_scores("general", "val")}

    def make_plan(claims, S):
        plan = []
        for i, c in enumerate(claims):
            order = np.argsort(-S[i])[:10]
            negs = [doc_ids[j] for j in order if doc_ids[j] not in c["gold"]]
            plan.append((c, rng.sample(negs, min(args.neg_per_claim, len(negs)))))
        return plan

    tr_plan, va_plan = make_plan(train, Sg["train"]), make_plan(val, Sg["val"])
    pairs = [(c["id"], d) for c, negs in tr_plan + va_plan for d in list(c["gold"]) + negs if d in index]

    res = C.load_results()
    sent_name = args.sent_model or res["task2"].get("general_model") or "BAAI/bge-base-en-v1.5"
    sm = SentenceTransformer(sent_name, device=dev)
    sel = V.select_sentences(sm, C.query_prefix(sent_name), C.doc_prefix(sent_name), corpus,
                             train + val, pairs, args.nsent)
    del sm
    if dev == "cuda":
        torch.cuda.empty_cache()

    # ---- model + label ids
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(args.base).float().to(dev)
    id2l = {int(i): l.lower() for i, l in model.config.id2label.items()}
    label_ids = {"SUPPORT": next(i for i, l in id2l.items() if l.startswith("entail")),
                 "CONTRADICT": next(i for i, l in id2l.items() if l.startswith("contra")),
                 "NEUTRAL": next(i for i, l in id2l.items() if l.startswith("neutral"))}

    def examples(plan):
        ex = []
        for c, negs in plan:
            for did, lab in c["gold"].items():
                if did not in index or lab not in label_ids:
                    continue
                ss = sents[index[did]]
                rat = [j for j in c.get("gold_sents", {}).get(did, []) if 0 <= j < len(ss)]
                top_idx, top_txt = sel[(c["id"], did)]
                lid = label_ids[lab]
                if rat:
                    ex.append((" ".join(ss[j] for j in rat), c["text"], lid))
                    if set(top_idx) & set(rat):
                        ex.append((" ".join(top_txt), c["text"], lid))
                elif top_txt:  # no sentence indices in the data: fall back to retrieved sentences
                    ex.append((" ".join(top_txt), c["text"], lid))
            for did in negs:
                _, top_txt = sel[(c["id"], did)]
                if top_txt:
                    ex.append((" ".join(top_txt), c["text"], label_ids["NEUTRAL"]))
        return ex

    tr, va = examples(tr_plan), examples(va_plan)
    cnt = lambda ex: {id2l[k]: sum(e[2] == k for e in ex) for k in label_ids.values()}  # noqa: E731
    print(f"train examples {len(tr)} {cnt(tr)} | val examples {len(va)} {cnt(va)}")
    if not tr:
        raise SystemExit("no training examples were built - check the evidence format in the claims files")

    def encode(batch):
        return tok([x[0] for x in batch], [x[1] for x in batch], truncation="only_first",
                   max_length=args.max_len, padding=True, return_tensors="pt").to(dev)

    def val_macro_f1():
        model.eval()
        pred = np.zeros(len(va), dtype=int)
        order = np.argsort([len(e[0]) for e in va])
        with torch.no_grad():
            for s in range(0, len(order), 64):
                idx = order[s:s + 64]
                with torch.autocast(device_type=dev, dtype=torch.float16, enabled=use_amp):
                    logits = model(**encode([va[i] for i in idx])).logits
                pred[idx] = logits.argmax(-1).cpu().numpy()
        model.train()
        return float(f1_score([e[2] for e in va], pred, average="macro"))

    def snapshot():
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.train()
    v0 = val_macro_f1()
    print(f"epoch 0 (pretrained): val macro-F1 {v0:.4f}")
    best_f1, best_ep, best_state = v0, 0, snapshot()
    history = [{"epoch": 0, "train_loss": None, "val_macro_f1": v0}]

    steps = int(np.ceil(len(tr) / args.bs)) * args.epochs
    warm = max(1, int(0.1 * steps))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (steps - s) / max(1, steps - warm)))
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for ep in range(1, args.epochs + 1):
        perm = np.random.RandomState(args.seed + ep).permutation(len(tr))
        losses = []
        for s in range(0, len(perm), args.bs):
            batch = [tr[i] for i in perm[s:s + args.bs]]
            labels = torch.tensor([x[2] for x in batch], device=dev)
            with torch.autocast(device_type=dev, dtype=torch.float16, enabled=use_amp):
                logits = model(**encode(batch)).logits
            loss = F.cross_entropy(logits.float(), labels)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
        v = val_macro_f1()
        history.append({"epoch": ep, "train_loss": float(np.mean(losses)), "val_macro_f1": v})
        print(f"epoch {ep}: loss {np.mean(losses):.4f} | val macro-F1 {v:.4f}")
        if v > best_f1:
            best_f1, best_ep, best_state = v, ep, snapshot()

    model.load_state_dict(best_state)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    with open(f"{args.out}/ft_meta.json", "w", encoding="utf-8") as f:
        json.dump({"base": args.base, "best_epoch": best_ep, "val_macro_f1": best_f1,
                   "history": history, "args": vars(args)}, f, indent=2)
    print(f"saved best epoch {best_ep} (val macro-F1 {best_f1:.4f}) to {args.out}")
    print("next: python code/task5_verify.py --nli cache/nli_ft --only_if_better")


if __name__ == "__main__":
    main()
