import copy
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

try:  
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: 
    pass

NAME = "Soumya Bajaj"
ROLL_NO = "25B2140"
SEED = 42
TOP_K = 100

ROOT = Path(os.environ.get("NLP_ROOT", Path(__file__).resolve().parent.parent))
DATA = ROOT / "data"
PRED = ROOT / "predictions"
CACHE = ROOT / "cache"
OUT = ROOT / "outputs"
RUNS = OUT / "runs"
for _p in (PRED, CACHE, OUT, RUNS):
    _p.mkdir(parents=True, exist_ok=True)


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception: 
        pass


def device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


_CORPUS = None


def load_corpus():
    """dict(doc_ids, titles, sents, texts, index); everything aligned to corpus order."""
    global _CORPUS
    if _CORPUS is None:
        rows = read_jsonl(DATA / "corpus.jsonl")
        doc_ids, titles, sents, texts = [], [], [], []
        for r in rows:
            ab = r.get("abstract", [])
            ab = [ab] if isinstance(ab, str) else list(ab)
            title = r.get("title", "") or ""
            doc_ids.append(int(r["doc_id"]))
            titles.append(title)
            sents.append(ab)
            texts.append((title + ". " + " ".join(ab)).strip())
        _CORPUS = dict(doc_ids=doc_ids, titles=titles, sents=sents, texts=texts,
                       index={d: i for i, d in enumerate(doc_ids)})
    return _CORPUS


def _norm_label(lab):
    lab = str(lab).upper()
    if lab.startswith("SUPPORT"):
        return "SUPPORT"
    if lab.startswith("CONTRA") or lab.startswith("REFUTE"):
        return "CONTRADICT"
    return lab


def parse_gold(claim):
    """-> {doc_id: label}. Handles the original SciFact format (dict doc_id -> [{sentences,label}])
    and a list-of-dicts format [{doc_id,label,sentences}]. Empty dict = NOT ENOUGH INFO."""
    ev = claim.get("evidence") or {}
    out = {}
    if isinstance(ev, dict):
        for did, items in ev.items():
            if items:
                out[int(did)] = _norm_label(items[0].get("label", ""))
    elif isinstance(ev, list):
        for it in ev:
            if isinstance(it, dict) and it.get("doc_id") is not None:
                out[int(it["doc_id"])] = _norm_label(it.get("label", ""))
    return out


def parse_gold_sents(claim):
    """-> {doc_id: [rationale sentence indices]} (empty if the file has no usable indices)."""
    ev = claim.get("evidence") or {}
    out = {}

    def add(did, sents):
        if isinstance(sents, list) and sents and all(isinstance(x, int) for x in sents):
            out.setdefault(int(did), set()).update(sents)

    if isinstance(ev, dict):
        for did, items in ev.items():
            for it in items or []:
                if isinstance(it, dict):
                    add(did, it.get("sentences"))
    elif isinstance(ev, list):
        for it in ev:
            if isinstance(it, dict) and it.get("doc_id") is not None:
                add(it["doc_id"], it.get("sentences"))
    return {d: sorted(v) for d, v in out.items()}


def load_claims(split):
    rows = read_jsonl(DATA / f"{split}_claims.jsonl")
    return [{"id": int(r["id"]), "text": r.get("claim") or r.get("text"), "gold": parse_gold(r),
             "gold_sents": parse_gold_sents(r)} for r in rows]

def per_claim_metrics(run, claims):
    """run: {cid: [(doc_id, score), ...] ranked}. Only claims with gold evidence are scored."""
    out = {}
    for c in claims:
        gold = set(c["gold"])
        if not gold:
            continue
        ranked = [d for d, _ in run[c["id"]]]
        dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked[:10]) if d in gold)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), 10)))
        rec = len(gold & set(ranked[:100])) / len(gold)
        rr = next((1.0 / (i + 1) for i, d in enumerate(ranked) if d in gold), 0.0)
        out[c["id"]] = {"ndcg10": dcg / idcg, "recall100": rec, "mrr": rr}
    return out


def evaluate(run, claims):
    pc = per_claim_metrics(run, claims)
    return {k: float(np.mean([m[k] for m in pc.values()])) for k in ("ndcg10", "recall100", "mrr")}


def scores_to_run(S, cids, doc_ids, k=TOP_K):
    """Score matrix (n_claims x n_docs) -> ranked top-k run. Ties broken by corpus order."""
    doc_ids = np.asarray(doc_ids)
    run = {}
    for i, cid in enumerate(cids):
        row = S[i]
        idx = np.argpartition(-row, k - 1)[:k]
        idx = idx[np.lexsort((idx, -row[idx]))]
        run[cid] = [(int(doc_ids[j]), float(row[j])) for j in idx]
    return run

def write_trec(run, path, run_name):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for cid in sorted(run):
            seen, rows = set(), []
            for d, s in run[cid]:
                if d not in seen:
                    seen.add(d)
                    rows.append((d, s))
            if len(rows) < TOP_K:
                raise ValueError(f"claim {cid}: only {len(rows)} unique docs, need {TOP_K}")
            prev = float("inf")
            for r, (d, s) in enumerate(rows[:TOP_K], 1):
                s = min(float(s), prev)  # scores must not increase with rank
                prev = s
                f.write(f"{cid} Q0 {d} {r} {s:.6f} {run_name}\n")


def read_trec(path):
    tmp = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) == 6:
                tmp.setdefault(int(p[0]), []).append((int(p[3]), int(p[2]), float(p[4])))
    return {cid: [(d, s) for _, d, s in sorted(v)] for cid, v in tmp.items()}

def save_scores(tag, split, S):
    np.save(CACHE / f"scores_{tag}_{split}.npy", np.asarray(S, dtype=np.float32))


def load_scores(tag, split):
    p = CACHE / f"scores_{tag}_{split}.npy"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found - run the earlier task first")
    return np.load(p)


def _load_json(path, default):
    path = Path(path)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return default


def _dump_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def register(name, val_ndcg, eval_run_path, val_run_path):
    reg = _load_json(OUT / "registry.json", {})
    reg[name] = {"val_ndcg10": val_ndcg, "eval_run": str(eval_run_path), "val_run": str(val_run_path)}
    _dump_json(OUT / "registry.json", reg)


def load_registry():
    return _load_json(OUT / "registry.json", {})


def save_system(name, run_eval, run_val, eval_file, val_path, val_claims):
    """Write eval run into predictions/, val run to val_path, register it. Returns val metrics."""
    ev_path = PRED / eval_file
    write_trec(run_eval, ev_path, Path(eval_file).stem)
    val_path = Path(val_path)
    write_trec(run_val, val_path, val_path.stem)
    m = evaluate(run_val, val_claims)
    register(name, m["ndcg10"], ev_path, val_path)
    return m


def set_metric(key, value):
    m = _load_json(OUT / "metrics.json", {})
    m[key] = value
    _dump_json(OUT / "metrics.json", m)


def get_metric(key, default=None):
    return _load_json(OUT / "metrics.json", {}).get(key, default)


TEMPLATE = {
    "name": "", "roll_no": "", "seed": 0, "gpu_used": "",
    "task1": {"bm25_settings": "", "val_ndcg10": None},
    "task2": {"general_model": "", "scientific_model": "", "val_ndcg10_general": None,
              "val_ndcg10_scientific": None,
              "index_time_sec": {"general": None, "scientific": None},
              "query_latency_ms": {"general": None, "scientific": None},
              "memory_mb": {"general": None, "scientific": None}},
    "task3": {"hybrid_method": "", "reranker_model": "", "rerank_top_k": None,
              "val_ndcg10_hybrid": None, "val_ndcg10_rerank": None, "rerank_query_latency_ms": None},
    "task4": {"base_model": "", "val_ndcg10_random_negatives": None, "val_ndcg10_hard_negatives": None},
    "task5": {"retriever_used": "", "nli_model": "", "zero_shot_or_finetuned": "", "val_f1": None},
    "report_q1": {"bm25_wins": [None, None], "general_wins": [None, None]},
}

def load_results():
    r = _load_json(ROOT / "results.json", None)
    if r is None:
        r = copy.deepcopy(TEMPLATE)
        r.update(name=NAME, roll_no=ROLL_NO, seed=SEED)
        try:
            import torch
            r["gpu_used"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU)"
        except Exception:  # noqa: BLE001
            r["gpu_used"] = "unknown"
    return r


def _merge(a, b):
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _merge(a[k], v)
        else:
            a[k] = v


def update_results(section, values):
    r = load_results()
    _merge(r.setdefault(section, {}), values)
    _dump_json(ROOT / "results.json", r)

def query_prefix(model_name):
    n = model_name.lower()
    if "bge" in n and "reranker" not in n:
        return "Represent this sentence for searching relevant passages: "
    if "e5" in n:
        return "query: "
    return ""


def doc_prefix(model_name):
    return "passage: " if "e5" in model_name.lower() else ""


def sync_cuda():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:  # noqa: BLE001
        pass


def now():
    sync_cuda()
    return time.perf_counter()


def resolve_model(name):
    """-> (path_or_hf_name, label_for_results). Local folders written by our fine-tuning scripts
    contain ft_meta.json with the base model name."""
    for cand in (Path(name), ROOT / name):
        if cand.is_dir():
            meta = _load_json(cand / "ft_meta.json", {})
            base = meta.get("base", cand.name)
            return str(cand), f"{base} (fine-tuned on train claims)"
    return name, name
