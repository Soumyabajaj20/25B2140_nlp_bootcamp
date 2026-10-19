# NLP Bootcamp

Soumya Bajaj, Roll no 25B2140.

## Setup

```
pip install numpy scipy scikit-learn nltk psutil torch sentence-transformers transformers
```

Run every command from the project root:

```
<root>/data/          corpus.jsonl, train_claims.jsonl, val_claims.jsonl, eval_claims.jsonl
<root>/check_format.py
<root>/code/          these scripts
```

Scripts create `predictions/`, `cache/`, `outputs/` and `results.json` automatically. Run the tasks in the order below; each one reads the cached scores of the earlier ones. Models download from the Hugging Face Hub on first use (an `HF_TOKEN` is optional and only raises rate limits).

## Commands (in the order used for the submission)

| Step | Command | What it does |
|------|---------|--------------|
| Task 1 | `python code/task1_bm25.py` | corpus/claim exploration, BM25 grid search (k1, b, stemming, stopwords) on validation; writes `t1_bm25.trec`, `val_bm25.trec`. Best: k1=1.2, b=0.3, stemming, stopwords removed |
| Task 2 | `python code/task2_dense.py` | `BAAI/bge-base-en-v1.5` (general) and `pritamdeka/S-PubMedBert-MS-MARCO` (scientific); index time, query latency, GPU memory into `results.json`; writes `t2_general.trec`, `t2_scientific.trec`, `val_general.trec` |
| Task 3 | `python code/task3_hybrid_rerank.py` | RRF and score-interpolation hybrids tuned on validation (best: min-max interpolation, BM25 0.3 + general 0.7); reranks the top-k of the best first stage with `cross-encoder/ms-marco-MiniLM-L-12-v2`, k chosen from 10/20/30/50; writes `t3_hybrid.trec`, `t3_rerank.trec` |
| Task 3b | `python code/task3b_finetune_ce.py` then `python code/task3_hybrid_rerank.py --reranker cache/ce_ft --ks 10 20 30 50 --only_if_better` | fine-tunes the cross-encoder on the training claims (1 gold + 7 mined hard negatives per group, listwise loss, epoch picked on validation) and reranks with it. The submitted `t3_rerank.trec` is from this model (k=10) |
| Task 4 | `python code/task4_finetune.py --mode random` then `python code/task4_finetune.py --mode hard` | same base model (bge-base), hyper-parameters, 4 epochs and seed; random vs hard negatives (mined from BM25 + both dense systems on train); writes `t4_random.trec`, `t4_hard.trec` |
| Task 5 | `python code/task5_verify.py` | best retriever by validation nDCG@10 (here `t4_hard`), top-3 sentence selection, zero-shot NLI `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, thresholds tuned on validation, heuristic error tagging; writes `t5_verification.csv`, `t5_val_verification.csv`, `t5_val_errors.csv` |
| Task 5b | `python code/task5b_finetune_nli.py` then `python code/task5_verify.py --nli cache/nli_ft --only_if_better` | fine-tunes the NLI classifier on the training claims (gold rationale -> SUPPORT/CONTRADICT, retrieved non-gold sentences -> NEUTRAL). The submitted verification files are from this model |
| final | `python code/finalize.py` | picks the Q1 claim ids, builds `25B2140_nlp_bootcamp.zip`, runs `check_format.py` |
| report | `python code/fill_q4.py` | writes the Q4 counts from `outputs/task5_stats.json` into `report.md` |

Useful options: `task2_dense.py --general <hf-name> --scientific <hf-name> --only general|scientific`,
`task3_hybrid_rerank.py --reranker BAAI/bge-reranker-base --ks 10 20 50`,
`task4_finetune.py --base <hf-name> --epochs 4 --bs 16 --hn_skip 0 --hn_topk 30`,
`task5_verify.py --nli <hf-name> --ncand 5`.

## Final validation results

| System | nDCG@10 |
|--------|---------|
| BM25 | 0.8664 |
| bge-base (general) | 0.8910 |
| S-PubMedBert (scientific) | 0.8438 |
| Hybrid (BM25 + general) | 0.9136 |
| Hybrid + fine-tuned cross-encoder, top-10 | 0.9061 |
| bge-base fine-tuned, random negatives | 0.9013 |
| bge-base fine-tuned, hard negatives | 0.9139 |

Task 5 (triple-level F1, retriever `t4_hard`): 0.6121 zero-shot, 0.7200 with the fine-tuned NLI model (precision 0.802, recall 0.653).

## Notes

- All tuning and model selection uses only the training and validation claims. Eval claims are only ever scored by the final models; they have no labels.
- Latency numbers are single-query (batch size 1) means over 100 validation claims (50 for the reranker). `memory_mb` is peak GPU memory while indexing the corpus.
- `--only_if_better` keeps the previously stored predictions/results unless the new run has a higher validation score, so an optional step can never make the submission worse. A message such as "Not better than the stored result" means the stored result was kept.
- Fine-tuning scripts load the model with `.float()` (fp32 master weights) and train under fp16 autocast with a `GradScaler`. Recent transformers versions load some checkpoints (e.g. the DeBERTa NLI model) in fp16, which otherwise raises "Attempting to unscale FP16 gradients".
- In `task3b_finetune_ce.py` the validation nDCG printed during training uses max_len 384 under autocast, so it differs slightly from the numbers `task3_hybrid_rerank.py` reports; the latter are the ones in `results.json`.
- Error categories in `t5_val_errors.csv` come from heuristics in `task5_verify.py` (retrieval_miss = no gold abstract among the N candidates used; N=1 in the submission, so this is the same as "the top-1 abstract is not gold", as in `report.md` Q4).
- Task 5 uses `t4_hard` (val nDCG@10 0.9139) rather than the hybrid (0.9136). The margin is only 0.0003, but the rule is the highest validation nDCG@10, so `t4_hard` is the selected best retriever.
- No generative models are used; every model is an encoder under 400M parameters.
