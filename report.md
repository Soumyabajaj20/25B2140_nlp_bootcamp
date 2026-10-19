# Report - 25B2140 (all analysis on validation claims)

## Q1. BM25 vs t2_general

BM25 wins on claims 673 (nDCG@10 1.00 vs 0.36) and 381 (1.00 vs 0.50). Their terms overlap strongly with the gold abstract (100%, 67%): "LDL-cholesterol", "breast cancer", "environmental". Exact matching ranks it first, while the dense model likely preferred topically related abstracts. t2_general wins on 766 (1.00 vs 0.00) and 645 (1.00 vs 0.36). Overlap is low (50%, 30%): "medications ... unwanted side effects" matches a paper on obesity pharmacotherapy, and "folic acid and VB6" matches "folate and vitamin B6". Paraphrases and synonyms defeat lexical matching.

## Q2. Largest Task 3 gain and its cost

The hybrid (min-max interpolation, BM25 0.3 + bge 0.7) gave the largest gain: nDCG@10 0.9136 vs 0.8910 for the best single system (+0.023), because BM25 and bge fail on different claims. Cost: one extra BM25 lookup, roughly +5 ms per query on top of ~16 ms dense. Reranking the top-10 with the fine-tuned MiniLM cross-encoder lowered nDCG@10 to 0.9061 (-0.0075) and added ~113 ms per query. The off-the-shelf MS-MARCO cross-encoder was worse still (0.887).

## Q3. Hard negatives and overfitting

Yes, modestly. With identical base model (bge-base), settings and seed, hard negatives reached 0.9139 vs 0.9013 for random negatives (zero-shot 0.8910). This is one seed and 106 evidence claims, so the evidence is limited. Overfitting: nDCG@10 on a train-claim subset rose to 0.971 while validation reached 0.914. The train-validation gap widened from 0.032 (zero-shot) to 0.057 as training loss fell to 0.058. Validation never dropped materially, so I kept epoch 4 but would not train longer.

## Q4. Fraction of retrieval_miss

14 of 46 validation errors (30%) are retrieval_miss: no gold abstract is among the N=1 candidate(s) used, i.e. the top-1 abstract is not gold, even though Recall@100 is 0.99, so the right abstract is retrieved but ranked too low. The remaining errors are mainly other (NEI claims given a label, or the wrong label on the right abstract), needs_multiple_docs (the pipeline outputs one document per claim), numerical and negation. Categories are assigned by heuristics, so the split is approximate.

## Q5. Next change

Verify the top-5 candidates but rank them by a joint score (retrieval score plus NLI confidence) instead of the current N=1, one-document rule. Validation tuning chose N=1 because extra candidates added false positives. Yet many claims are retrieval_misses at rank 1 despite 0.99 Recall@100, and some need more than one document. Letting NLI evidence re-order candidates should recover both, at the price of a few more NLI calls per claim.
