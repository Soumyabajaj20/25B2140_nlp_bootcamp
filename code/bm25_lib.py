import re
from functools import lru_cache
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

_TOKEN = re.compile(r"\w+")


@lru_cache(maxsize=None)
def _stem(word):
    return _STEMMER.stem(word)


try:
    from nltk.stem import PorterStemmer
    _STEMMER = PorterStemmer()
except Exception:  # noqa: BLE001
    _STEMMER = None


class BM25Index:
    def __init__(self, texts, stem=False, stop=False):
        if stem and _STEMMER is None:
            raise RuntimeError("pip install nltk  (needed for stemming)")
        self.stem, self.stop = stem, stop
        self.vec = CountVectorizer(analyzer=self._analyze, dtype=np.float32)
        self.tf = self.vec.fit_transform(texts).tocsr()
        n = self.tf.shape[0]
        self.dl = np.asarray(self.tf.sum(1)).ravel()
        self.avgdl = float(self.dl.mean())
        df = np.asarray((self.tf > 0).sum(0)).ravel()
        self.idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        self.rows = np.repeat(np.arange(n), np.diff(self.tf.indptr))

    def _analyze(self, text):
        toks = _TOKEN.findall(text.lower())
        if self.stop:
            toks = [t for t in toks if t not in ENGLISH_STOP_WORDS]
        if self.stem:
            toks = [_stem(t) for t in toks]
        return toks

    def doc_weights(self, k1, b):
        W = self.tf.copy()
        tf = self.tf.data
        norm = k1 * (1.0 - b + b * self.dl[self.rows] / self.avgdl)
        W.data = (tf * (k1 + 1.0) / (tf + norm) * self.idf[self.tf.indices]).astype(np.float32)
        return W

    def score(self, queries, k1=1.2, b=0.75, W=None):
        """-> dense float32 matrix (n_queries x n_docs)."""
        if W is None:
            W = self.doc_weights(k1, b)
        Q = self.vec.transform(queries).tocsr()
        Q.data[:] = 1.0  # binary query terms
        return (Q @ W.T).toarray().astype(np.float32)
