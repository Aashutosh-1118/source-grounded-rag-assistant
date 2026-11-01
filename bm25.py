"""A small BM25 keyword index.

Used next to vector search because embeddings are weak on exact tokens such as
numbers, names and codes ("6.8", "FHA", "Q2"), while BM25 is strong on them.
Written by hand (about 40 lines) so scores stay positive on tiny corpora.
"""

import math
import re
from collections import Counter

STOPWORDS = frozenset(
    "the a an of to in and or is are was were for on what how why does do did it this that with as at by be from".split()
)


def tokenize(text: str) -> list:
    return [t for t in re.findall(r"\w+(?:[.,]\w+)*", text.lower()) if t not in STOPWORDS]


class BM25Index:
    def __init__(self, docs, k1: float = 1.5, b: float = 0.75):
        self.docs = list(docs)
        self.k1, self.b = k1, b
        self._tf, self._dl = [], []
        df = Counter()
        for d in self.docs:
            tokens = tokenize(d.page_content)
            counts = Counter(tokens)
            self._tf.append(counts)
            self._dl.append(len(tokens))
            df.update(counts.keys())
        n = len(self.docs)
        self._avgdl = (sum(self._dl) / n) if n else 0.0
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int) -> list:
        """Return up to k (document, score) pairs, best first. Zero-score docs are dropped."""
        terms = set(tokenize(query))
        scored = []
        for i, counts in enumerate(self._tf):
            score = 0.0
            for t in terms:
                f = counts.get(t)
                if not f:
                    continue
                norm = 1 - self.b + self.b * self._dl[i] / (self._avgdl or 1.0)
                score += self._idf[t] * f * (self.k1 + 1) / (f + self.k1 * norm)
            if score > 0:
                scored.append((score, i))
        scored.sort(reverse=True)
        return [(self.docs[i], s) for s, i in scored[:k]]
