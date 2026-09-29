"""Shared helpers: dense (MiniLM), lexical sparse (bag-of-words / TF-IDF) and learned sparse (SpladeX) vectors."""
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

# Offline mode is opt-in, not forced: set HF_HUB_OFFLINE=1 yourself before running if
# you know the venue has no wifi and the models are already cached locally. Left unset,
# this also works on a fresh cloud deploy (Streamlit Cloud / HF Spaces) where the models
# have to be downloaded from Hugging Face on first run.

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForMaskedLM, AutoTokenizer

DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SPLADE_MODEL = "Arvind0101/static-query-splade-code-docs"

STOPWORDS = set(
    "a an the of to in on for and or is are be by with how do i my from into that this it as at can "
    "what which want need".split()
)

_cache = {}


def dense_model():
    if "dense" not in _cache:
        from sentence_transformers import SentenceTransformer

        _cache["dense"] = SentenceTransformer(DENSE_MODEL, device="cpu")
    return _cache["dense"]


def splade_parts():
    if "splade" not in _cache:
        tok = AutoTokenizer.from_pretrained(SPLADE_MODEL)
        model = AutoModelForMaskedLM.from_pretrained(SPLADE_MODEL).eval()
        weights = torch.load(hf_hub_download(SPLADE_MODEL, "static_query_weights.pt"), map_location="cpu")
        if isinstance(weights, dict):
            weights = next(iter(weights.values()))
        weights = torch.as_tensor(weights).float().flatten()
        if weights.min() < 0:  # stored as logits; the model applies softplus
            weights = torch.nn.functional.softplus(weights)
        _cache["splade"] = (tok, model, weights)
    return _cache["splade"]


def dense_vector(text):
    return dense_model().encode(text, normalize_embeddings=True)


def words(text):
    return [w for w in re.findall(r"[a-z0-9_]+", text.lower()) if w not in STOPWORDS]


def lexical_vector(text):
    """Classic sparse bag-of-words: only words that literally appear get a weight."""
    return dict(Counter(words(text)))


# Repo-relative bundled file (term -> document-frequency count only, ~540KB) --
# extracted once from the real shipped model_semantic_index.json's bm25_index so this
# ships with the app instead of depending on a 73MB local build output. See
# extract_bm25_stats.py to regenerate it if the NumPy docs get rebuilt.
NUMPY_BM25_STATS_PATH = Path(__file__).parent / "numpy_bm25_stats.json"


def numpy_bm25_stats():
    """Real document-frequency stats from the actual shipped NumPy semantic index."""
    if "numpy_bm25" not in _cache:
        with open(NUMPY_BM25_STATS_PATH, encoding="utf-8") as f:
            stats = json.load(f)
        _cache["numpy_bm25"] = (stats["doc_freq"], stats["num_documents"], stats["num_bm25_terms"])
    return _cache["numpy_bm25"]


def numpy_corpus_sparse_vector(text):
    """Classic sparse (TF x IDF), but IDF computed from the REAL ~6,600-doc NumPy corpus:
    idf = log(1 + (N - df + 0.5) / (df + 0.5)), same formula the shipped BM25 index uses.
    Words never seen in NumPy's docs get df=0 -> maximum, 'unknown to this corpus' weight."""
    doc_freq, num_docs, vocab_size = numpy_bm25_stats()
    counts = Counter(words(text))
    out = {}
    for w, tf in counts.items():
        df = doc_freq.get(w, 0)
        idf = math.log(1 + (num_docs - df + 0.5) / (df + 0.5))
        out[w] = tf * idf
    return out, vocab_size


def splade_doc_vector(text, top_k=40):
    """Learned sparse: the model predicts weights over the whole ~30k vocabulary, incl. words NOT in the text."""
    tok, model, _ = splade_parts()
    enc = tok(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        logits = model(**enc).logits[0]
    act = torch.log1p(torch.relu(logits)) * enc["attention_mask"][0].unsqueeze(-1)
    vec = act.max(dim=0).values
    nz = int((vec > 0).sum())
    vals, ids = vec.topk(min(top_k, nz)) if nz else (torch.tensor([]), torch.tensor([], dtype=torch.long))
    terms = {tok.convert_ids_to_tokens(int(i)): float(v) for v, i in zip(vals, ids)}
    specials = set(tok.all_special_tokens)
    return {t: w for t, w in terms.items() if t not in specials}, nz, vec.numel()


def splade_query_vector(text):
    """SpladeX's inference-free query side: tokenise + look up a learned weight per token. No neural network."""
    tok, _, weights = splade_parts()
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return {tok.convert_ids_to_tokens(i): float(weights[i]) for i in dict.fromkeys(ids)}


def sparse_dot(a, b):
    return sum(w * b[t] for t, w in a.items() if t in b)


def sparse_cosine(a, b):
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return sparse_dot(a, b) / (na * nb) if na and nb else 0.0


def dense_cosine(a, b):
    return float(np.dot(a, b))


if __name__ == "__main__":
    q = "remove missing values"
    d = "numpy.nan_to_num: Replace NaN with zero and infinity with large finite numbers."
    print("lexical q:", lexical_vector(q))
    print("lexical d:", lexical_vector(d))
    print("lexical cosine:", round(sparse_cosine(lexical_vector(q), lexical_vector(d)), 3))
    dq, dd = dense_vector(q), dense_vector(d)
    print("dense dims:", dq.shape, "first 8:", np.round(dq[:8], 3), "cosine:", round(dense_cosine(dq, dd), 3))
    sq = splade_query_vector(q)
    sd, nz, total = splade_doc_vector(d)
    print("splade query (static weights):", {k: round(v, 2) for k, v in sq.items()})
    print(f"splade doc: {nz}/{total} active dims; top:", {k: round(v, 2) for k, v in list(sd.items())[:15]})
    full_doc, _, _ = splade_doc_vector(d, top_k=10_000)
    print("splade score (query . doc):", round(sparse_dot(sq, full_doc), 3))
