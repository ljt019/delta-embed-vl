from __future__ import annotations

import math
from collections.abc import Sequence

import torch

RelevantDocs = Sequence[set[int]]


def similarity_matrix(
    query_embeddings: torch.Tensor,
    corpus_embeddings: torch.Tensor,
) -> torch.Tensor:
    return query_embeddings @ corpus_embeddings.T


def mean_recall_at_k(
    query_embeddings: torch.Tensor,
    corpus_embeddings: torch.Tensor,
    relevant_docs: RelevantDocs,
    *,
    k: int,
) -> float:
    top_indices = top_k_indices(query_embeddings, corpus_embeddings, k=k)
    hits = 0.0
    counted = 0
    for row_indices, relevant in zip(top_indices, relevant_docs, strict=True):
        if not relevant:
            continue
        counted += 1
        if any(index in relevant for index in row_indices):
            hits += 1.0
    return hits / max(counted, 1)


def mean_ndcg_at_k(
    query_embeddings: torch.Tensor,
    corpus_embeddings: torch.Tensor,
    relevant_docs: RelevantDocs,
    *,
    k: int,
) -> float:
    top_indices = top_k_indices(query_embeddings, corpus_embeddings, k=k)
    total = 0.0
    counted = 0
    for row_indices, relevant in zip(top_indices, relevant_docs, strict=True):
        if not relevant:
            continue
        counted += 1
        dcg = 0.0
        for rank, index in enumerate(row_indices, start=1):
            if index in relevant:
                dcg += 1.0 / math.log2(rank + 1)
        ideal_hits = min(k, len(relevant))
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        total += dcg / idcg
    return total / max(counted, 1)


def top_k_indices(
    query_embeddings: torch.Tensor,
    corpus_embeddings: torch.Tensor,
    *,
    k: int,
    query_batch_size: int = 256,
) -> list[list[int]]:
    if len(corpus_embeddings) == 0 or len(query_embeddings) == 0:
        return [[] for _ in range(len(query_embeddings))]

    resolved_k = min(k, len(corpus_embeddings))
    query_embeddings = query_embeddings.float()
    corpus_embeddings = corpus_embeddings.float()
    all_indices: list[list[int]] = []
    for start in range(0, len(query_embeddings), query_batch_size):
        stop = start + query_batch_size
        scores = similarity_matrix(query_embeddings[start:stop], corpus_embeddings)
        indices = torch.topk(scores, k=resolved_k, dim=1).indices.cpu().tolist()
        all_indices.extend(indices)
    return all_indices
