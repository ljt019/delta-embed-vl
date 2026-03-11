from __future__ import annotations

import torch
import torch.nn.functional as F


def last_token_pool(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    left_padded = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padded:
        return last_hidden_state[:, -1]

    seq_lengths = attention_mask.sum(dim=1) - 1
    batch_idx = torch.arange(
        last_hidden_state.shape[0],
        device=last_hidden_state.device,
    )
    return last_hidden_state[batch_idx, seq_lengths]


def normalize(embeddings: torch.Tensor) -> torch.Tensor:
    return F.normalize(embeddings, p=2, dim=-1)
