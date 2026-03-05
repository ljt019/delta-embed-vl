import torch
import torch.nn.functional as F
from torch import Tensor


def last_token_pool(last_hidden_state: Tensor, attention_mask: Tensor) -> Tensor:
    """
    Extract the embedding from the last non-padding token.

    Works with both left-padded (vLLM/generation style) and
    right-padded (standard HF tokenizer) inputs.
    """
    left_padded = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padded:
        return last_hidden_state[:, -1]

    seq_lengths = attention_mask.sum(dim=1) - 1
    batch_idx = torch.arange(
        last_hidden_state.shape[0], device=last_hidden_state.device
    )
    return last_hidden_state[batch_idx, seq_lengths]


def normalize(embeddings: Tensor) -> Tensor:
    """
    L2-normalize embeddings to unit length.
    """
    return F.normalize(embeddings, p=2, dim=-1)
