import torch.nn.functional as F
from torch import Tensor


def cosine_distill_loss(student_emb: Tensor, teacher_emb: Tensor) -> Tensor:
    """Cosine embedding distillation loss.

    Minimizes 1 - cos_sim(student, teacher), pushing student embeddings
    to align with teacher embeddings in direction.

    Both inputs should be L2-normalized for best stability, but this
    function normalizes defensively anyway.
    """
    student_norm = F.normalize(student_emb, p=2, dim=-1)
    teacher_norm = F.normalize(teacher_emb, p=2, dim=-1)
    return (1.0 - (student_norm * teacher_norm).sum(dim=-1)).mean()
