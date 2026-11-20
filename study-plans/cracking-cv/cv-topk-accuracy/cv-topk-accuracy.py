import torch

def topk_accuracy(logits: list, targets: list, k: int) -> float:
    """
    Returns the top-k accuracy as a float, rounded to 4 decimals.
    """
    logits = torch.tensor(logits)
    targets = torch.tensor(targets).unsqueeze(1)

    order = torch.argsort(logits, dim=1, descending=True, stable=True)
    top_k_indices = order[:, :k]

    hits = (top_k_indices == targets).any(dim=1).to(float)
    return float(hits.mean())