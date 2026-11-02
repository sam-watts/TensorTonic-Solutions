import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    if len(seqs) == 0:
        return np.asarray([], dtype=int).reshape(0, 0)
    
    if max_len is None:
        max_len = max(len(seq) for seq in seqs)
    
    out_array = np.zeros((len(seqs), max_len), dtype=int)

    for i, s in enumerate(seqs):
        len_array = len(s)
        out_array[i, :] = s[:max_len] + [pad_value] * (max_len - len_array)

    return out_array