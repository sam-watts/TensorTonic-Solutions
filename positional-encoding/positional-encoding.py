import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Returns a NumPy array of shape (seq_len, d_model).
    """
    pe = np.zeros((seq_len, d_model), dtype=np.float64)
    for i in range(d_model):
        a = np.arange(0, seq_len)
        a = a / (base ** (2*(i//2)/d_model))
        if i % 2 == 0:
            a = np.sin(a)
        else:
            a = np.cos(a)
            
        pe[:, i] = a

    return pe