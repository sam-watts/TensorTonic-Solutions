import numpy as np

def adam_step(
    param: list,
    grad: list,
    m: list,
    v: list,
    t: int,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (param_new, m_new, v_new) as NumPy arrays.
    """
    mt = beta1 * np.asarray(m) + (1 - beta1) * np.asarray(grad)
    vt = beta2 * np.asarray(v) + (1 - beta2) * np.asarray(grad) ** 2
    mt_hat = mt / (1 - beta1 ** t)
    vt_hat = vt / (1 - beta2 ** t)

    w = np.asarray(param) - lr * mt_hat / (np.sqrt(vt_hat) + eps)
    return w, mt, vt