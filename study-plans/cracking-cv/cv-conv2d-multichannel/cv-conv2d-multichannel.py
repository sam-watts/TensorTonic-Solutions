import numpy as np

def conv2d(image: list, weight: list, bias: list, stride: int, padding: int) -> list:
    """
    Returns the channel-first convolution result as a nested list.
    """
    # c, h, w
    x = np.asarray(image, dtype=np.float64)
    # cout, cin, kh, kw
    w = np.asarray(weight, dtype=np.float64)
    # cout
    b = np.asarray(bias, dtype=np.float64)
    
    p = padding
    s = stride

    c, h, wi = x.shape
    cout, cin, kh, kw = w.shape

    hout = ((h + 2*p - kh) // s) + 1
    wiout = ((wi + 2*p - kw) // s) + 1

    y = np.zeros((cout, hout, wiout), dtype=np.float64)

    if p > 0:
        x = np.pad(x, ((0,0), (p,p), (p,p)))

    # loop over output, and i and j windows in the input
    for out in range(cout):
        for i in range(hout):
            for j in range(wiout):
                prod = w[out] * x[:, i * s: i * s + kh, j * s: j * s + kw]
                y[out, i, j] = prod.sum() + b[out]


    return [[[round(v, 4) for v in row] for row in ch] for ch in y]

    