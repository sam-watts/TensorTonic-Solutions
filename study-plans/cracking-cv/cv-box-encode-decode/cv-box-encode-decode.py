import numpy as np

def encode_and_decode_boxes(proposals: list, gt_boxes: list) -> dict:
    """
    Returns deltas and decoded, each an N-by-4 nested list, in a dictionary.
    """
    p = np.asarray(proposals, dtype=np.float64)
    g = np.asarray(gt_boxes, dtype=np.float64)
    
    wp = p[:, 2] - p[:, 0]
    hp = p[:, 3] - p[:, 1]

    wg = g[:, 2] - g[:, 0]
    hg = g[:, 3] - g[:, 1]

    cxp = p[:, 2] - wp / 2
    cyp = p[:, 3] - hp / 2
    cxg = g[:, 2] - wg / 2
    cyg = g[:, 3] - hg / 2

    tx = (cxg - cxp) / wp
    ty = (cyg - cyp) / hp
    tw = np.log(wg / wp)
    th = np.log(hg / hp)

    cdx = cxp + tx * wp
    cdy = cyp + ty * hp
    wd = wp * np.exp(tw)
    hd = hp * np.exp(th)

    decoded = np.round(np.dstack([
        cdx - wd / 2, cdy - hd / 2, cdx + wd / 2, cdy + hd / 2
    ]), 4).squeeze(0).tolist()
    
    deltas = np.round(np.dstack([tx, ty, tw, th]), 4).squeeze(0).tolist()
    return dict(deltas=deltas, decoded=decoded)
    