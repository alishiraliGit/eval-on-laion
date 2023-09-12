import numpy as np
from sklearn.preprocessing import normalize


def find_inverse_map(maps):
    if isinstance(maps, dict):
        maps = [maps]

    im = {}
    for m in maps:
        for k, values in m.items():
            if not isinstance(values, list):
                values = [values]

            for v in values:
                if v not in im:
                    im[v] = []
                im[v].append(k)

    return im


def drop_keys_with_multiple_values(m):
    drop_keys = []
    for k, values in m.items():
        if len(values) > 1:
            drop_keys.append(k)

    for drop_k in drop_keys:
        m.pop(drop_k)

    return drop_keys


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_norm = normalize(x, axis=1, norm='l2')
    y_norm = normalize(y, axis=1, norm='l2')

    sims = np.sum(x_norm * y_norm, axis=1)

    return sims


if __name__ == '__main__':
    d = {'a': [1, 2, 3], 'b': 3}

    print(find_inverse_map(d))
