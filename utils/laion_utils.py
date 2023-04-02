import os
import string
import urllib.request
import numpy as np

import configs


def get_laion_part_file_name(part):
    return 'part-%05d-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet' % part


def get_laion_subset_file_name(from_part, to_part):
    return 'part-%05d-to-part%05d-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet' % (from_part, to_part)


def download_laion_part(part, laion_path):
    url_base = configs.LAIONConfig.URL_BASE
    urllib.request.urlretrieve(
        url=url_base + get_laion_part_file_name(part),
        filename=os.path.join(laion_path, get_laion_part_file_name(part))
    )


def map_index(idx, part):
    return idx + part*configs.LAIONConfig.INDEX_SHIFT_PER_PART


def rename_index(df, part):
    return df.rename(mapper=lambda idx: map_index(idx, part))


translation = str.maketrans(string.punctuation, ' ' * len(string.punctuation))


def transform_lemma(lemma):
    # Extend the lemma to make sure 'paid' is not a kind of 'ai'!
    return ' ' + lemma + ' '


def transform_text(txt):
    return ' ' + txt.translate(translation) + ' '


def icdf_bins(df):
    # Find CDF
    sim_rng = configs.LAIONConfig.SIMILARITY_BINS_RANGE
    xs = np.linspace(sim_rng[0], sim_rng[1], 1000)
    cdfs = np.zeros(xs.shape)
    for i_x, x in enumerate(xs):
        cdfs[i_x] = np.mean(df['similarity'] <= x)

    # Find ICDF
    dp = configs.LAIONConfig.SIMILARITY_BINS_DELTA_P
    i_x = 0
    cdf_0 = cdfs[0]
    icdfs = [xs[0]]
    while True:
        if i_x == len(xs):
            icdfs.append(xs[i_x - 1])
            break

        if cdfs[i_x] - cdf_0 >= dp:
            icdfs.append(xs[i_x])
            cdf_0 = cdfs[i_x]

        i_x += 1

    return np.array(icdfs)
