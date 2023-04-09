import pandas as pd
import numpy as np

import configs


def load_lemmas_and_wnids(synsets_path):
    id_lemmas_df = pd.read_csv(
        synsets_path,
        sep=': ',
        engine='python'
    )

    return id_lemmas_df


def get_lemmas(id_lemmas_df):
    lemmas = []
    for idx in id_lemmas_df.index:
        lemmas_txt_i = id_lemmas_df.loc[idx, configs.ILSVRCConfigs.LEMMAS_COL]
        lemmas_i = lemmas_txt_i.split(configs.ILSVRCConfigs.LEMMAS_SEP)
        lemmas.extend(lemmas_i)

    return lemmas


def lemma_is_unique(lemma, id_lemmas_df):
    lemmas = get_lemmas(id_lemmas_df)

    mask = np.array(lemmas) == lemma

    rept = np.sum(mask)

    assert rept > 0, f'Lemma {lemma} is not present in the data.'

    return rept == 1
