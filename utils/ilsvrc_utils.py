import pandas as pd


def load_lemmas_and_wnids(synsets_path):
    id_lemmas_df = pd.read_csv(
        synsets_path,
        sep=': ',
        engine='python'
    )

    return id_lemmas_df
