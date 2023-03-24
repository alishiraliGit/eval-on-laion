import sys
import os
import pandas as pd
import pickle

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

from tools.wordnet_tools import lemma_is_unique, extend_with_well_known_hypernym


def main():
    # ----- Settings -----
    settings = dict()

    # Path
    settings['load_path'] = os.path.join('..', 'ilsvrc2012')
    settings['load_file_name'] = 'ILSVRC2012_synsets.txt'

    settings['save_path'] = os.path.join(settings['load_path'], 'processed')
    settings['save_file_name'] = 'ILSVRC2012_lemma2wnid.pkl'

    os.makedirs(settings['save_path'], exist_ok=True)

    # ----- Load the classes from txt file -----
    id_lemmas_df = pd.read_csv(
        os.path.join(settings['load_path'], settings['load_file_name']),
        sep=': ',
        engine='python'
    )

    # ----- Create lemma2wnids map -----
    lemma2wnid = {}

    for idx in range(id_lemmas_df.shape[0]):
        wnid = id_lemmas_df.loc[idx, 'id']

        included_lemmas = []

        for lemma in id_lemmas_df.loc[idx, 'lemmas'].split(', '):
            # Check if the lemma is unique and ignore otherwise as it probably has a general meaning.
            if not lemma_is_unique(lemma):
                continue

            if lemma in lemma2wnid:
                raise Exception(f'lemma {lemma} is already present in another sysnset!')
            else:
                lemma2wnid[lemma] = wnid
                included_lemmas.append(lemma)

        if len(included_lemmas) == 0:
            for lemma in id_lemmas_df.loc[idx, 'lemmas'].split(', '):
                lemma_ex = extend_with_well_known_hypernym(lemma, wnid)
                print(f'{lemma} extended as {lemma_ex}')
                lemma = lemma_ex

                if lemma in lemma2wnid:
                    raise Exception(f'lemma {lemma} is already present in another sysnset!')
                else:
                    lemma2wnid[lemma] = wnid

    # ----- Sanity check -----
    print('\n\nThere are %d wnids with at least one valid lemma.' % len({wnid for _, wnid in lemma2wnid.items()}))

    # ----- Save -----
    with open(os.path.join(settings['save_path'], settings['save_file_name']), 'wb') as f:
        pickle.dump(lemma2wnid, f)


if __name__ == '__main__':
    main()
