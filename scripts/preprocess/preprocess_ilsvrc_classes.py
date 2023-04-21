import sys
import os
import pickle

from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..'))

from utils import logging_utils as logu
from utils.logging_utils import print_verbose
from utils import wordnet_utils as wnu
from utils import ilsvrc_utils as ilsvrcu


class UniquenessScope:
    WORDNET = 'wordnet'
    ILSVRC = 'ilsvrc'


def main():
    # ----- Settings -----
    settings = dict()

    # Path
    settings['load_path'] = os.path.join('..', '..', 'ilsvrc2012')
    settings['load_file_name'] = 'ILSVRC2012_synsets.txt'

    settings['save_path'] = os.path.join(settings['load_path'], 'processed')

    # Method
    settings['ignore_empty_wnids'] = True
    settings['uniqueness_scope'] = UniquenessScope.ILSVRC

    # Verbose
    settings['verbose'] = True

    # Overwrite?
    settings['safe'] = False

    # ----- Init. -----
    logu.verbose = settings['verbose']

    os.makedirs(settings['save_path'], exist_ok=True)

    open_type = 'xb' if settings['safe'] else 'wb'

    # ----- Load the classes from txt file -----
    id_lemmas_df = ilsvrcu.load_lemmas_and_wnids(os.path.join(settings['load_path'], settings['load_file_name']))

    # ----- Create lemma2wnids map -----
    lemma2wnid = {}

    for idx in tqdm(range(id_lemmas_df.shape[0])):
        wnid = id_lemmas_df.loc[idx, 'id']

        included_lemmas = []

        for lemma in id_lemmas_df.loc[idx, 'lemmas'].split(', '):
            # Check if the lemma is unique and ignore otherwise as it probably has a general meaning.
            if settings['uniqueness_scope'] == UniquenessScope.WORDNET:
                is_unique = wnu.lemma_is_unique(lemma)
            elif settings['uniqueness_scope'] == UniquenessScope.ILSVRC:
                is_unique = ilsvrcu.lemma_is_unique(lemma, id_lemmas_df)
            else:
                raise Exception('Unrecognized uniqueness scope.')

            if not is_unique:
                continue

            if lemma in lemma2wnid:
                raise Exception(f'lemma {lemma} is already present in another sysnset!')
            else:
                lemma2wnid[lemma] = wnid
                included_lemmas.append(lemma)

        if (len(included_lemmas) == 0) and (not settings['ignore_empty_wnids']):
            for lemma in id_lemmas_df.loc[idx, 'lemmas'].split(', '):
                lemma_ex = wnu.extend_with_well_known_hypernym(lemma, wnid)
                print_verbose(f'{lemma} extended as {lemma_ex}')
                lemma = lemma_ex

                if lemma in lemma2wnid:
                    raise Exception(f'lemma {lemma} is already present in another sysnset!')
                else:
                    lemma2wnid[lemma] = wnid

    # ----- Sanity check -----
    print_verbose('\nThere are %d wnids with at least one valid lemma.' % len({wnid for _, wnid in lemma2wnid.items()}))

    # ----- Save -----
    postfix = f'unique_in_{settings["uniqueness_scope"]}'
    if settings['ignore_empty_wnids']:
        postfix += '_ignored_empty_wnids'
    file_name = f'lemma2wnid({postfix}).pkl'

    with open(os.path.join(settings['save_path'], file_name), open_type) as f:
        pickle.dump(lemma2wnid, f)


if __name__ == '__main__':
    main()
