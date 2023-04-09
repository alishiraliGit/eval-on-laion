import sys
import os
import argparse
import pickle
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

import configs
import utils.laion_utils as laionu


def main():
    # ----- Get arguments from input -----
    parser = argparse.ArgumentParser()

    # Paths
    parser.add_argument('--lemma2wnid_path', type=str, default=os.path.join('ilsvrc2012', 'processed'))
    parser.add_argument('--lemma2wnid_file_name', type=str, default='ILSVRC2012_lemma2wnid.pkl')

    parser.add_argument('--laion_path', type=str, default=os.path.join('laion400m'))
    parser.add_argument('--laion_part', type=int)

    parser.add_argument('--save_path', type=str, default=os.path.join('laion400m', 'processed'))

    # Logging
    parser.add_argument('--no_verbose', dest='verbose', action='store_false')

    # Destruction
    parser.add_argument('--self_destruct', action='store_true')

    # Convert to dictionary
    args = parser.parse_args()
    params = vars(args)

    # ----- Loading -----
    # Load lemma2wnid
    if params['verbose']:
        print('loading lemma2wnid ...')
    with open(os.path.join(params['lemma2wnid_path'], params['lemma2wnid_file_name']), 'rb') as f:
        lemma2wnid = pickle.load(f)

    # Load a part of LAION
    laion_file_path = os.path.join(params['laion_path'], laionu.get_laion_part_file_name(params['laion_part']))
    if not os.path.exists(laion_file_path):
        if params['verbose']:
            print(f'downloading LAION part {params["laion_part"]} ...')
        laionu.download_laion_part(part=params['laion_part'], laion_path=params['laion_path'])

    if params['verbose']:
        print(f'loading LAION part {params["laion_part"]} ...')
    df = pd.read_parquet(laion_file_path)

    # ----- Loop over LAION texts ------
    wnid2laionindices = {}
    for idx in tqdm(range(df.shape[0]),
                    desc=f'processing LAION part {params["laion_part"]}',
                    disable=not params['verbose']):

        try:
            txt = laionu.transform_text(df.loc[idx, 'TEXT'])

            wnid_set_i = {wnid for lemma, wnid in lemma2wnid.items() if laionu.transform_lemma(lemma) in txt}
        except Exception as e:
            print(e)
            continue

        for wnid in wnid_set_i:
            if wnid in wnid2laionindices:
                wnid2laionindices[wnid].append(idx)
            else:
                wnid2laionindices[wnid] = [idx]

    # ----- Finding the subset of data labeled -----
    laion_labeled_indices = set()

    for wnid, indices in wnid2laionindices.items():
        laion_labeled_indices.update(indices)

    laion_labeled_indices = sorted(laion_labeled_indices)

    # ----- Saving -----
    if params['verbose']:
        print(f'saving LAION part {params["laion_part"]} ...')

    # Save wnid2laionindices
    os.makedirs(params['save_path'], exist_ok=True)
    map_file_name = f'ILSVRC2012_wnid2laionindices(part{params["laion_part"]}).pkl'

    with open(os.path.join(params['save_path'], map_file_name), 'wb') as f:
        pickle.dump(wnid2laionindices, f)

    # Save the labeled data
    df_labeled = df.loc[laion_labeled_indices]

    subset_file_name = configs.LAIONConfig.LABELED_PREFIX + laionu.get_laion_part_file_name(params['laion_part'])

    df_labeled.to_parquet(os.path.join(params['laion_path'], subset_file_name), index=True)

    # ----- Remove the original LAION file if requested ------
    if params['self_destruct']:
        if params['verbose']:
            print(f'removing original LAION file part {params["laion_part"]} ...')

        os.remove(laion_file_path)

    print('done!')


if __name__ == '__main__':
    main()
