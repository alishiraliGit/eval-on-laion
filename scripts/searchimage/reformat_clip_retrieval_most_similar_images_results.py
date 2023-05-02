import sys
import os
import argparse
import pickle
import json

from tqdm import tqdm
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

import configs
from utils import logging_utils as logu
from utils.logging_utils import print_verbose
from utils import laion_utils as laionu


if __name__ == '__main__':
    # ----- Get arguments from input -----
    parser = argparse.ArgumentParser()

    # Path
    parser.add_argument('--clip_retrieval_json_path', type=str,
                        default=os.path.join('laion400m', 'processed', 'from_clip_retrieval',
                                             'top20_subset_most_similars_from_laion_400m.json'))

    parser.add_argument('--labels_path', type=str, default=os.path.join('laion400m', 'processed', 'ilsvrc_labels'))
    parser.add_argument('--labels_file_name', type=str, default='wnid2laionindices(substring_matched).pkl')

    parser.add_argument('--laion_path', type=str, default=os.path.join('laion400m'))

    # Logging
    parser.add_argument('--no_verbose', dest='verbose', action='store_false')

    # Overwrite?
    parser.add_argument('--no_safe', dest='safe', action='store_false')

    # Convert to dictionary
    params = vars(parser.parse_args())

    # ----- Init. -----
    logu.verbose = params['verbose']

    open_type = 'xb' if params['safe'] else 'wb'

    # ----- Loading -----
    print_verbose('loading ...')

    print_verbose('\tloading clip retrieval results ...')
    with open(params['clip_retrieval_json_path'], 'r') as f:
        cr_results = json.load(f)
    cr_results = {int(k): v for k, v in cr_results.items()}

    print_verbose('\tloading labels ...')
    with open(os.path.join(params['labels_path'], params['labels_file_name']), 'rb') as f:
        wnid2laionindices = pickle.load(f)

    print_verbose('done!\n')

    # ----- Find the inverse map -----
    print_verbose('finding inverse map ...')

    laionindex2wnids = {}
    for wnid, laion_indices in wnid2laionindices.items():
        for laion_idx in laion_indices:
            if laion_idx not in laionindex2wnids:
                laionindex2wnids[laion_idx] = []
            laionindex2wnids[laion_idx].append(wnid)

    print_verbose('done!\n')

    # ----- Reformat -----
    wnid2crindex2sims = {}
    wnid2crindex2text = {}
    wnid2crindex2url = {}

    for laion_idx, results in tqdm(cr_results.items(), desc='collecting results into dicts'):
        wnids = laionindex2wnids[laion_idx]

        for wnid in wnids:
            if wnid not in wnid2crindex2sims:
                wnid2crindex2sims[wnid] = {}
                wnid2crindex2text[wnid] = {}
                wnid2crindex2url[wnid] = {}

            if isinstance(results, dict):
                continue

            for res in results:
                cr_idx = res[configs.CLIPRetrievalConfig.ID_COL]

                similarity = res[configs.CLIPRetrievalConfig.SIMILARITY_COL]
                text = res[configs.CLIPRetrievalConfig.TEXT_COL]
                url = res[configs.CLIPRetrievalConfig.URL_COL]

                if cr_idx not in wnid2crindex2sims[wnid]:
                    wnid2crindex2sims[wnid][cr_idx] = [similarity]
                    wnid2crindex2text[wnid][cr_idx] = text
                    wnid2crindex2url[wnid][cr_idx] = url
                else:
                    wnid2crindex2sims[wnid][cr_idx].append(similarity)

    # Reduction
    wnid2crindices = {}
    wnid2crsims = {}
    wnid2texts = {}
    wnid2urls = {}

    for wnid in tqdm(wnid2crindex2sims, desc='reducing the results'):
        crindex2sims = wnid2crindex2sims[wnid]
        crindex2text = wnid2crindex2text[wnid]
        crindex2url = wnid2crindex2url[wnid]

        wnid2crindices[wnid] = []
        wnid2crsims[wnid] = []
        wnid2texts[wnid] = []
        wnid2urls[wnid] = []

        for cr_idx in crindex2sims:
            rept = len(crindex2sims[cr_idx])

            wnid2crindices[wnid].extend([cr_idx]*rept)
            wnid2crsims[wnid].extend(crindex2sims[cr_idx])
            wnid2texts[wnid].extend([crindex2text[cr_idx]]*rept)
            wnid2urls[wnid].extend([crindex2url[cr_idx]]*rept)

    # ----- Parse -----
    text_col = configs.LAIONConfig.TEXT_COL
    url_col = configs.LAIONConfig.URL_COL

    df_index = []
    df_dict = {text_col: [], url_col: []}

    for wnid, crindices in tqdm(wnid2crindices.items(), desc='getting a dataframe out of results'):
        df_index.extend(crindices)
        df_dict[text_col].extend(wnid2texts[wnid])
        df_dict[url_col].extend(wnid2urls[wnid])

    # Create a dataframe
    df = pd.DataFrame(df_dict, index=df_index)

    # Drop duplicates
    df.index.name = 'cr_index'
    df = df.groupby('cr_index').first()

    # ----- Save -----
    print_verbose('saving...')

    # Save labels
    print_verbose(f'\tsaving distinct {len(wnid2crindices)} labels.')

    with open(os.path.join(params['labels_path'], 'wnid2smcrindices.pkl'), open_type) as f:
        pickle.dump(wnid2crindices, f)

    # Save similarities
    print_verbose(f'\tsaving image to image similarities.')

    with open(os.path.join(params['labels_path'], 'wnid2smcrimgimgsims.pkl'), open_type) as f:
        pickle.dump(wnid2crsims, f)

    # Save df
    print_verbose(f'\tsaving df with {len(df)} rows.')

    prefix = configs.LAIONConfig.SUBSET_SM_FILTERED_MOST_SIMILAR_IMG_IMG_PREFIX
    # For compatibility only
    subset_file_name = prefix + laionu.get_laion_subset_file_name(0, configs.LAIONConfig.NUM_PARTS - 1)
    subset_file_path = os.path.join(params['laion_path'], subset_file_name)

    if os.path.exists(subset_file_path) and params['safe']:
        raise Exception('Subset already exists!')

    df.to_parquet(subset_file_path, index=True)

    print_verbose('done!\n')
