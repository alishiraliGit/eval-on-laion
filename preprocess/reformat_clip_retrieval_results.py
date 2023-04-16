import sys
import os
import argparse
import pickle
import json
from tqdm import tqdm
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

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
                                             'top50_val_most_similars_from_laion_400m.json'))

    parser.add_argument('--image_labels_path', type=str,
                        default=os.path.join('ilsvrc2012', 'processed', 'imagename2wnid.pkl'))

    parser.add_argument('--laion_path', type=str, default=os.path.join('laion400m'))

    parser.add_argument('--labels_path', type=str, default=os.path.join('laion400m', 'processed', 'ilsvrc_labels'))

    # Sampling
    parser.add_argument('--do_sample', action='store_true')

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

    print_verbose('\tloading image labels ...')
    with open(params['image_labels_path'], 'rb') as f:
        imagename2wnid = pickle.load(f)

    print_verbose('done!\n')

    # ----- Reformat -----
    wnid2crindices = {}
    df = pd.DataFrame(
        {configs.LAIONConfig.URL_COL: [], configs.LAIONConfig.TEXT_COL: [], 'image_to_image_similarity': []}
    )
    for image_idx, results in tqdm(cr_results.items()):
        image_name = 'ILSVRC2012_val_%08d.JPEG' % image_idx
        wnid = imagename2wnid[image_name]

        if wnid not in wnid2crindices:
            wnid2crindices[wnid] = []

        for res in results:
            if params['do_sample'] and len(wnid2crindices[wnid]) >= configs.LAIONSamplingConfig.UNIFORM_SAMPLES:
                break

            cr_idx = res[configs.CLIPRetrievalConfig.ID_COL]

            wnid2crindices[wnid].append(cr_idx)

            df.loc[cr_idx, [configs.LAIONConfig.URL_COL, configs.LAIONConfig.TEXT_COL, 'image_to_image_similarity']] = [
                res[configs.CLIPRetrievalConfig.URL_COL],
                res[configs.CLIPRetrievalConfig.TEXT_COL],
                res[configs.CLIPRetrievalConfig.SIMILARITY_COL]
            ]

    # ----- Save -----
    print_verbose('saving...')

    # Save labels
    print_verbose(f'\tsaving distinct {len(wnid2crindices)} labels.')

    with open(os.path.join(params['labels_path'], 'wnid2crindices.pkl'), open_type) as f:
        pickle.dump(wnid2crindices, f)

    # Save df
    print_verbose(f'\tsaving df with {len(df)} rows.')

    prefix = configs.LAIONConfig.SUBSET_CLIP_RETRIEVAL
    # For compatibility only
    subset_file_name = prefix + laionu.get_laion_subset_file_name(0, configs.LAIONConfig.NUM_PARTS - 1)
    subset_file_path = os.path.join(params['laion_path'], subset_file_name)

    if os.path.exists(subset_file_path) and params['safe']:
        raise Exception('Subset already exists!')

    df.to_parquet(subset_file_path, index=True)

    print_verbose('done!\n')
