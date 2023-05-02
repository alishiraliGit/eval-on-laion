import sys
import os
import argparse
import pickle
import json

import numpy as np
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
    wnid2crindex2sims = {}
    wnid2crindex2imgindices = {}
    wnid2crindex2text = {}
    wnid2crindex2url = {}

    for image_idx, results in tqdm(cr_results.items(), desc='collecting results into dicts'):
        image_name = 'ILSVRC2012_val_%08d.JPEG' % image_idx
        wnid = imagename2wnid[image_name]

        if wnid not in wnid2crindex2sims:
            wnid2crindex2sims[wnid] = {}
            wnid2crindex2imgindices[wnid] = {}
            wnid2crindex2text[wnid] = {}
            wnid2crindex2url[wnid] = {}

        for res in results:
            cr_idx = res[configs.CLIPRetrievalConfig.ID_COL]

            similarity = res[configs.CLIPRetrievalConfig.SIMILARITY_COL]
            text = res[configs.CLIPRetrievalConfig.TEXT_COL]
            url = res[configs.CLIPRetrievalConfig.URL_COL]

            if cr_idx not in wnid2crindex2sims[wnid]:
                wnid2crindex2sims[wnid][cr_idx] = [similarity]
                wnid2crindex2imgindices[wnid][cr_idx] = [image_idx]
                wnid2crindex2text[wnid][cr_idx] = text
                wnid2crindex2url[wnid][cr_idx] = url
            else:
                wnid2crindex2sims[wnid][cr_idx].append(similarity)
                wnid2crindex2imgindices[wnid][cr_idx].append(image_idx)

    # Reduction
    wnid2crindices = {}
    wnid2crsims = {}
    wnid2imgindices = {}
    wnid2texts = {}
    wnid2urls = {}

    for wnid in tqdm(wnid2crindex2sims, desc='reducing the results'):
        crindex2sims = wnid2crindex2sims[wnid]
        crindex2imgindices = wnid2crindex2imgindices[wnid]
        crindex2text = wnid2crindex2text[wnid]
        crindex2url = wnid2crindex2url[wnid]

        wnid2crindices[wnid] = []
        wnid2crsims[wnid] = []
        wnid2imgindices[wnid] = []
        wnid2texts[wnid] = []
        wnid2urls[wnid] = []

        for cr_idx in crindex2sims:
            wnid2crindices[wnid].append(cr_idx)
            wnid2crsims[wnid].append(np.max(crindex2sims[cr_idx]))
            wnid2imgindices[wnid].append(crindex2imgindices[cr_idx][np.argmax(crindex2sims[cr_idx])])
            wnid2texts[wnid].append(crindex2text[cr_idx])
            wnid2urls[wnid].append(crindex2url[cr_idx])

    # ----- Sample -----
    if params['do_sample']:

        for wnid, sims in tqdm(wnid2crsims.items(), desc='sampling'):
            pos = np.argsort(sims)[-configs.LAIONSamplingConfig.UNIFORM_SAMPLES:]

            wnid2crindices[wnid] = np.array(wnid2crindices[wnid])[pos].tolist()
            wnid2crsims[wnid] = np.array(wnid2crsims[wnid])[pos].tolist()
            wnid2imgindices[wnid] = np.array(wnid2imgindices[wnid])[pos].tolist()
            wnid2texts[wnid] = np.array(wnid2texts[wnid])[pos].tolist()
            wnid2urls[wnid] = np.array(wnid2urls[wnid])[pos].tolist()

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

    with open(os.path.join(params['labels_path'], 'wnid2crindices.pkl'), open_type) as f:
        pickle.dump(wnid2crindices, f)

    print_verbose(f'\tsaving which ilsvrc images correspond to the sampled images.')

    with open(os.path.join(params['labels_path'], 'wnid2ilsvrcimgindices.pkl'), open_type) as f:
        pickle.dump(wnid2imgindices, f)

    # Save similarities
    print_verbose(f'\tsaving image to image similarities.')

    with open(os.path.join(params['labels_path'], 'wnid2crimgimgsims.pkl'), open_type) as f:
        pickle.dump(wnid2crsims, f)

    # Save df
    print_verbose(f'\tsaving df with {len(df)} rows.')

    prefix = configs.LAIONConfig.SUBSET_VAL_MOST_SIMILAR_IMG_IMG_PREFIX
    # For compatibility only
    subset_file_name = prefix + laionu.get_laion_subset_file_name(0, configs.LAIONConfig.NUM_PARTS - 1)
    subset_file_path = os.path.join(params['laion_path'], subset_file_name)

    if os.path.exists(subset_file_path) and params['safe']:
        raise Exception('Subset already exists!')

    df.to_parquet(subset_file_path, index=True)

    print_verbose('done!\n')
