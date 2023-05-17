import sys
import os
import argparse
import pickle
# FAISS will crash if you don't import it here!
import faiss
import glob

import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import normalize

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

import configs
from core.clip import CLIP
from core.faiss_index import FaissIndex
from utils import pytorch_utils as ptu
from utils import logging_utils as logu
from utils.logging_utils import print_verbose


if __name__ == '__main__':
    # ----- Get arguments from input -----
    parser = argparse.ArgumentParser()

    # Path
    parser.add_argument('--dataframe_path', type=str, default=os.path.join('ilsvrc2012', 'imagenet_captions.parquet'))
    parser.add_argument('--index2wnid_path', type=str,
                        default=os.path.join('ilsvrc2012', 'processed', 'labels', 'icimagename2wnid.pkl'))
    parser.add_argument('--labels_path', type=str, default=os.path.join('laion400m', 'processed', 'ilsvrc_labels'))
    parser.add_argument('--labels_filter', type=str, default='wnid2laionindices(substring_matched_part*).pkl')

    parser.add_argument('--indices_path', type=str,
                        default=os.path.join('laion400m', 'processed', 'clip_text_indices', 'all_indices.npy'))

    parser.add_argument('--faiss_index_path', type=str,
                        default=os.path.join('laion400m', 'processed', 'faiss_index', 'knn.index'))

    parser.add_argument('--save_path', type=str, default=os.path.join('laion400m', 'processed', 'ilsvrc_labels'))

    # Compute
    parser.add_argument('--no_gpu', action='store_true')
    parser.add_argument('--gpu_id', type=int, default=0)

    # Logging
    parser.add_argument('--no_verbose', dest='verbose', action='store_false')

    # Overwrite?
    parser.add_argument('--no_safe', dest='safe', action='store_false')

    # Convert to dictionary
    params = vars(parser.parse_args())

    # ----- Init. -----
    logu.verbose = params['verbose']

    print_verbose('initializing ...')

    # Env
    ptu.init_gpu(use_gpu=not params['no_gpu'], gpu_id=params['gpu_id'])

    # Path
    os.makedirs(params['save_path'], exist_ok=True)

    # Safety
    open_type = 'xb' if params['safe'] else 'wb'

    print_verbose('done!\n')

    # ----- Init. CLIP -----
    print_verbose('init clip ...')

    clip = CLIP()

    print_verbose('done!\n')

    # ----- Load dataframe labels -----
    print_verbose('loading dataframe labels ...')

    with open(os.path.join(params['index2wnid_path']), 'rb') as f:
        index2wnid = pickle.load(f)

    print_verbose('done!\n')

    # ----- Load LAION labels -----
    print_verbose('loading laion labels (maps) ...')

    maps_paths = glob.glob(os.path.join(params['labels_path'], params['labels_filter']))

    print_verbose(f'\tfound {len(maps_paths)} wnid2laion maps:\n')
    print_verbose('\t- ' + '\n\t- '.join(maps_paths))

    wnid2laionindices = {}
    for path in tqdm(maps_paths):
        with open(path, 'rb') as f:
            wnid2laionindices_i = pickle.load(f)

        for wnid, laionindices in wnid2laionindices_i.items():
            if wnid not in wnid2laionindices:
                wnid2laionindices[wnid] = []
            wnid2laionindices[wnid].extend(laionindices)

    print_verbose('done!\n')

    # ----- Load the text index -----
    faiss_index = FaissIndex.load(params['faiss_index_path'], params['indices_path'])

    # ----- Load dataframe -----
    print_verbose('loading dataframe ...')

    df = pd.read_parquet(params['dataframe_path'])

    print_verbose(f'\tfound {len(df)} rows.')

    print_verbose('done!\n')

    # ----- Collect downloads and calc. embeddings -----
    # Init.
    index2laionindices = {}
    index2sims = {}

    indices_batch = []
    wnids_batch = []
    texts_batch = []
    i_row = -1
    for idx, row in tqdm(df.iterrows(), desc='calc embeds and find most similars', total=len(df)):
        i_row += 1

        # Parse
        text = row[configs.LAIONConfig.TEXT_COL]

        # Add to batch
        indices_batch.append(idx)
        wnids_batch.append(index2wnid[idx])
        texts_batch.append(text)

        if len(indices_batch) < configs.CLIPConfig.BATCH_SIZE and i_row < (len(df) - 1):
            continue
        if len(indices_batch) == 0:
            continue

        # Get CLIP embedding
        embeds_batch = clip.text_embeds(texts_batch)
        embeds_batch_norm = normalize(embeds_batch, norm='l2', axis=1)

        # Search the index
        k = 512
        laion_indices_bk, sims_bk = faiss_index.search(embeds_batch_norm, k=k)

        laion_indices_b, sims_b = [], []
        for i_row in range(len(laion_indices_bk)):
            laion_indices_k, sims_k = laion_indices_bk[i_row], sims_bk[i_row]

            laion_index, sim = None, None

            wnid = wnids_batch[i_row]
            if wnid in wnid2laionindices:
                for k_res in range(k):
                    if laion_indices_k[k_res] in wnid2laionindices[wnid]:
                        laion_index = laion_indices_k[k_res]
                        sim = sims_k[k_res]
                        break

            laion_indices_b.append(laion_index)
            sims_b.append(sim)

        # Update
        for i_row, index in enumerate(indices_batch):
            if laion_indices_b[i_row] is None:
                continue
            laion_idx = laion_indices_b[i_row]
            sim = sims_b[i_row]

            # For compatibility with subset_laion, use array
            index2laionindices[index] = [laion_idx]
            index2sims[index] = [sim]

        # Step
        indices_batch = []
        wnids_batch = []
        texts_batch = []

    # ----- Save -----
    print_verbose('saving ...')

    prefix = configs.LAIONConfig.SUBSET_IC_MOST_SIMILAR_TXT_TXT_WITH_SIMILAR_LABELS_PREFIX

    file_name = f'icimagename2laionindices({prefix[:-1]}).pkl'
    with open(os.path.join(params['save_path'], file_name), open_type) as f:
        pickle.dump(index2laionindices, f)

    sim_file_name = f'icimagename2sims({prefix[:-1]}).pkl'
    with open(os.path.join(params['save_path'], sim_file_name), open_type) as f:
        pickle.dump(index2sims, f)

    print_verbose('done!\n')
