import sys
import os
import argparse
import pickle
# FAISS will crash if you don't import it here!
import faiss

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
    parser.add_argument('--dataframe_path', type=str,
                        default=os.path.join('imagenet-captions', 'imagenet_captions.parquet'))

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
    texts_batch = []
    i_row = -1
    for idx, row in tqdm(df.iterrows(), desc='calc embeds and find most similars', total=len(df)):
        i_row += 1

        # Parse
        text = row[configs.LAIONConfig.TEXT_COL]

        # Add to batch
        indices_batch.append(idx)
        texts_batch.append(text)

        if len(indices_batch) < configs.CLIPConfig.BATCH_SIZE and i_row < (len(df) - 1):
            continue
        if len(indices_batch) == 0:
            continue

        # Get CLIP embedding
        embeds_batch = clip.text_embeds(texts_batch)
        embeds_batch_norm = normalize(embeds_batch, norm='l2', axis=1)

        # Search the index
        laion_indices_batch, sims_batch = faiss_index.search(embeds_batch_norm, k=1)
        laion_indices_batch, sims_batch = laion_indices_batch[:, 0], sims_batch[:, 0]

        # Update
        for i_idx, index in enumerate(indices_batch):
            laion_idx = laion_indices_batch[i_idx]
            sim = sims_batch[i_idx]

            # For compatibility with subset_laion, use array
            index2laionindices[index] = [laion_idx]
            index2sims[index] = [sim]

        # Step
        indices_batch = []
        texts_batch = []

    # ----- Save -----
    print_verbose('saving ...')

    file_name = 'icimagename2laionindices.pkl'
    with open(os.path.join(params['save_path'], file_name), open_type) as f:
        pickle.dump(index2laionindices, f)

    sim_file_name = 'icimagename2sims.pkl'
    with open(os.path.join(params['save_path'], sim_file_name), open_type) as f:
        pickle.dump(index2sims, f)

    print_verbose('done!\n')
