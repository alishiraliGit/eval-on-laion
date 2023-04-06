import sys
import os
import argparse
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import normalize
import faiss

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

import configs
from models import CLIP
from utils import laion_utils as laionu
from utils import pytorch_utils as ptu
from utils import logging_utils as logu
from utils.logging_utils import print_verbose


def load_data_part(laion_path, laion_part, self_destruct):
    print_verbose(f'loading laion part {laion_part} ...')

    # Download if required
    laion_file_path = os.path.join(laion_path, laionu.get_laion_part_file_name(laion_part))
    if not os.path.exists(laion_file_path):
        print_verbose(f'\tdownloading laion part {laion_part} ...')

        laionu.download_laion_part(part=laion_part, laion_path=laion_path)

        print_verbose('\tdownloaded!')

    # Load LAION part
    part_df = pd.read_parquet(laion_file_path)

    # Self-destruct
    if self_destruct:
        print_verbose(f'\tremoving laion part {laion_part} from the disk ...')

        os.remove(laion_file_path)

        print_verbose('\tremoved!')

    # Reindex
    part_df = laionu.rename_index(part_df, laion_part)

    print_verbose('done!\n')

    return part_df


if __name__ == '__main__':
    # ----- Get arguments from input -----
    parser = argparse.ArgumentParser()

    # Path
    parser.add_argument('--laion_path', type=str, default=os.path.join('laion400m'))
    parser.add_argument('--laion_part', type=int)

    parser.add_argument('--indices_path', type=str,
                        default=os.path.join('laion400m', 'processed', 'clip_text_indices', 'all_indices.npy'))

    parser.add_argument('--faiss_index_path', type=str,
                        default=os.path.join('laion400m', 'processed', 'faiss_index', 'knn.index'))

    # Size
    parser.add_argument('--chunk_size', type=int)

    # Compute
    parser.add_argument('--no_gpu', action='store_true')
    parser.add_argument('--gpu_id', type=int, default=0)

    # Logging
    parser.add_argument('--verbose', type=bool, default=True)

    # Destruction
    parser.add_argument('--self_destruct', action='store_true')

    # Convert to dictionary
    params = vars(parser.parse_args())

    # ----- Init. -----
    logu.verbose = params['verbose']

    print_verbose('initializing ...')

    # Env
    ptu.init_gpu(use_gpu=not params['no_gpu'], gpu_id=params['gpu_id'])

    #  Revisit params to be a multiple of batch size
    chunk_size = (params['chunk_size'] // configs.CLIPConfig.BATCH_SIZE + 1) * configs.CLIPConfig.BATCH_SIZE
    print_verbose(f'\teach chunk will contain {chunk_size} embeddings.')

    print_verbose('done!\n')

    # ----- Load data -----
    df = load_data_part(params['laion_path'], params['laion_part'], params['self_destruct'])

    # ----- Load indices -----
    print_verbose('loading indices ...')

    with open(params['indices_path'], 'rb') as f:
        # noinspection PyTypeChecker
        all_indices = np.load(f)
    print_verbose(f'\tfound {len(all_indices)} rows in all_indices.')

    print_verbose('done!\n')

    # ----- Drop already existing data from df -----
    print_verbose('dropping rows already encoded in the index ...')

    _, intersec_locs, _ = np.intersect1d(df.index, all_indices, return_indices=True)
    keep_mask = np.ones((len(df),)).astype(bool)
    keep_mask[intersec_locs] = False

    print_verbose(f'\tfound {np.sum(~keep_mask)} rows already existing in the index.')

    # Check if any new row is there
    if not np.any(keep_mask):
        raise Exception('no new row to be encoded and stored.')

    df = df.iloc[keep_mask]

    print_verbose('done!\n')

    # ----- Load faiss index -----
    print_verbose('loading faiss index ...')

    faiss_index = faiss.read_index(params['faiss_index_path'])

    print_verbose(f'\tfound {faiss_index.ntotal} rows in faiss index.')

    print_verbose('done!\n')

    # ----- Init. CLIP -----
    print_verbose('init clip ...')

    clip = CLIP()

    print_verbose('done!\n')

    # ----- Loop over data chunks ------
    embeds = np.zeros((chunk_size, configs.CLIPConfig.DIM))
    indices = np.zeros((chunk_size,)).astype(int)
    chunk_cnt = 0
    for iloc in tqdm(range(0, len(df), configs.CLIPConfig.BATCH_SIZE),
                     desc='calc and store clip text embeddings', disable=not logu.verbose):
        rng = range(iloc, np.minimum(iloc + configs.CLIPConfig.BATCH_SIZE, len(df)))

        # Extract a batch
        texts_batch = df.iloc[rng]['TEXT'].fillna(configs.CLIPConfig.REPLACE_NA_STR).tolist()
        indices_batch = df.index[rng].to_numpy(dtype=int)

        try:
            # Get embeddings
            embeds_batch = clip.text_embeds(texts_batch)

            # Normalize
            embeds_batch_norm = normalize(embeds_batch, axis=1, norm='l2')

        except Exception as e:
            print(str(e))
            embeds_batch_norm = np.zeros((len(rng), configs.CLIPConfig.DIM))

        # Add to the chunk
        chunk_rng = range(chunk_cnt, chunk_cnt + len(rng))
        indices[chunk_rng] = indices_batch
        embeds[chunk_rng] = embeds_batch_norm

        # Step
        chunk_cnt += len(rng)

        # Store the chunk if full or reached the end of df
        if chunk_rng[-1] == chunk_size - 1 or rng[-1] == len(df) - 1:
            i_chunk = iloc // chunk_size  # Starts from 0

            # Add to the index
            print_verbose('adding to faiss index ...')
            faiss_index.add(embeds[:chunk_cnt])
            print_verbose('done!\n')

            # Save the index
            print_verbose('saving faiss index ...')
            faiss.write_index(faiss_index, params['faiss_index_path'])
            print_verbose('done!\n')

            # Update and save the indices
            print_verbose('updating and saving indices ...')

            all_indices = np.append(all_indices, indices[:chunk_cnt])

            with open(params['indices_path'], 'wb') as f:
                # noinspection PyTypeChecker
                np.save(f, all_indices)

            print_verbose('done!\n')

            # Reset the params
            chunk_cnt = 0
            indices *= 0
            embeds *= 0
