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
    # Download if required
    laion_file_path = os.path.join(laion_path, laionu.get_laion_part_file_name(laion_part))
    if not os.path.exists(laion_file_path):
        print_verbose(f'downloading LAION part {laion_part} ...')

        laionu.download_laion_part(part=laion_part, laion_path=laion_path)

        print_verbose('done!')

    # Load LAION part
    part_df = pd.read_parquet(laion_file_path)

    # Self-destruct
    if self_destruct:
        print_verbose(f'removing LAION part {laion_part} from the disk ...')

        os.remove(laion_file_path)

        print_verbose('done!')

    # Reindex
    part_df = laionu.rename_index(part_df, laion_part)

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

    # Logging
    parser.add_argument('--verbose', type=bool, default=True)

    # Destruction
    parser.add_argument('--self_destruct', action='store_true')

    # Convert to dictionary
    params = vars(parser.parse_args())

    # ----- Init. -----
    # Env
    ptu.init_gpu(not params['no_gpu'])
    logu.verbose = params['verbose']

    #  Revisit params to be a multiple of batch size
    chunk_size = (params['chunk_size'] // configs.CLIPConfig.BATCH_SIZE + 1) * configs.CLIPConfig.BATCH_SIZE
    print_verbose(f'Each npy file will contain {chunk_size} embeddings.')

    # ----- Load data -----
    df = load_data_part(params['laion_path'], params['laion_part'], params['self_destruct'])

    # ----- Load indices and drop already existing data from df -----
    # Load indices
    with open(params['indices_path'], 'rb') as f:
        # noinspection PyTypeChecker
        all_indices = np.load(f)

    # Update df
    max_idx = np.max(all_indices)
    if max_idx in df.index:
        print_verbose('dropping samples already encoded in the index ...')

        df = df.iloc[df.index.get_loc(max_idx) + 1:]

        print_verbose('done!')

    # ----- Load the index -----
    print_verbose('loading the index ...')

    faiss_index = faiss.read_index(params['faiss_index_path'])

    print_verbose('done!')

    # ----- Init. CLIP -----
    clip = CLIP()

    # ----- Loop over data chunks ------
    embeds = np.zeros((chunk_size, configs.CLIPConfig.DIM))
    indices = np.zeros((chunk_size,)).astype(int)
    chunk_cnt = 0
    for iloc in tqdm(range(0, len(df), configs.CLIPConfig.BATCH_SIZE),
                     desc='getting clip text embeddings', disable=not logu.verbose):
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
            print_verbose('adding to the index ...')
            faiss_index.add(embeds[:chunk_cnt])
            print_verbose('done!')

            # Save the index
            print_verbose('saving the index ...')
            faiss.write_index(faiss_index, params['faiss_index_path'])
            print_verbose('done!')

            # Update the indices
            all_indices = np.append(all_indices, indices[:chunk_cnt])

            # Save the indices
            with open(params['indices_path'], 'wb') as f:
                # noinspection PyTypeChecker
                np.save(f, all_indices)

            # Reset the params
            chunk_cnt = 0
            indices *= 0
            embeds *= 0
