import sys
import os
import gc
import argparse
import pickle
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import normalize

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

import configs
from core.clip import CLIP
from utils import laion_utils as laionu
from utils import pytorch_utils as ptu
from utils import logging_utils as logu
from utils.logging_utils import print_verbose


def load_data(laion_path, n_sample, last_idx, self_destruct):
    # Load as much as required to obtain n_sample
    part_dfs = []
    total_samples = 0
    laion_part, _ = laionu.imap_index(last_idx + 1)
    start_storing = False
    while laion_part < configs.LAIONConfig.NUM_PARTS:
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

        # Rename index
        part_df = laionu.rename_index(part_df, laion_part)

        # Check if should start storing
        if (not start_storing) and (last_idx in part_df.index):
            part_df = part_df.iloc[part_df.index.get_loc(last_idx) + 1:]
            start_storing = True

        if part_df.index[0] > last_idx:
            start_storing = True

        # Count the samples
        if start_storing:
            total_samples += len(part_df)

        # Check if sufficient
        if total_samples >= n_sample:
            total_samples -= len(part_df)
            part_df = part_df.iloc[:(n_sample - total_samples)]
            total_samples += len(part_df)

        # Reindex and add
        if start_storing:
            part_dfs.append(part_df)
        else:
            del part_df

        if total_samples >= n_sample:
            break

        # Step
        laion_part += 1

        gc.collect()

    # Concat part dfs
    concat_df = pd.concat(part_dfs, axis=0)

    return concat_df


if __name__ == '__main__':
    # ----- Get arguments from input -----
    parser = argparse.ArgumentParser()

    # Path
    parser.add_argument('--laion_path', type=str, default=os.path.join('laion400m'))

    parser.add_argument('--save_path', type=str, default=os.path.join('laion400m', 'processed', 'clip_text_embeddings'))
    parser.add_argument('--indices_save_path', type=str,
                        default=os.path.join('laion400m', 'processed', 'clip_text_indices'))

    # Size
    parser.add_argument('--n_sample', type=int)
    parser.add_argument('--chunk_size', type=int)
    parser.add_argument('--last_index', type=int, default=-1)
    parser.add_argument('--find_last_index', action='store_true')

    # Compute
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--no_gpu', action='store_true')

    # Logging
    parser.add_argument('--no_verbose', dest='verbose', action='store_false')

    # Destruction
    parser.add_argument('--self_destruct', action='store_true')

    # Convert to dictionary
    params = vars(parser.parse_args())

    # ----- Init. -----
    # Env
    ptu.init_gpu(use_gpu=not params['no_gpu'], gpu_id=params['gpu_id'])
    logu.verbose = params['verbose']

    # Path
    os.makedirs(params['save_path'], exist_ok=True)
    os.makedirs(params['indices_save_path'], exist_ok=True)

    # Revisit params to be a multiple of batch size
    chunk_size = (params['chunk_size'] // configs.CLIPConfig.BATCH_SIZE + 1) * configs.CLIPConfig.BATCH_SIZE
    print_verbose(f'Each npy file will contain {chunk_size} embeddings.')

    num_sample = (params['n_sample'] // chunk_size + 1) * chunk_size
    print_verbose(f'{num_sample} total samples will be used.')

    # Find last_index if asked
    if params['find_last_index']:
        with open(os.path.join(params['indices_save_path'], 'all_indices.npy'), 'rb') as f:
            # noinspection PyTypeChecker
            all_indices = np.load(f)
        last_index = np.max(all_indices)
        print_verbose(f'last index found at {last_index}.')
    else:
        last_index = params['last_index']

    # ----- Load data -----
    df = load_data(params['laion_path'], num_sample, last_index, params['self_destruct'])

    # ----- Init. CLIP -----
    clip = CLIP()

    # ----- Loop over data chunks ------
    embeds = np.zeros((chunk_size, configs.CLIPConfig.DIM))
    indices = np.zeros((chunk_size,)).astype(int)
    for iloc in tqdm(range(0, len(df), configs.CLIPConfig.BATCH_SIZE),
                     desc='getting clip text embeddings', disable=not logu.verbose):
        rng = range(iloc, iloc + configs.CLIPConfig.BATCH_SIZE)

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
            embeds_batch_norm = np.zeros((configs.CLIPConfig.BATCH_SIZE, configs.CLIPConfig.DIM))

        # Add to the chunk
        chunk_rng = range(iloc % chunk_size, (iloc % chunk_size) + configs.CLIPConfig.BATCH_SIZE)
        indices[chunk_rng] = indices_batch
        embeds[chunk_rng] = embeds_batch_norm

        # Save the chunk if full
        if chunk_rng[-1] == chunk_size - 1:
            i_chunk = (iloc + last_index + 1) // chunk_size  # Starts from 0

            with open(os.path.join(params['indices_save_path'], 'indices-%05d.npy' % i_chunk), 'wb') as f:
                pickle.dump(indices, f)

            with open(os.path.join(params['save_path'], 'embeddings-%05d.npy' % i_chunk), 'wb') as f:
                pickle.dump(embeds, f)

            indices *= 0
            embeds *= 0
