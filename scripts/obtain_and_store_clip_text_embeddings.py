import sys
import os
import argparse
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import normalize

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

import configs
from models import CLIP
from utils import laion_utils as laionu
from utils import pytorch_utils as ptu
from utils import logging_utils as logu
from utils.logging_utils import print_verbose


def load_data(laion_path, n_sample, self_destruct):
    # Load as much as required to obtain n_sample
    part_dfs = []
    total_samples = 0
    laion_part = 0
    while True:
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

        # Count the samples
        total_samples += len(part_df)

        # Check if sufficient
        if total_samples >= n_sample:
            total_samples -= len(part_df)
            part_df = part_df.iloc[:(n_sample - total_samples)]
            total_samples += len(part_df)

        # Reindex
        part_dfs.append(laionu.rename_index(part_df, laion_part))

        if total_samples >= n_sample:
            break

        # Step
        laion_part += 1

    # Concat part dfs
    concat_df = pd.concat(part_dfs, axis=0)

    return concat_df


if __name__ == '__main__':
    # ----- Get arguments from input -----
    parser = argparse.ArgumentParser()

    # Path
    parser.add_argument('--laion_path', type=str, default=os.path.join('laion400m'))

    parser.add_argument('--save_path', type=str, default=os.path.join('laion400m', 'processed', 'clip_text_embeddings'))

    # Size
    parser.add_argument('--n_sample', type=int)
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

    # Path
    os.makedirs(params['save_path'], exist_ok=True)

    #  Revisit params to be a multiple of batch size
    num_sample = (params['n_sample'] // configs.CLIPConfig.BATCH_SIZE + 1) * configs.CLIPConfig.BATCH_SIZE
    print_verbose(f'{num_sample} total samples will be used.')

    chunk_size = (params['chunk_size'] // configs.CLIPConfig.BATCH_SIZE + 1) * configs.CLIPConfig.BATCH_SIZE
    print_verbose(f'Each numpy stored file will contain {chunk_size} samples.')

    # ----- Load data -----
    df = load_data(params['laion_path'], num_sample, params['self_destruct'])

    # ----- Init. CLIP -----
    clip = CLIP()

    # ----- Loop over data chunks ------
    embeds = np.zeros((chunk_size, configs.CLIPConfig.DIM))
    for iloc in tqdm(range(0, len(df), configs.CLIPConfig.BATCH_SIZE),
                     desc='getting clip text embeddings', disable=not logu.verbose):
        rng = range(iloc, iloc + configs.CLIPConfig.BATCH_SIZE)

        # Extract a batch
        texts_batch = df.iloc[rng]['TEXT'].tolist()
        indices_batch = df.index[rng]

        # Get embeddings
        embeds_batch = clip.text_embeds(texts_batch)

        # Normalize
        embeds_batch_norm = normalize(embeds_batch, axis=1, norm='l2')

        # Add to the chunk
        embeds[(iloc % chunk_size): (iloc % chunk_size) + configs.CLIPConfig.BATCH_SIZE] = embeds_batch_norm

        # Save the chunk if full
        if (iloc % chunk_size) + configs.CLIPConfig.BATCH_SIZE == chunk_size:
            i_chunk = iloc // chunk_size  # Starts from 0
            with open(os.path.join(params['save_path'], f'part{i_chunk}.npy'), 'wb') as f:
                np.save(f, embeds)
