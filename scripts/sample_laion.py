import sys
import os
import pandas as pd
import pickle
from tqdm import tqdm
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

import configs
import utils.laion_utils as laionu


if __name__ == '__main__':
    # ----- Settings -----
    settings = dict()

    # Path
    settings['laion_path'] = os.path.join('..', 'laion400m')
    settings['map_path'] = os.path.join(settings['laion_path'], 'processed')

    # ----- Load and merge available labeled parts -----
    part_dfs = []
    wnid2laionindices = {}
    found_parts = []
    for part in tqdm(range(configs.LAIONConfig.NUM_PARTS), desc='loading labeled data and maps'):
        laion_part_file_path = os.path.join(
            settings['laion_path'],
            configs.LAIONConfig.LABELED_PREFIX + laionu.get_laion_part_file_name(part)
        )

        if not os.path.exists(laion_part_file_path):
            continue

        found_parts.append(part)

        # Load labeled LAION part
        part_df = pd.read_parquet(laion_part_file_path)

        # Reindex
        part_dfs.append(laionu.rename_index(part_df, part))

        # Load wnid2laion map
        map_file_name = f'ILSVRC2012_wnid2laionindices(part{part}).pkl'
        with open(os.path.join(settings['map_path'], map_file_name), 'rb') as f:
            wnid2laionpartindices = pickle.load(f)

        # Reindex
        for wnid, laionpartindices in wnid2laionpartindices.items():
            if wnid not in wnid2laionindices:
                wnid2laionindices[wnid] = []

            wnid2laionindices[wnid].extend([laionu.map_index(idx, part) for idx in laionpartindices])

    # Concat part dfs
    df = pd.concat(part_dfs, axis=0)

    # Clean the memory
    del wnid2laionpartindices
    del part_dfs, part_df

    # ----- Remove NSFW -----
    wnid2safelaionindices = {}
    for wnid, laionindices in tqdm(wnid2laionindices.items(), desc='removing NSFW'):
        wnid2safelaionindices[wnid] = \
            [idx for idx in laionindices if df.loc[idx, 'NSFW'] == configs.LAIONConfig.SAFE_TAG]

    wnid2laionindices = wnid2safelaionindices
    df = df.loc[df.loc[:, 'NSFW'] == configs.LAIONConfig.SAFE_TAG]

    # ----- Sample -----
    # Uniform samples
    wnid2uniformlaionindices = {}
    for wnid, laionindices in tqdm(wnid2laionindices.items(), desc='uniform sampling'):
        wnid2uniformlaionindices[wnid] = laionindices[:configs.LAIONSamplingConfig.UNIFORM_SAMPLES]

    # Samples from different ranges of CLIP similarity
    sim_bins = laionu.icdf_bins(df)
    wnid2icdflaionindices = {}
    for wnid, laionindices in tqdm(wnid2laionindices.items(), desc='icdf sampling'):
        wnid2icdflaionindices[wnid] = []

        laionindices = np.array(laionindices)
        sims = np.array(df.loc[laionindices, 'similarity'].tolist())

        for i_b in range(len(sim_bins) - 1):
            lb = sim_bins[i_b]
            rb = sim_bins[i_b + 1]
            bin_indices = laionindices[np.logical_and(sims >= lb, sims < rb)]
            wnid2icdflaionindices[wnid].extend(bin_indices[:configs.LAIONSamplingConfig.SAMPLES_PER_SIMILARITY_BIN])

    # Union samples
    wnid2sampledlaionindices = {}
    for wnid in wnid2laionindices:
        wnid2sampledlaionindices[wnid] = list(set(wnid2uniformlaionindices[wnid] + wnid2icdflaionindices[wnid]))

    # Select only sampled data from LAION
    all_laionindices = []
    for _, sampledlaionindices in wnid2sampledlaionindices.items():
        all_laionindices.extend(sampledlaionindices)

    all_laionindices = sorted(set(all_laionindices))

    sampled_df = df.loc[all_laionindices]

    # ----- Save -----
    print('saving ...')
    # Save maps
    uniform_map_file_name = f'ILSVRC2012_wnid2uniformlaionindices.pkl'
    with open(os.path.join(settings['map_path'], uniform_map_file_name), 'wb') as f:
        pickle.dump(wnid2uniformlaionindices, f)

    icdf_map_file_name = f'ILSVRC2012_wnid2icdflaionindices.pkl'
    with open(os.path.join(settings['map_path'], icdf_map_file_name), 'wb') as f:
        pickle.dump(wnid2icdflaionindices, f)

    sampled_map_file_name = f'ILSVRC2012_wnid2sampledlaionindices.pkl'
    with open(os.path.join(settings['map_path'], sampled_map_file_name), 'wb') as f:
        pickle.dump(wnid2sampledlaionindices, f)

    # Save sampled LAION dataframe
    subset_file_name = \
        configs.LAIONConfig.SUBSET_PREFIX \
        + laionu.get_laion_subset_file_name(min(found_parts), max(found_parts))

    sampled_df.to_parquet(os.path.join(settings['laion_path'], subset_file_name), index=True)

    print('done!')
