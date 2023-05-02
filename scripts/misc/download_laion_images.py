import sys
import os
import multiprocessing
import pandas as pd
import pickle
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

import configs
import utils.laion_utils as lt
from core.retrieve_image import download_and_save_image


def download_and_save_image_wrapper(args):
    idx, data_frame, save_path = args

    # Check if already downloaded
    file_path_wo_ext = os.path.join(save_path, str(idx))
    if os.path.exists(file_path_wo_ext + '.jpg') or os.path.exists(file_path_wo_ext + '.png'):
        return idx

    url = data_frame.loc[idx, 'URL']
    try:
        download_and_save_image(url, save_path, file_name=str(idx))
        return idx
    except Exception as e:
        return e


if __name__ == '__main__':
    # ----- Settings -----
    settings = dict()

    # Path
    settings['laion_path'] = os.path.join('..', '..', 'laion400m')
    settings['map_path'] = os.path.join(settings['laion_path'], 'processed')
    settings['imgs_path'] = os.path.join(settings['laion_path'], 'imgs')

    settings['laion_until_part'] = 22

    settings['n_process'] = 8

    # ----- Load data and maps -----
    # Load LAION sampled
    subset_file_name = \
        configs.LAIONConfig.SUBSET_SM_FILTERED_PREFIX \
        + lt.get_laion_subset_file_name(0, settings['laion_until_part'])

    df = pd.read_parquet(os.path.join(settings['laion_path'], subset_file_name))

    # Load wnid2sampledlaionindices
    sampled_map_file_name = f'ILSVRC2012_wnid2sampledlaionindices.pkl'
    with open(os.path.join(settings['map_path'], sampled_map_file_name), 'rb') as f:
        wnid2laionindices = pickle.load(f)

    # ----- Parallel download -----
    with multiprocessing.Pool(processes=settings['n_process']) as pool:
        # Map the download function to each url
        results = list(tqdm(
            pool.imap_unordered(
                download_and_save_image_wrapper,
                [(idx, df, settings['imgs_path']) for idx in df.index]
            ),
            total=len(df)
        ))

    # ----- Save the successful indices ------
    successful_indices = sorted([res for res in results if isinstance(res, int)])

    successful_indices_file_name = \
        'successful_indices_' + lt.get_laion_subset_file_name(0, settings['laion_until_part'])
    with open(os.path.join(settings['laion_path'], successful_indices_file_name), 'wb') as f:
        pickle.dump(successful_indices, f)
