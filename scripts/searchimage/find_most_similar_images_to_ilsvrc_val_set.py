import sys
import os
import multiprocessing
import time
import argparse
import pickle
from PIL import Image
from io import BytesIO
import numpy as np
from sklearn.preprocessing import normalize
import pandas as pd
from tqdm.auto import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

import configs
from utils import logging_utils as logu
from utils.logging_utils import print_verbose
from utils import pytorch_utils as ptu
from utils import laion_utils as laionu
from core.retrieve_image import download_image_content, verify_image
from core.clip import CLIP


def download_image_wrapper(args):
    idx, row = args
    try:
        img_content = download_image_content(row['URL'])
        verify_image(img_content)
        return idx, img_content, None
    except Exception as e:
        return idx, None, {'cause': f'In downloading image of index {idx} an error occurred.', 'error': e}


def calc_image_embeddings(idx_img_list, clip_mdl: CLIP):
    # Load the images
    imgs = []
    inds = []
    errs = []
    for idx, img_content in idx_img_list:
        try:
            img = Image.open(BytesIO(img_content))

            inds.append(idx)
            imgs.append(img)

        except Exception as e:
            errs.append({'cause': f'In loading image of index {idx} from image content an error occurred.', 'error': e})

    # Calc. embeddings
    try:
        embs = clip_mdl.image_embeds(imgs)
        return inds, embs, errs
    except Exception as e:
        errs.append({'cause': 'In calc. image embeddings an error occurred.', 'error': e})
        return inds, None, errs


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    # ----- Get arguments from input -----
    parser = argparse.ArgumentParser()

    # Path
    parser.add_argument('--laion_path', type=str, default=os.path.join('laion400m'))
    parser.add_argument('--laion_part', type=int)

    parser.add_argument('--image_embeddings_path', type=str,
                        default=os.path.join('ilsvrc2012', 'processed', 'val_img_embeddings.pkl'))

    parser.add_argument('--save_path', type=str, default=os.path.join('laion400m', 'processed', 'ilsvrc_similarities'))

    # Multiprocessing
    parser.add_argument('--n_process_download', type=int, default=6)

    # Size
    parser.add_argument('--top_k', type=int, default=50)

    # Compute
    parser.add_argument('--no_gpu', action='store_true')
    parser.add_argument('--gpu_id', type=int, default=0)

    # Logging
    parser.add_argument('--no_verbose', dest='verbose', action='store_false')
    parser.add_argument('--save_freq', type=int, default=1000)

    # Continue
    parser.add_argument('--no_continue', dest='continue', action='store_false')

    # Destruction
    parser.add_argument('--self_destruct', action='store_true')

    # Convert to dictionary
    params = vars(parser.parse_args())

    # ----- Init. -----
    logu.verbose = params['verbose']

    print_verbose('initializing ...')

    # Env
    ptu.init_gpu(use_gpu=not params['no_gpu'], gpu_id=params['gpu_id'])

    # Path
    os.makedirs(params['save_path'], exist_ok=True)

    print_verbose('done!\n')

    # ----- Init. CLIP -----
    print_verbose('init clip ...')

    clip = CLIP()

    print_verbose('done!\n')

    # ----- Load ILSVRC validation set embeddings -----
    print_verbose('loading ilsvrc validation set embeddings ...')

    with open(os.path.join(params['image_embeddings_path']), 'rb') as f:
        embeds_val = pickle.load(f)

    n_val = embeds_val.shape[0]

    # Normalize
    embeds_val = normalize(embeds_val, axis=1, norm='l2')

    print_verbose('done!\n')

    # ----- Load LAION data -----
    df = laionu.load_data_part(params['laion_path'], params['laion_part'], params['self_destruct'])

    # ----- Load most similars (if any) -----
    top_save_path = os.path.join(params['save_path'], f'top{params["top_k"]}_val_most_similars.csv')
    top_sims_cols = [f'sim_{t + 1}' for t in range(params['top_k'])]
    top_indices_cols = [f'index_{t + 1}' for t in range(params['top_k'])]

    if os.path.exists(top_save_path) and params['continue']:
        # Load
        print_verbose('loading previous top similars to continue ...')

        top_df = pd.read_csv(top_save_path)

        assert top_df.shape[0] == n_val

        print_verbose('done!\n')

        # Drop data already existing
        print_verbose('trimming data from the last found index ...')

        max_idx = np.max(top_df[top_indices_cols].to_numpy())

        print_verbose(f'\ttrimming from index {max_idx}')

        max_loc = df.index.get_loc(max_idx)
        df = df.iloc[max_loc + 1:]

        print_verbose('done!\n')

        # Convert to appropriate format
        top_sims = top_df[top_sims_cols].to_numpy()
        top_indices = top_df[top_indices_cols].to_numpy()

        top_sims = [row.copy() for row in top_sims]
        top_indices = [row.copy() for row in top_indices]

    else:
        top_sims = [-np.ones((params['top_k'],)) for _ in range(n_val)]
        top_indices = [-np.ones((params['top_k'],)).astype(int) for _ in range(n_val)]

    # ----- Init. parallel download -----
    pool_download = multiprocessing.Pool(params['n_process_download'])

    # ----- Start download -----
    download_results = pool_download.imap(download_image_wrapper, df.iterrows())

    # ----- Collect downloads and calc. embeddings -----
    # Init.
    download_ready_results = []
    errors = []

    i_batch = 0
    for i_res, down_res in tqdm(enumerate(download_results), desc='download and calc. embeddings', total=len(df)):
        # Catch the errors in downloading
        index, image_content, error = down_res

        if error is not None:
            errors.append('\n' + error['cause'])
            errors.append(str(error['error']))
            continue

        # Append the downloaded image to the batch
        download_ready_results.append([index, image_content])
        if len(download_ready_results) < configs.CLIPConfig.BATCH_SIZE and i_res < (len(df) - 1):
            continue

        # Calc. embeddings
        indices_batch, embeds_batch, errors_embeds = calc_image_embeddings(download_ready_results, clip)
        errors.extend(errors_embeds)

        # Normalize
        embeds_batch = normalize(embeds_batch, axis=1, norm='l2')

        # Calc. similarities
        sims_vb = embeds_val.dot(embeds_batch.T)

        # Add
        for i_v in range(n_val):
            locs_i = np.searchsorted(top_sims[i_v], sims_vb[i_v])

            # Insert
            top_sims[i_v] = np.insert(top_sims[i_v], locs_i, sims_vb[i_v])[-params['top_k']:]
            top_indices[i_v] = np.insert(top_indices[i_v], locs_i, indices_batch)[-params['top_k']:]

        # Save
        if ((i_batch + 1) % params['save_freq'] == 0) or i_res == (len(df) - 1):
            print_verbose('saving ....')

            top_indices_df = pd.DataFrame(
                top_indices,
                index=range(1, configs.ILSVRCConfigs.NUM_VAL + 1),
                columns=top_indices_cols
            )
            top_sims_df = pd.DataFrame(
                top_sims,
                index=range(1, configs.ILSVRCConfigs.NUM_VAL + 1),
                columns=top_sims_cols
            )

            top_df = pd.concat((top_indices_df, top_sims_df), axis=1)

            top_df.to_csv(os.path.join(params['save_path'], f'top{params["top_k"]}_val_most_similars.csv'),
                          index=True, float_format='%.4f')

            print_verbose('done!\n')

        # Empty current batch
        download_ready_results = []
        i_batch += 1

    # ----- Close progress bars and processes -----
    pool_download.close()
    pool_download.join()

    time.sleep(1)

    # ----- Save error logs ------
    print_verbose('saving error logs ....')

    err_file_name = f'errors.txt'
    with open(os.path.join(params['save_path'], err_file_name), 'w') as f:
        f.write('\n'.join(errors))

    print_verbose('done!\n')
