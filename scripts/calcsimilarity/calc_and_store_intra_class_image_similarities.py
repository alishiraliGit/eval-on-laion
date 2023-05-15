import sys
import os
import multiprocessing
import time
import argparse
import pickle
from PIL import Image
from io import BytesIO
import pandas as pd
from tqdm.auto import tqdm
from sklearn.preprocessing import normalize

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

import configs
from utils import logging_utils as logu
from utils.logging_utils import print_verbose
from utils import pytorch_utils as ptu
from utils import laion_utils as laionu
from core.retrieve_image import download_image_content, verify_image
from core.clip import CLIP


def download_images_wrapper(args):
    w, inds, urls = args

    assert len(inds) == len(urls)

    img_contents = []
    errs = []
    for i_url, url in enumerate(urls):
        try:
            img_content = download_image_content(url)
            verify_image(img_content)
            img_contents.append(img_content)
        except Exception as e:
            img_contents.append(None)
            errs.append({'cause': f'In downloading image of index {inds[i_url]} an error occurred.', 'error': e})

    return w, inds, img_contents, errs


def df_gen(wnid2inds, dataframe, min_len=2):
    wnid2inds = dict(sorted(wnid2inds.items(), key=lambda item: len(item[1])))

    for w, inds in wnid2inds.items():
        if len(inds) < min_len:
            continue

        urls = dataframe.loc[inds, configs.LAIONConfig.URL_COL].tolist()
        yield w, inds, urls


def calc_image_cross_similarities(inds, img_contents, clip_mdl: CLIP):
    # Load the images
    errs = []
    imgs = []
    success_inds = []
    for i_i, img_content in enumerate(img_contents):
        if img_content is None:
            continue

        try:
            img = Image.open(BytesIO(img_content))
            imgs.append(img)
            success_inds.append(inds[i_i])
        except Exception as e:
            errs.append(
                {
                    'cause': f'In loading image of index {inds[i_i]} from image content an error occurred.',
                    'error': e
                }
            )

    if len(imgs) == 0:
        sims = []
        errs.append({'cause': 'No image for this class.', 'error': None})
        return success_inds, sims, errs

    # Calc. similarities
    try:
        embs = clip_mdl.image_embeds(imgs)
        embs = normalize(embs, axis=1, norm='l2')
        sims = embs.dot(embs.T)
    except Exception as e:
        sims = []
        errs.append({'cause': 'In calc. image cross similarities an error occurred.', 'error': e})

    # Close the images
    for img in imgs:
        img.close()

    return success_inds, sims, errs


if __name__ == '__main__':
    # ----- Get arguments from input -----
    parser = argparse.ArgumentParser()

    # Path
    parser.add_argument('--laion_path', type=str, default=os.path.join('laion400m'))
    parser.add_argument('--laion_until_part', type=int, default=31)

    parser.add_argument('--labels_path', type=str, default=os.path.join('laion400m', 'processed', 'ilsvrc_labels'))
    parser.add_argument('--labels_file_name', type=str)

    parser.add_argument('--save_path', type=str,
                        default=os.path.join('laion400m', 'processed', 'clip_image_similarities'))

    # Method
    parser.add_argument('--method', type=str, help='Look at configs.LAIONConfig.')

    # Sample
    parser.add_argument('--do_sample', action='store_true')
    parser.add_argument('--n_sample', type=int, default=100)

    # Multiprocessing
    parser.add_argument('--n_process_download', type=int, default=6)

    # Compute
    parser.add_argument('--no_gpu', action='store_true')
    parser.add_argument('--gpu_id', type=int, default=0)

    # Logging
    parser.add_argument('--no_verbose', dest='verbose', action='store_false')

    # Convert to dictionary
    params = vars(parser.parse_args())

    # ----- Init. -----
    logu.verbose = params['verbose']

    print_verbose('initializing ...')

    # Compute
    ptu.init_gpu(use_gpu=not params['no_gpu'], gpu_id=params['gpu_id'])

    # Set the files prefix
    prefix = configs.LAIONConfig.method_to_prefix(params['method'])

    # Saving
    os.makedirs(params['save_path'], exist_ok=True)
    wnid2savefilename = lambda w: prefix + f'img_img_sims({w}).pkl'

    print_verbose('done!\n')

    # ----- Init. CLIP -----
    print_verbose('init clip ...')

    clip = CLIP()

    print_verbose('done!\n')

    # ----- Load dataframe -----
    print_verbose('loading dataframe ...')

    # Load
    file_name_wo_prefix = laionu.get_laion_subset_file_name(0, params['laion_until_part'])
    subset_file_name = prefix + file_name_wo_prefix
    subset_file_path = os.path.join(params['laion_path'], subset_file_name)

    df = pd.read_parquet(subset_file_path)

    print_verbose('done!\n')

    # ----- Load LAION labels -----
    print_verbose('loading labels ...')

    with open(os.path.join(params['labels_path'], params['labels_file_name']), 'rb') as f:
        wnid2laionindices = pickle.load(f)

    print_verbose('done!\n')

    # ----- Sample LAION -----
    if params['do_sample']:
        print_verbose('sampling laion ...')

        wnid2laionindices = {wnid: laion_indices[:params['n_sample']] for
                             wnid, laion_indices in wnid2laionindices.items()}

        print_verbose('done!\n')

    # ----- Init. parallel download -----
    pool_download = multiprocessing.Pool(params['n_process_download'])

    # ----- Start download -----
    download_results = pool_download.imap(download_images_wrapper, df_gen(wnid2laionindices, df))

    # ----- Collect downloads and calc. embeddings -----
    # Init.
    errors = []

    for i_res, down_res in tqdm(enumerate(download_results),
                                desc='download and calc. sim.', total=len(wnid2laionindices)):
        # Parse download results
        wnid, laion_indices, image_contents, down_errors = down_res

        for error in down_errors:
            errors.append('\n' + error['cause'])
            errors.append(str(error['error']))

        # Calc. similarities
        success_laion_indices, similarities, sim_errors = \
            calc_image_cross_similarities(laion_indices, image_contents, clip)

        for error in sim_errors:
            errors.append('\n' + error['cause'])
            errors.append(str(error['error']))

        if len(similarities) == 0:
            continue
        if similarities.shape == (1, 1):
            continue

        # Save similarities
        with open(os.path.join(params['save_path'], wnid2savefilename(wnid)), 'wb') as f:
            pickle.dump(
                {
                    'index': success_laion_indices,
                    'similarities': similarities
                },
                f
            )

    # ----- Close progress bars and processes -----
    pool_download.close()
    pool_download.join()

    time.sleep(1)

    # ----- Save error logs ------
    print_verbose('saving error logs ....')

    err_file_name = prefix + 'img_img_sims_errors.txt'
    with open(os.path.join(params['save_path'], err_file_name), 'w') as f:
        f.write('\n'.join(errors))

    print_verbose('done!\n')
