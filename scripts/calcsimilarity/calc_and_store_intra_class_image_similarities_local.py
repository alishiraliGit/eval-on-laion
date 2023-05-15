import sys
import os
import argparse
import pickle
from PIL import Image
import pandas as pd
from tqdm.auto import tqdm
from sklearn.preprocessing import normalize

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

from utils import logging_utils as logu
from utils.logging_utils import print_verbose
from utils import pytorch_utils as ptu
from core.clip import CLIP


def calc_image_cross_similarities(imgs, clip_mdl: CLIP):
    embs = clip_mdl.image_embeds(imgs)
    embs = normalize(embs, axis=1, norm='l2')
    sims = embs.dot(embs.T)

    return sims


if __name__ == '__main__':
    # ----- Get arguments from input -----
    parser = argparse.ArgumentParser()

    # Path
    parser.add_argument('--images_path', type=str, default=os.path.join('ilsvrc2012', 'ILSVRC2012_img_train_selected'))
    parser.add_argument('--dataframe_path', type=str, default=os.path.join('ilsvrc2012', 'imagenet_captions.parquet'))
    parser.add_argument('--index2filename_path', type=str, default=None)

    parser.add_argument('--index2wnid_path', type=str,
                        default=os.path.join('ilsvrc2012', 'processed', 'labels', 'icimagename2wnid.pkl'))

    parser.add_argument('--save_path', type=str,
                        default=os.path.join('laion400m', 'processed', 'clip_image_similarities'))

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

    # Saving
    os.makedirs(params['save_path'], exist_ok=True)

    prefix = os.path.split(params['dataframe_path'])[1].replace('.parquet', '_')
    wnid2savefilename = lambda w: prefix + f'img_img_sims({w}).pkl'

    print_verbose('done!\n')

    # ----- Init. CLIP -----
    print_verbose('init clip ...')

    clip = CLIP()

    print_verbose('done!\n')

    # ----- Load dataframe -----
    print_verbose('loading dataframe ...')

    df = pd.read_parquet(params['dataframe_path'])

    print_verbose('done!\n')

    # ----- Load labels -----
    print_verbose('loading labels ...')

    with open(os.path.join(params['index2wnid_path']), 'rb') as f:
        index2wnid = pickle.load(f)

    print_verbose('done!\n')

    # ----- Load the map to files -----
    print_verbose('loading index2filename map ...')

    if params['index2filename_path'] is None:
        print_verbose('\tdefaulting to identical map ...')
        index2filename = {idx: idx for idx in df.index}
    else:
        with open(params['index2filename_path'], 'rb') as f:
            index2filename = pickle.load(f)

    print_verbose('done!\n')

    # ----- Find the inverse map -----
    print_verbose('finding the inverse map ...')

    wnid2filenames = {}
    for idx, wnid in index2wnid.items():
        file_name = index2filename[idx]

        if wnid not in wnid2filenames:
            wnid2filenames[wnid] = []
        wnid2filenames[wnid].append(file_name)

    print_verbose('done!\n')

    # ----- Load and calc. similarity -----
    for wnid, file_names in tqdm(wnid2filenames.items(), desc='calc. cross image sim.'):
        # Load images
        images = []
        success_file_names = []
        for file_name in file_names:
            file_path = os.path.join(params['images_path'], file_name)
            image = Image.open(file_path)

            if image.mode != 'RGB':
                continue

            images.append(image)
            success_file_names.append(file_name)

        if len(images) <= 1:
            continue

        # Calc. similarities
        similarities = calc_image_cross_similarities(images, clip)

        # Save similarities
        with open(os.path.join(params['save_path'], wnid2savefilename(wnid)), 'wb') as f:
            pickle.dump(
                {
                    'index': success_file_names,
                    'similarities': similarities
                },
                f
            )
