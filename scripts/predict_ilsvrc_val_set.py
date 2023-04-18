import sys
import os
import time
import argparse
from PIL import Image
from io import BytesIO
import pandas as pd
from tqdm.auto import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

import configs
from utils import logging_utils as logu
from utils.logging_utils import print_verbose
from utils import pytorch_utils as ptu
from utils.ilsvrc_utils import load_lemmas_and_wnids
from utils import hugging_face_utils as hfu
from core.ilsvrc_predictors import ILSVRCPredictorType, select_ilsvrc_predictors


if __name__ == '__main__':
    # ----- Get arguments from input -----
    parser = argparse.ArgumentParser()

    # Path
    parser.add_argument('--images_path', type=str, default=os.path.join('ilsvrc2012', 'ILSVRC2012_img_val'))

    parser.add_argument('--ilsvrc_synsets_path', type=str, default=os.path.join('ilsvrc2012', 'ILSVRC2012_synsets.txt'))

    parser.add_argument('--save_path', type=str, default=os.path.join('ilsvrc2012', 'processed', 'predictions'))

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

    # Env
    ptu.init_gpu(use_gpu=not params['no_gpu'], gpu_id=params['gpu_id'])

    # Path
    os.makedirs(params['save_path'], exist_ok=True)

    print_verbose('done!\n')

    # ----- Select the models -----
    print_verbose('loading models ...')

    model_names, processors, models = select_ilsvrc_predictors([
        ILSVRCPredictorType.IMAGENET_1K,
        ILSVRCPredictorType.IMAGENET_PT21k_FT1K,
        ILSVRCPredictorType.IMAGENET_21K
    ])

    print_verbose('done!\n')

    # ----- Load ILSVRC labels -----
    print_verbose('loading ILSVRC lemmas and wnids ...')

    id_lemmas_df = load_lemmas_and_wnids(params['ilsvrc_synsets_path'])

    print_verbose('done!\n')

    # ----- Map model outputs to wnids -----
    model2label2wnid = {}
    for model_name in model_names:
        print_verbose(f'mapping {model_name} outputs to ilsvrc classes')

        model2label2wnid[model_name] = hfu.get_label2wnid_map(models[model_name], id_lemmas_df)

        print_verbose('done!\n')

    # ----- Collect downloads and predict -----
    images_batch = []
    indices_batch = []
    model2pred = {model_name: None for model_name in model_names}
    for idx in tqdm(range(1, 30 + 1)):
        # Load the image
        image = Image.open(os.path.join(params['images_path'], 'ILSVRC2012_val_%08d.JPEG' % idx))

        images_batch.append(image)
        indices_batch.append(idx)

        if len(images_batch) < configs.ILSVRCPredictorsConfig.BATCH_SIZE and idx < 30: # configs.ILSVRCConfigs.NUM_VAL:
            continue

        # Predict
        model2pred_batch = {}
        for model_name in model_names:
            model2pred_batch[model_name] = hfu.predict(
                processors[model_name],
                models[model_name],
                model2label2wnid[model_name],
                images_batch
            )

        # Append
        for model_name in model_names:
            model_df_i = pd.DataFrame(model2pred_batch[model_name], index=indices_batch)

            if model2pred[model_name] is None:
                model2pred[model_name] = model_df_i
            else:
                model2pred[model_name] = pd.concat((model2pred[model_name], model_df_i), axis=0)

        # Step
        images_batch = []
        indices_batch = []

    # ----- Save the successful predictions ------
    print_verbose('saving ....')

    time_str = time.strftime('%d-%m-%Y_%H-%M-%S')

    prefix = 'ilsvrc_val_set_'

    for model_name in model_names:
        pred_file_name = prefix + f'{model_name}_predictions_{time_str}.csv'
        model2pred[model_name].to_csv(os.path.join(params['save_path'], pred_file_name), index=True)

    print_verbose('done!\n')
