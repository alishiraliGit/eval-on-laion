import sys
import os
import multiprocessing
import time
import argparse
from PIL import Image
from io import BytesIO
import pandas as pd
from tqdm.auto import tqdm
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

import configs
from utils import logging_utils as logu
from utils.logging_utils import print_verbose
from utils import pytorch_utils as ptu
from utils.ilsvrc_utils import load_lemmas_and_wnids
from utils import laion_utils as laionu
from utils import hugging_face_utils as hfu
from core.retrieve_image import download_image_content, verify_image
from core.ilsvrc_predictors import ILSVRCPredictorType, select_ilsvrc_predictors


unpaused = None


def setup(ev):
    global unpaused
    unpaused = ev


def download_image_wrapper(args):
    idx, row = args
    try:
        img_content = download_image_content(row[configs.LAIONConfig.URL_COL])
        verify_image(img_content)
        return idx, img_content, None
    except Exception as e:
        return idx, None, {'cause': f'In downloading image of index {idx} an error occurred.', 'error': e}


def predict(args):
    idx_img_list, mdl_names, ps, mdls, mdl2label2wnid, device = args

    # Init.
    ptu.device = device
    # torch.cuda.set_per_device_memory_fraction(device, 0.2)

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

    # Predict
    mdl2pred = {}
    try:
        for mdl_name in mdl_names:
            mdl2pred[mdl_name] = hfu.predict(ps[mdl_name], mdls[mdl_name], mdl2label2wnid[mdl_name], imgs)
        return inds, mdl2pred, errs
    except Exception as e:
        errs.append({'cause': 'In predicting labels of a batch of images an error occurred.', 'error': e})
        return inds, None, errs


def num_ready_results(results):
    return sum([r.ready() for r in results])


latest_num_ready_results = 0


def update_pred_pb(pred_pb, pred_results):
    global latest_num_ready_results

    pred_pb.total = len(pred_results)
    pred_pb.update(num_ready_results(pred_results) - latest_num_ready_results)
    latest_num_ready_results = num_ready_results(pred_results)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    # ----- Get arguments from input -----
    parser = argparse.ArgumentParser()

    # Path
    parser.add_argument('--laion_path', type=str, default=os.path.join('laion400m'))
    parser.add_argument('--laion_until_part', type=int, default=31)

    parser.add_argument('--ilsvrc_synsets_path', type=str, default=os.path.join('ilsvrc2012', 'ILSVRC2012_synsets.txt'))

    parser.add_argument('--save_path', type=str, default=os.path.join('laion400m', 'processed', 'ilsvrc_predictions'))

    # Method
    parser.add_argument('--queried_clip_retrieval', action='store_true')
    parser.add_argument('--queried', action='store_true')

    # Multiprocessing
    parser.add_argument('--n_process_download', type=int, default=2)
    parser.add_argument('--n_process_predict', type=int, default=6)
    parser.add_argument('--pred_max_todo', type=int, default=1000)  # 1000 roughly corresponds to 4GB in RAM

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

    # ----- Load LAION subset -----
    print_verbose('loading laion subset ...')

    if params['queried_clip_retrieval']:
        prefix = configs.LAIONConfig.SUBSET_CLIP_RETRIEVAL_PREFIX
    else:
        prefix = configs.LAIONConfig.SUBSET_QUERIED_PREFIX if params['queried'] else configs.LAIONConfig.SUBSET_PREFIX

    subset_file_name = prefix + laionu.get_laion_subset_file_name(0, params['laion_until_part'])

    # TODO
    df = pd.read_parquet(os.path.join(params['laion_path'], subset_file_name)).iloc[:1000]

    print_verbose('done!\n')

    # ----- Init. parallel download and predict -----
    event = multiprocessing.Event()
    pool_download = multiprocessing.Pool(params['n_process_download'], setup, (event,))
    pool_predict = multiprocessing.Pool(params['n_process_predict'])

    # ----- Start download -----
    download_results = pool_download.imap(download_image_wrapper, df.iterrows())
    event.set()

    # ----- Collect downloads and predict -----
    download_ready_results = []
    prediction_results = []
    errors = []

    prediction_pb = tqdm(desc='predicting', leave=True)
    for down_res in tqdm(download_results, desc='(mostly) downloading', total=len(df), leave=True):
        # Catch the errors in downloading
        index, image_content, error = down_res
        if error is not None:
            errors.append('\n' + error['cause'])
            errors.append(str(error['error']))
            continue

        # Append the downloaded image to the batch
        download_ready_results.append([index, image_content])
        if len(download_ready_results) < configs.ILSVRCPredictorsConfig.BATCH_SIZE:
            continue

        # Send the batch for prediction
        pred_res = pool_predict.apply_async(
            predict,
            [(
                download_ready_results,
                model_names,
                processors,
                models,
                model2label2wnid,
                ptu.device
            )]
        )
        prediction_results.append(pred_res)

        # Empty current batch
        download_ready_results = []

        # Monitor prediction progress
        update_pred_pb(prediction_pb, prediction_results)

        # Wait for predictions
        while len(prediction_results) - num_ready_results(prediction_results) > params['pred_max_todo']:
            event.clear()
            time.sleep(1)
        event.set()

    # Send the remaining for prediction
    pred_res = pool_predict.apply_async(
        predict,
        [(
            download_ready_results,
            model_names,
            processors,
            models,
            model2label2wnid,
            ptu.device
        )]
    )
    prediction_results.append(pred_res)

    # ----- Waiting for predictions -----
    update_pred_pb(prediction_pb, prediction_results)
    while latest_num_ready_results < len(prediction_results):
        update_pred_pb(prediction_pb, prediction_results)
        time.sleep(0.05)

    # ----- Close progress bars and processes -----
    prediction_pb.close()

    pool_download.close()
    pool_download.join()

    pool_predict.close()
    pool_predict.join()

    time.sleep(3)

    # ----- Collect the results ------
    model2pred = {model_name: None for model_name in model_names}
    for i_pred, pred_res in tqdm(enumerate(prediction_results), desc='concatenate results'):
        success_indices_i, model2pred_i, pred_errors_i = pred_res.get()

        # Append errors
        for error in pred_errors_i:
            errors.append('\n' + error['cause'])
            errors.append(str(error['error']))

        # Check for fatal error
        if model2pred_i is None:
            continue

        # Append predictions
        for model_name in model_names:
            try:
                model_df_i = pd.DataFrame(model2pred_i[model_name], index=success_indices_i)
                if model2pred[model_name] is None:
                    model2pred[model_name] = model_df_i
                else:
                    model2pred[model_name] = pd.concat((model2pred[model_name], model_df_i), axis=0)
            except Exception as ex:
                errors.append('\n' + f'In concat. result {i_pred} of model {model_name}, an error happened.')
                errors.append(str(ex))

    # ----- Save the successful predictions ------
    print_verbose('saving ....')

    time_str = time.strftime('%d-%m-%Y_%H-%M-%S')

    err_file_name = prefix + f'errors_{time_str}.txt'
    with open(os.path.join(params['save_path'], err_file_name), 'w') as f:
        f.write('\n'.join(errors))

    for model_name in model_names:
        pred_file_name = prefix + f'{model_name}_predictions_{time_str}.csv'
        model2pred[model_name].to_csv(os.path.join(params['save_path'], pred_file_name), index=True)

    print_verbose('done!\n')
