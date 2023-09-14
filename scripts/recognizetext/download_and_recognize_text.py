import sys
import os
import multiprocessing
import time
import argparse
import pandas as pd
from tqdm.auto import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

import configs
from utils import logging_utils as logu
from utils.logging_utils import print_verbose
from utils import pytorch_utils as ptu
from utils import laion_utils as laionu
from core.text_detectors import EAST
from core.ocr import TrOCR


###############
# Image download
###############

from scripts.predict.download_and_predict import unpaused, setup, download_image_wrapper


###############
# Text detection and recognition
###############

text_detector: EAST = None
text_recognizer: TrOCR = None


def load_models():
    global text_detector, text_recognizer

    if text_detector is None:
        text_detector = EAST()

    if text_recognizer is None:
        text_recognizer = TrOCR()

    return text_detector, text_recognizer


def init_worker(pars):
    global text_detector, text_recognizer

    worker_id = multiprocessing.current_process().name

    logu.verbose = pars['verbose']

    # Init
    print_verbose(f'initializing worker {worker_id} ...')

    ptu.init_gpu(use_gpu=not pars['no_gpu'], gpu_id=pars['gpu_id'])

    print_verbose('done!\n')

    # Load the models
    print_verbose(f'loading text detector and recognizer in worker {worker_id} ...')

    load_models()

    print_verbose('done!\n')


def detect_and_recognize(args):
    global text_detector, text_recognizer

    idx_img_list = args

    # Load the images
    txts = []
    inds = []
    errs = []
    for idx, img_content in idx_img_list:
        try:
            _, img_boxes = text_detector.detect_and_select_boxes(img_content, draw_box=False)

            if len(img_boxes) == 0:
                continue

            txt = text_recognizer.recognize(img_boxes)

            inds.append(idx)
            txts.append(txt)

        except Exception as e:
            errs.append({
                'cause': 'In detection or recognizing texts of a batch of images an error occurred.',
                'error': e}
            )

    return inds, txts, errs


def dummy_func(_args):
    return None


###############
# Progress monitor
###############

def num_ready_results(results):
    return sum([r.ready() for r in results])


latest_num_ready_results = 0


def update_recognition_pb(rec_pb, rec_results):
    global latest_num_ready_results

    rec_pb.total = len(rec_results)
    rec_pb.update(num_ready_results(rec_results) - latest_num_ready_results)
    latest_num_ready_results = num_ready_results(rec_results)


if __name__ == '__main__':
    # multiprocessing.set_start_method('spawn')

    # ----- Get arguments from input -----
    parser = argparse.ArgumentParser()

    # Path
    parser.add_argument('--laion_path', type=str, default=os.path.join('laion400m'))
    parser.add_argument('--laion_until_part', type=int, default=31)
    parser.add_argument('--prefix', type=str, help='Look at configs.NamingConfig for conventions.')

    parser.add_argument('--save_path', type=str, default=os.path.join('laion400m', 'processed', 'text_recognition'))

    # Subset
    parser.add_argument('--from_iloc', type=int, default=0)
    parser.add_argument('--to_iloc', type=int, default=-1)

    # Multiprocessing
    parser.add_argument('--n_process_download', type=int, default=2)
    parser.add_argument('--n_process_recognition', type=int, default=6)
    parser.add_argument('--recognition_max_todo', type=int, default=1000)  # 1000 roughly corresponds to 4GB in RAM

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

    # Path
    os.makedirs(params['save_path'], exist_ok=True)

    # Prefix
    prefix = params['prefix']

    print_verbose('done!\n')

    # Recognized text column
    rec_text_col = 'recognized_text'

    # Download EAST if required
    EAST.download()

    # ----- Load LAION subset -----
    print_verbose('loading laion subset ...')

    subset_file_name = prefix + '_' + laionu.get_laion_subset_file_name(0, params['laion_until_part'])
    subset_file_path = os.path.join(params['laion_path'], subset_file_name)

    df_all = pd.read_parquet(subset_file_path)

    # Subset
    if params['to_iloc'] > 0:
        df = df_all.iloc[params['from_iloc']: params['to_iloc']]
    else:
        df = df_all.iloc[params['from_iloc']:]

    print_verbose('done!\n')

    # ----- Init. parallel download and predict -----
    event = multiprocessing.Event()
    pool_download = multiprocessing.Pool(params['n_process_download'], setup, (event,))
    pool_recognition = multiprocessing.Pool(params['n_process_recognition'], initializer=init_worker, initargs=(params,))

    # Init pool_recognition
    pool_recognition.map(dummy_func, [1] * params['n_process_recognition'])

    # ----- Start download -----
    download_results = pool_download.imap(download_image_wrapper, df.iterrows())
    event.set()

    # ----- Collect downloads and predict -----
    download_ready_results = []
    recognition_results = []
    errors = []

    recognition_pb = tqdm(desc='recognizing', leave=True)
    for down_res in tqdm(download_results, desc='(mostly) downloading', total=len(df), leave=True):
        # Catch the errors in downloading
        index, image_content, error = down_res
        if error is not None:
            errors.append('\n' + error['cause'])
            errors.append(str(error['error']))
            continue

        # Append the downloaded image to the batch
        download_ready_results.append([index, image_content])
        if len(download_ready_results) < configs.EASTConfig.BATCH_SIZE:
            continue

        # Send the batch for prediction
        rec_res = pool_recognition.apply_async(
            detect_and_recognize,
            [download_ready_results]
        )
        recognition_results.append(rec_res)

        # Empty current batch
        download_ready_results = []

        # Monitor prediction progress
        update_recognition_pb(recognition_pb, recognition_results)

        # Wait for predictions
        while len(recognition_results) - num_ready_results(recognition_results) > params['recognition_max_todo']:
            event.clear()
            time.sleep(1)
        event.set()

    # Send the remaining for prediction
    rec_res = pool_recognition.apply_async(
        detect_and_recognize,
        [download_ready_results]
    )
    recognition_results.append(rec_res)

    # ----- Waiting for predictions -----
    update_recognition_pb(recognition_pb, recognition_results)
    while latest_num_ready_results < len(recognition_results):
        update_recognition_pb(recognition_pb, recognition_results)
        time.sleep(0.05)

    # ----- Close progress bars and processes -----
    recognition_pb.close()

    pool_download.close()
    pool_download.join()

    pool_recognition.close()
    pool_recognition.join()

    time.sleep(3)

    # ----- Collect the results ------
    for i_pred, rec_res in tqdm(enumerate(recognition_results), desc='adding recognized texts to df'):
        success_indices_i, texts_i, rec_errors_i = rec_res.get()

        # Append errors
        for error in rec_errors_i:
            errors.append('\n' + error['cause'])
            errors.append(str(error['error']))

        # Check for fatal error
        if len(success_indices_i) == 0:
            continue

        # Add texts to the df
        df_all.loc[success_indices_i, rec_text_col] = texts_i

    # ----- Save the successful predictions ------
    print_verbose('saving ....')

    ver = time.strftime('%d-%m-%Y_%H-%M-%S')

    err_file_path = subset_file_path.replace('parquet', 'textrecognition_errors.txt')
    with open(err_file_path, 'w') as f:
        f.write('\n'.join(errors))

    df_all.to_parquet(subset_file_path, index=True)

    print_verbose('done!\n')
