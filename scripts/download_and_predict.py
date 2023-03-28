import sys
import os
import multiprocessing
import time
import argparse
from PIL import Image
from io import BytesIO
import pandas as pd
import pickle
from tqdm.auto import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

import configs
import tools.laion_tools as lt
import tools.hugging_face_tools as hft
from retrieve import download_image_content, verify_image
from models import model_names, processors, models


def setup(ev):
    global unpaused
    unpaused = ev


def download_image_wrapper(args):
    idx, row = args
    try:
        image_content = download_image_content(row['URL'])
        verify_image(image_content)
        return idx, image_content
    except Exception as e:
        return -1, str(e)


def predict(args):
    idx_image_list, mdl2label2wnids = args

    # Load the images
    images = []
    indices = []
    for idx, image_content in idx_image_list:
        try:
            image = Image.open(BytesIO(image_content))
            images.append(image)
            indices.append(idx)
        except Exception:
            continue

    # Predict
    mdl2pred = {}
    try:
        for mdl_name in model_names:
            mdl2pred[mdl_name] = hft.predict(processors[mdl_name], models[mdl_name], mdl2label2wnids[mdl_name], images)
        return indices, mdl2pred
    except Exception as e:
        return [], str(e)


def num_ready_results(results):
    return sum([r.ready() for r in results])


if __name__ == '__main__':
    # ----- Get arguments from input -----
    parser = argparse.ArgumentParser()

    # Path
    parser.add_argument('--laion_path', type=str, default=os.path.join('laion400m'))
    parser.add_argument('--ilsvrc_path', type=str, default=os.path.join('ilsvrc2012'))
    parser.add_argument('--map_path', type=str, default=os.path.join('laion400m', 'processed'))
    parser.add_argument('--laion_until_part', type=int, default=31)

    # Multiprocessing
    parser.add_argument('--n_process_download', type=int, default=2)
    parser.add_argument('--n_process_predict', type=int, default=6)
    parser.add_argument('--pred_max_todo', type=int, default=1000)  # 1000 roughly corresponds to 4GB in RAM

    # Convert to dictionary
    settings = vars(parser.parse_args())

    # ----- Load data and maps -----
    # Load LAION sampled
    subset_file_name = \
        configs.LAIONConfig.SAMPLED_LABELED_PREFIX \
        + lt.get_laion_subset_file_name(0, settings['laion_until_part'])

    df = pd.read_parquet(os.path.join(settings['laion_path'], subset_file_name))

    # Load wnid2sampledlaionindices
    sampled_map_file_name = f'ILSVRC2012_wnid2sampledlaionindices.pkl'
    with open(os.path.join(settings['map_path'], sampled_map_file_name), 'rb') as f:
        wnid2laionindices = pickle.load(f)

    # Load ILSVRC labels and map model's outputs
    ilsvrc_classes_file_name = 'ILSVRC2012_synsets.txt'
    id_lemmas_df = pd.read_csv(
        os.path.join(settings['ilsvrc_path'], ilsvrc_classes_file_name),
        sep=': ',
        engine='python'
    )

    model2label2wnids = {}
    for model_name in tqdm(model_names, desc='mapping model outputs to desired classes'):
        model2label2wnids[model_name] = hft.get_label2wnids_map(models[model_name], id_lemmas_df, verbose=False)

    # ----- Parallel download and predict -----
    event = multiprocessing.Event()
    pool_download = multiprocessing.Pool(settings['n_process_download'], setup, (event,))
    pool_predict = multiprocessing.Pool(settings['n_process_predict'])

    # ----- Download and send for prediction -----
    download_results = pool_download.imap(download_image_wrapper, df.iterrows())
    event.set()

    download_ready_results = []
    prediction_results = []
    pred_pb = tqdm(desc='predicting', leave=True)
    latest_num_ready_results = 0
    for down_res in tqdm(download_results, desc='(mostly) downloading', total=len(df), leave=True):
        # Append the downloaded image to the batch
        download_ready_results.append(down_res)
        if len(download_ready_results) < configs.LAIONConfig.BATCH_SIZE:
            continue

        # Send the batch for prediction
        pred_res = pool_predict.apply_async(predict, [(download_ready_results, model2label2wnids)])
        prediction_results.append(pred_res)

        # Empty current batch
        download_ready_results = []

        # Monitor prediction progress
        pred_pb.total = len(prediction_results)
        pred_pb.update(num_ready_results(prediction_results) - latest_num_ready_results)
        latest_num_ready_results = num_ready_results(prediction_results)

        # Wait for predictions
        while len(prediction_results) - num_ready_results(prediction_results) > settings['pred_max_todo']:
            event.clear()
            time.sleep(1)
        event.set()

    # Send remaining for prediction
    pred_res = pool_predict.apply_async(predict, [(download_ready_results, model2label2wnids)])
    prediction_results.append(pred_res)

    # ----- Waiting for predictions -----
    pred_pb.total = len(prediction_results)
    pred_pb.update(num_ready_results(prediction_results) - latest_num_ready_results)
    latest_num_ready_results = num_ready_results(prediction_results)
    while latest_num_ready_results < len(prediction_results):
        pred_pb.update(num_ready_results(prediction_results) - latest_num_ready_results)
        latest_num_ready_results = num_ready_results(prediction_results)
        time.sleep(0.05)

    # ----- Close processes -----
    pred_pb.close()
    pool_download.close()
    pool_download.join()
    pool_predict.close()
    pool_predict.join()

    time.sleep(3)

    # ----- Collect the results ------
    model2pred = {model_name: None for model_name in model_names}
    errors = []
    for i_pred, pred_res in tqdm(enumerate(prediction_results), desc='concatenate results'):
        success_indices_i, model2pred_i = pred_res.get()

        if len(success_indices_i) == 0:
            continue

        for model_name in model_names:
            try:
                model_df_i = pd.DataFrame(model2pred_i[model_name], index=success_indices_i)
                if model2pred[model_name] is None:
                    model2pred[model_name] = model_df_i
                else:
                    model2pred[model_name] = pd.concat((model2pred[model_name], model_df_i), axis=0)
            except Exception as ex:
                errors.append(f'in concat. result {i_pred} of model {model_name}, following error happened:\n')
                errors.append(str(ex))
                errors.append('\n\n')

    # ----- Save the successful indices ------
    print('saving ....')

    for model_name in model_names:
        pred_file_name = f'ILSVRC2012_predictions[{model_name}]_{time.strftime("%d-%m-%Y_%H-%M-%S")}.csv'
        model2pred[model_name].to_csv(os.path.join(settings['map_path'], pred_file_name), index=True)

    err_file_name = f'ILSVRC2012_predictions_error_{time.strftime("%d-%m-%Y_%H-%M-%S")}.txt'
    with open(os.path.join(settings['map_path'], err_file_name), 'w') as f:
        f.write(''.join(errors))

    print('done!')
