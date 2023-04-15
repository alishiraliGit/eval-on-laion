import sys
import os
import argparse
import json
from tqdm import tqdm

from clip_retrieval.clip_client import ClipClient, Modality

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

import configs


if __name__ == '__main__':
    # ----- Get arguments from input -----
    parser = argparse.ArgumentParser()

    # Path
    parser.add_argument('--images_path', type=str, default=os.path.join('ilsvrc2012', 'ILSVRC2012_img_val'))

    parser.add_argument('--save_path', type=str, default=os.path.join('laion400m', 'processed', 'from_clip_retrieval'))

    # CLIP retrieval
    parser.add_argument('--clip_retrieval_index_name', type=str, default='laion_400m',
                        help='laion5B-L-14, laion5B-H-14, or laion_400m')

    # Size
    parser.add_argument('--top_k', type=int, default=50)

    # Convert to dictionary
    params = vars(parser.parse_args())

    # ----- Init. -----
    os.makedirs(params['save_path'], exist_ok=True)

    # ----- Init. the client -----
    client = ClipClient(
        url=configs.CLIPRetrievalConfig.BACKEND_URL,
        indice_name=params['clip_retrieval_index_name'],
        aesthetic_score=0,
        aesthetic_weight=0,
        modality=Modality.IMAGE,
        num_images=params['top_k'],
        deduplicate=True,
        use_safety_model=True,
        use_violence_detector=True
    )

    # ----- Retrieve -----
    all_results = {}
    for idx in tqdm(range(1, configs.ILSVRCConfigs.NUM_VAL + 1)):
        results = client.query(image=os.path.join(params['images_path'], 'ILSVRC2012_val_%08d.JPEG' % idx))

        all_results[idx] = results

    # ----- Save -----
    with open(os.path.join(params['save_path'],
                           f'top{params["top_k"]}_val_most_similars_from_{params["clip_retrieval_index_name"]}.json'
                           ), 'w') as f:
        json.dump(all_results, f)
