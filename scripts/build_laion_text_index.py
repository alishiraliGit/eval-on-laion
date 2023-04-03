import os
import sys
import glob
import numpy as np
from autofaiss import build_index

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

import configs


if __name__ == '__main__':
    # ----- Settings -----
    # Path
    embed_path = os.path.join('..', 'laion400m', 'processed', 'clip_text_embeddings')
    indices_path = os.path.join('..', 'laion400m', 'processed', 'clip_text_indices')

    faiss_index_folder = os.path.join('..', 'laion400m', 'processed', 'faiss_index')
    os.makedirs(faiss_index_folder, exist_ok=True)

    faiss_index_name = 'knn.index'
    faiss_infos_name = 'infos.json'

    # ----- Read and map all indices -----
    indices_file_paths = sorted(glob.glob(os.path.join(indices_path, 'indices*.npy')))

    indices = []
    for file_path in indices_file_paths:
        with open(file_path, 'rb') as f:
            # noinspection PyTypeChecker
            indices_i = np.load(f)

            indices.extend(indices_i)

    with open(os.path.join(indices_path, 'all_indices.npy'), 'wb') as f:
        # noinspection PyTypeChecker
        np.save(f, indices)

    # ----- Train and save the index -----
    build_index(
        embeddings=embed_path,
        index_path=os.path.join(faiss_index_folder, faiss_index_name),
        index_infos_path=os.path.join(faiss_index_folder, faiss_infos_name),
        save_on_disk=True,
        file_format='npy',
        max_index_memory_usage=configs.AutoFaissConfig.MAX_INDEX_MEMORY,
        min_nearest_neighbors_to_retrieve=configs.AutoFaissConfig.MIN_NN,
        metric_type=configs.CLIPConfig.METRIC_TYPE
    )
