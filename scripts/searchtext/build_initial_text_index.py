import os
import sys
import glob
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

from core.faiss_index import FaissIndex


if __name__ == '__main__':
    # ----- Settings -----
    # Path
    embed_path = os.path.join('..', '..', 'laion400m', 'processed', 'clip_text_embeddings')
    indices_path = os.path.join('..', '..', 'laion400m', 'processed', 'clip_text_indices')

    faiss_index_folder = os.path.join('..', '..', 'laion400m', 'processed', 'faiss_index')
    os.makedirs(faiss_index_folder, exist_ok=True)

    postfix = ''  # Use this for versioning

    # ----- Read and map all indices -----
    indices_file_paths = sorted(glob.glob(os.path.join(indices_path, 'indices*.npy')))

    indices = []
    for file_path in indices_file_paths:
        with open(file_path, 'rb') as f:
            # noinspection PyTypeChecker
            indices_i = np.load(f)

            indices.extend(indices_i)

    with open(os.path.join(indices_path, f'all_indices{postfix}.npy'), 'xb') as f:
        # noinspection PyTypeChecker
        np.save(f, indices)

    # ----- Train and save the index -----
    FaissIndex.build_index(
        embeddings_folder_path=embed_path,
        index_folder_path=faiss_index_folder,
        postfix=postfix
    )
