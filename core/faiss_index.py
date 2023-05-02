import os
import pickle

from autofaiss import build_index
import faiss
import numpy as np

import configs
from utils.logging_utils import print_verbose


class FaissIndex:
    def __init__(self, faiss_index, indices):
        self.faiss_index = faiss_index
        self.indices = indices

    def search(self, embeds, k):
        # Search the index
        cos_sims, faiss_indices = self.faiss_index.search(embeds, k)

        # Map the indices
        mapped_indices = np.array([self.indices[f_indices_i] for f_indices_i in faiss_indices])

        return mapped_indices, cos_sims

    def update(self, new_embeds, new_indices):
        print('updating faiss index ...')

        # Add to the index
        self.faiss_index.add(new_embeds)

        # Add to the indices
        self.indices = np.append(self.indices, new_indices)

        print_verbose('done!\n')

    def save(self, faiss_index_path, indices_path):
        print_verbose('saving faiss index ...')

        # Save the index
        faiss.write_index(self.faiss_index, faiss_index_path)

        # Save the indices
        with open(indices_path, 'wb') as f:
            pickle.dump(self.indices, f)

        print_verbose('done!\n')

    @staticmethod
    def load(faiss_index_path, indices_path):
        # Load indices
        print_verbose('loading indices ...')

        with open(indices_path, 'rb') as f:
            all_indices = pickle.load(f)

        print_verbose(f'\tfound {len(all_indices)} rows in all_indices.')

        print_verbose('done!\n')

        # Load faiss index
        print_verbose('loading faiss index ...')

        faiss_index = faiss.read_index(faiss_index_path)

        print_verbose(f'\tfound {faiss_index.ntotal} rows in faiss index.')

        print_verbose('done!\n')

        # Check there are equal indices and entries
        assert len(all_indices) == faiss_index.ntotal

        # Instantiate
        return FaissIndex(faiss_index, all_indices)

    @staticmethod
    def build_index(
            embeddings_folder_path, index_folder_path,
            postfix='',
            max_index_memory_usage=configs.AutoFaissConfig.MAX_INDEX_MEMORY,
            min_nearest_neighbors_to_retrieve=configs.AutoFaissConfig.MIN_NN,
            safe=True
    ):
        faiss_index_path = os.path.join(index_folder_path, f'knn{postfix}.index')
        index_infos_path = os.path.join(index_folder_path, f'infos{postfix}.json')
        if safe:
            if os.path.exists(faiss_index_path) or os.path.exists(index_infos_path):
                raise Exception('A knn index or infos with the same name exists.')

        build_index(
            embeddings=embeddings_folder_path,
            index_path=faiss_index_path,
            index_infos_path=index_infos_path,
            save_on_disk=True,
            file_format='npy',
            metric_type=configs.CLIPConfig.METRIC_TYPE,
            max_index_memory_usage=max_index_memory_usage,
            min_nearest_neighbors_to_retrieve=min_nearest_neighbors_to_retrieve
        )
