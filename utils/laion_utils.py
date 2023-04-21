import os
import string
import urllib.request
import pandas as pd

import configs
from utils.logging_utils import print_verbose


def get_laion_part_file_name(part):
    return 'part-%05d-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet' % part


def get_laion_subset_file_name(from_part, to_part):
    return 'part-%05d-to-part%05d-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet' % (from_part, to_part)


def download_laion_part(part, laion_path):
    url_base = configs.LAIONConfig.URL_BASE
    urllib.request.urlretrieve(
        url=url_base + get_laion_part_file_name(part),
        filename=os.path.join(laion_path, get_laion_part_file_name(part))
    )


def map_index(idx, part):
    return idx + part*configs.LAIONConfig.INDEX_SHIFT_PER_PART


def imap_index(idx):
    part = idx // configs.LAIONConfig.INDEX_SHIFT_PER_PART
    original_index = idx % configs.LAIONConfig.INDEX_SHIFT_PER_PART

    return part, original_index


def rename_index(df, part):
    return df.rename(mapper=lambda idx: map_index(idx, part))


def load_data_part(laion_path, laion_part, self_destruct):
    print_verbose(f'loading laion part {laion_part} ...')

    # Download if required
    laion_file_path = os.path.join(laion_path, get_laion_part_file_name(laion_part))
    if not os.path.exists(laion_file_path):
        print_verbose(f'\tdownloading laion part {laion_part} ...')

        download_laion_part(part=laion_part, laion_path=laion_path)

        print_verbose('\tdownloaded!')

    # Load LAION part
    part_df = pd.read_parquet(laion_file_path)

    # Self-destruct
    if self_destruct:
        print_verbose(f'\tremoving laion part {laion_part} from the disk ...')

        os.remove(laion_file_path)

        print_verbose('\tremoved!')

    # Reindex
    part_df = rename_index(part_df, laion_part)

    print_verbose('done!\n')

    return part_df


translation = str.maketrans(string.punctuation, ' ' * len(string.punctuation))


def transform_lemma(lemma):
    # Extend the lemma to make sure 'paid' is not a kind of 'ai'!
    return ' ' + lemma + ' '


def transform_text(txt):
    return ' ' + txt.translate(translation) + ' '
