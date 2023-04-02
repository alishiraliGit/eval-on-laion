
class WordNetConfig:
    WELL_KNOWN_HYPERNYM_MIN_COUNT = 2


class CLIPConfig:
    DEFAULT_VERSION = 'clip-vit-base-patch32'
    MAX_SEQ_LENGTH = 77  # Based on my observations, num of tokens shouldn't exceed 77. Currently, useless.
    BATCH_SIZE = 32  # 512
    DIM = 512


class ImageNetModelsConfig:
    BATCH_SIZE = 64


class LAIONConfig:
    URL_BASE = 'https://deploy.laion.ai/8f83b608504d46bb81708ec86e912220/dataset/'
    NUM_PARTS = 32
    INDEX_SHIFT_PER_PART = 100000000
    LABELED_PREFIX = 'labeled_'
    SAMPLED_LABELED_PREFIX = 'sampled_labeled_'
    SAFE_TAG = 'UNLIKELY'


class LAIONSamplingConfig:
    UNIFORM_SAMPLES = 200
    SIMILARITY_BINS_RANGE = (0.3, 0.5)
    SIMILARITY_BINS_DELTA_P = 0.2
    SAMPLES_PER_SIMILARITY_BIN = 50


class RetrieveConfig:
    IMAGE_DOWNLOAD_TIMEOUT = 5
    MIN_IMAGE_SIZE = 10*10
