
class WordNetConfig:
    WELL_KNOWN_HYPERNYM_MIN_COUNT = 2


class CLIPConfig:
    DEFAULT_VERSION = 'clip-vit-base-patch32'
    MAX_SEQ_LENGTH = 77  # Based on my observations, num of tokens shouldn't exceed 77. Currently, useless.
    BATCH_SIZE = 64
    DIM = 512
    REPLACE_NA_STR = 'None'
    METRIC_TYPE = 'ip'


class AutoFaissConfig:
    MIN_NN = 50
    MAX_INDEX_MEMORY = '%dMB' % int(1e6 / 4e8 * 16000)
    CURRENT_MEMORY = '8GB'


class ILSVRCPredictorsConfig:
    BATCH_SIZE = 64


class LAIONConfig:
    URL_BASE = 'https://deploy.laion.ai/8f83b608504d46bb81708ec86e912220/dataset/'
    NUM_PARTS = 32
    SAFE_TAG = 'UNLIKELY'
    INDEX_SHIFT_PER_PART = 100000000
    LABELED_PREFIX = 'labeled_'
    SUBSET_PREFIX = 'subset_'
    SUBSET_QUERIED_PREFIX = 'subset_queried_'


class LAIONSamplingConfig:
    UNIFORM_SAMPLES = 1000
    SIMILARITY_BINS_RANGE = (0.3, 0.5)
    SIMILARITY_BINS_DELTA_P = 0.2
    SAMPLES_PER_SIMILARITY_BIN = 50


class RetrieveConfig:
    IMAGE_DOWNLOAD_TIMEOUT = 5
    MIN_IMAGE_SIZE = 10*10


class GDriveConfig:
    SCOPES = ['https://www.googleapis.com/auth/drive']
    EMB_FOLDER_ID = '1-s-Drt1IecCgodj7u7bwhEho7ZLY8CPa'
    IND_FOLDER_ID = '1-uuFos6XucFIRsU-7SdPn8uy65FYBFwu'
    CRED_FILE_NAME = 'client_secret_756745105460-2en3aa9elbhsni4k10u8o2utmc0l5mjn.apps.googleusercontent.com.json'
