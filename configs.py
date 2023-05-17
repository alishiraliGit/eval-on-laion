
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


class ILSVRCConfigs:
    LEMMAS_SEP = ', '
    WNID_COL = 'id'
    LEMMAS_COL = 'lemmas'
    NUM_WNID = 1000
    NUM_VAL = 50000


class ILSVRCPredictorsConfig:
    BATCH_SIZE = 32  # Used to be 64. Decreased for a better GPU memory management.


class LAIONConfig:
    URL_BASE = 'https://deploy.laion.ai/8f83b608504d46bb81708ec86e912220/dataset/'
    NUM_PARTS = 32
    SAFE_TAG = 'UNLIKELY'
    INDEX_SHIFT_PER_PART = 100000000
    ID_COL = 'SAMPLE_ID'
    URL_COL = 'URL'
    TEXT_COL = 'TEXT'
    NSFW_COL = 'NSFW'
    SIMILARITY_COL = 'similarity'

    SUBSET_SM_PREFIX = 'subset_sm_'
    SUBSET_SM_FILTERED_PREFIX = 'subset_sm_filt_'
    SUBSET_SM_CLEAR_PHOTO_FILTERED_PREFIX = 'subset_sm_clear_photo_filt_'
    SUBSET_SM_FILTERED_MOST_SIMILAR_IMG_IMG_PREFIX = 'subset_sm_filt_most_sim_imgimg_'
    SUBSET_VAL_MOST_SIMILAR_IMG_IMG_PREFIX = 'subset_val_most_sim_imgimg_'
    SUBSET_IC_MOST_SIMILAR_TXT_TXT_PREFIX = 'subset_ic_most_sim_txttxt_'
    SUBSET_IC_MOST_SIMILAR_TXT_TXT_WITH_SIMILAR_LABELS_PREFIX = 'subset_ic_most_sim_txttxt_sim_labels_'
    SUBSET_QUERIED_PREFIX = 'subset_queried_'
    PREDICTED_PREFIX = 'predicted_'

    @staticmethod
    def method_to_prefix(method):
        if method == 'substring_matched':
            prefix = LAIONConfig.SUBSET_SM_PREFIX
        elif method == 'substring_matched_filtered':
            prefix = LAIONConfig.SUBSET_SM_FILTERED_PREFIX
        elif method == 'substring_matched_clear_photo_filtered':
            prefix = LAIONConfig.SUBSET_SM_CLEAR_PHOTO_FILTERED_PREFIX
        elif method == 'substring_matched_filtered_most_similar_images':
            prefix = LAIONConfig.SUBSET_SM_FILTERED_MOST_SIMILAR_IMG_IMG_PREFIX
        elif method == 'ilsvrc_val_most_similar_images':
            prefix = LAIONConfig.SUBSET_VAL_MOST_SIMILAR_IMG_IMG_PREFIX
        elif method == 'imagenet_captions_most_similar_text_to_texts':
            prefix = LAIONConfig.SUBSET_IC_MOST_SIMILAR_TXT_TXT_PREFIX
        elif method == 'imagenet_captions_most_similar_text_to_texts_with_similar_labels':
            prefix = LAIONConfig.SUBSET_IC_MOST_SIMILAR_TXT_TXT_WITH_SIMILAR_LABELS_PREFIX
        elif method == 'queried':
            prefix = LAIONConfig.SUBSET_QUERIED_PREFIX
        else:
            raise Exception('Unknown method!')

        return prefix


class LAIONSamplingConfig:
    UNIFORM_SAMPLES = 500
    CLIP_SIMILARITY_RANGE = (0.3, 0.5)  # Not yet used


class RetrieveConfig:
    IMAGE_DOWNLOAD_TIMEOUT = 5
    MIN_IMAGE_SIZE = 10*10


class GDriveConfig:
    SCOPES = ['https://www.googleapis.com/auth/drive']
    EMB_FOLDER_ID = '1-s-Drt1IecCgodj7u7bwhEho7ZLY8CPa'
    IND_FOLDER_ID = '1-uuFos6XucFIRsU-7SdPn8uy65FYBFwu'
    CRED_FILE_NAME = 'client_secret_756745105460-2en3aa9elbhsni4k10u8o2utmc0l5mjn.apps.googleusercontent.com.json'


class CLIPRetrievalConfig:
    BACKEND_URL = 'https://knn.laion.ai/knn-service'
    URL_COL = 'url'
    TEXT_COL = 'caption'
    SIMILARITY_COL = 'similarity'
    ID_COL = 'id'
