import configs
from core.foundationmodels.clip import CLIP
from core.foundationmodels.bert import Bert
from core.foundationmodels.sentencetransformer import SentTransformer

txt_encoder_ver = None
txt_encoder = None
txt_encoder_batch_size = None


def select_text_encoder(ver:  str):
    global txt_encoder_ver, txt_encoder, txt_encoder_batch_size

    if ver == txt_encoder_ver:
        return txt_encoder, txt_encoder_batch_size

    if ver.startswith('clip'):
        clip = CLIP(ver=ver)

        txt_encoder = clip.text_embeds
        txt_encoder_ver = ver
        txt_encoder_batch_size = configs.CLIPConfig.BATCH_SIZE

    elif ver.startswith('bert'):
        bert = Bert(ver=ver)

        txt_encoder = bert.text_embeds
        txt_encoder_ver = ver
        txt_encoder_batch_size = configs.BertConfig.BATCH_SIZE

    else:
        sent_trans = SentTransformer(ver=ver)

        txt_encoder = sent_trans.text_embeds
        txt_encoder_ver = ver
        txt_encoder_batch_size = configs.SentTransformer.BATCH_SIZE

    return txt_encoder, txt_encoder_batch_size
