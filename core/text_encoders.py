import configs
from core.foundationmodels.clip import CLIP
from core.foundationmodels.bert import Bert
from core.foundationmodels.sentencetransformer import SentTransformer

encoder_ver = None
encoder = None
encoder_batch_size = None


def select_text_encoder(ver:  str):
    global encoder_ver, encoder, encoder_batch_size

    if ver == encoder_ver:
        return encoder, encoder_batch_size

    if ver.startswith('clip'):
        clip = CLIP(ver=ver)

        encoder = clip.text_embeds
        encoder_ver = ver
        encoder_batch_size = configs.CLIPConfig.BATCH_SIZE

    elif ver.startswith('bert'):
        bert = Bert(ver=ver)

        encoder = bert.text_embeds
        encoder_ver = ver
        encoder_batch_size = configs.BertConfig.BATCH_SIZE

    else:
        sent_trans = SentTransformer(ver=ver)

        encoder = sent_trans.text_embeds
        encoder_ver = ver
        encoder_batch_size = configs.SentTransformer.BATCH_SIZE

    return encoder, encoder_batch_size
