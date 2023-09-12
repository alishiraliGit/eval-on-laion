

from utils.wordnet_utils import get_synset


class QueryKey:
    WNID = 'wnid'  # Query will use wnid to load the synset and use synset name
    LEMMA = 'lemma'  # Query will use the given lemma in place of name

    @staticmethod
    def assert_query_key(query_key):
        assert query_key in [QueryKey.WNID, QueryKey.LEMMA], f'{query_key} is an invalid query_key!'


class QueryType:
    NAME = 'name'
    NAME_DEF = 'name_def'
    LEMMAS = 'lemmas'
    A_PHOTO_OF_NAME = 'a_photo_of_name'
    A_PHOTO_OF_NAME_DEF = 'a_photo_of_name_def'
    A_CLEAR_PHOTO_OF_NAME_DEF = 'a_clear_photo_of_name_def'
    A_PHOTO_OF_LEMMAS = 'a_photo_of_lemmas'


def select_queries(query_types):
    qs = []
    for q_type in query_types:
        if q_type == QueryType.NAME:
            qs.append(query_name)
        elif q_type == QueryType.NAME_DEF:
            qs.append(query_name_def)
        elif q_type == QueryType.LEMMAS:
            qs.append(query_lammas)
        elif q_type == QueryType.A_PHOTO_OF_NAME:
            qs.append(query_a_photo_of_name)
        elif q_type == QueryType.A_PHOTO_OF_NAME_DEF:
            qs.append(query_a_photo_of_name_def)
        elif q_type == QueryType.A_CLEAR_PHOTO_OF_NAME_DEF:
            qs.append(query_a_clear_photo_of_name_def)
        elif q_type == QueryType.A_PHOTO_OF_LEMMAS:
            qs.append(query_a_photo_of_lemmas)
        else:
            raise Exception(f'{q_type} is an invalid query type!')

    return qs


def query_name(wnid, lemma=None):
    if lemma is not None:
        name = lemma
    else:
        synset = get_synset(wnid)

        name = synset.name()
        name = name[:name.find('.n.')]
        name = name.replace('_', ' ')

    return name


def query_name_def(wnid, lemma=None):
    name = query_name(wnid, lemma)

    synset = get_synset(wnid)

    name_def = name + ' which is ' + synset.definition()

    return name_def


def query_lammas(wnid, lemma=None):
    # lemma argument just for compatibility
    synset = get_synset(wnid)

    lemmas = ' or '.join([lemma.name().replace('_', ' ') for lemma in synset.lemmas()])

    return lemmas


def query_a_photo_of_name(wnid, lemma=None):
    name = query_name(wnid, lemma)

    a_photo_of_name = 'a photo of ' + name

    return a_photo_of_name


def query_a_photo_of_name_def(wnid, lemma=None):
    name_def = query_name_def(wnid, lemma)

    a_photo_of_name_def = 'a photo of ' + name_def

    return a_photo_of_name_def


def query_a_clear_photo_of_name_def(wnid, lemma=None):
    name_def = query_name_def(wnid, lemma)

    a_photo_of_name_def = 'a clear photo of ' + name_def

    return a_photo_of_name_def


def query_a_photo_of_lemmas(wnid, lemma=None):
    lemmas = query_lammas(wnid, lemma)

    a_photo_of_lemmas = 'a photo of ' + lemmas

    return a_photo_of_lemmas
