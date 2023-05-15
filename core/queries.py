

from utils.wordnet_utils import get_synset


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
            raise Exception('Invalid query type!')

    return qs


def query_name(wnid):
    synset = get_synset(wnid)

    name = synset.name()
    name = name[:name.find('.n.')]
    name = name.replace('_', ' ')

    return name


def query_name_def(wnid):
    synset = get_synset(wnid)

    name = query_name(wnid)

    name_def = name + ' which is ' + synset.definition()

    return name_def


def query_lammas(wnid):
    synset = get_synset(wnid)

    lemmas = ' or '.join([lemma.name().replace('_', ' ') for lemma in synset.lemmas()])

    return lemmas


def query_a_photo_of_name(wnid):
    name = query_name(wnid)

    a_photo_of_name = 'a photo of ' + name

    return a_photo_of_name


def query_a_photo_of_name_def(wnid):
    name_def = query_name_def(wnid)

    a_photo_of_name_def = 'a photo of ' + name_def

    return a_photo_of_name_def


def query_a_clear_photo_of_name_def(wnid):
    name_def = query_name_def(wnid)

    a_photo_of_name_def = 'a clear photo of ' + name_def

    return a_photo_of_name_def


def query_a_photo_of_lemmas(wnid):
    lemmas = query_lammas(wnid)

    a_photo_of_lemmas = 'a photo of ' + lemmas

    return a_photo_of_lemmas
