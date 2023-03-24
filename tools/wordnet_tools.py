import nltk
from nltk.corpus import wordnet as wn

import configs

nltk.download('wordnet')


def lemma_is_unique(lemma: str):
    lemma_u = lemma.replace(' ', '_')

    found_lemmas = wn.lemmas(lemma_u)

    if len(found_lemmas) == 0:
        raise Exception('lemma not found!')

    return len(found_lemmas) == 1


def get_synset(wnid):
    if wnid == 'n02112837':
        return wn.synsets('siberian_husky')[0]
    else:
        return wn.synset_from_pos_and_offset(wnid[0], int(wnid[1:]))


def synset_count(synset):
    return sum([lemma.count() for lemma in synset.lemmas()])


def get_most_frequent_lemma(synset):
    lemma_count = {lemma: lemma.count() for lemma in synset.lemmas()}
    return max(lemma_count, key=lambda k: lemma_count[k])


def get_well_known_hypernym(synset):
    synset = synset.hypernyms()[0]
    while synset_count(synset) < configs.WordNetConfig.WELL_KNOWN_HYPERNYM_MIN_COUNT:
        synset = synset.hypernyms()[0]

    return synset


def extend_with_well_known_hypernym(lemma, wnid):
    synset = get_synset(wnid)

    hypernym = get_well_known_hypernym(synset)

    ex = get_most_frequent_lemma(hypernym).name().replace('_', ' ')

    if ex in lemma:
        return lemma
    else:
        return lemma + ' ' + ex