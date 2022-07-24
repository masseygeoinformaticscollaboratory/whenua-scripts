import re
from copy import deepcopy

import gensim
import regex
import unidecode
from nltk.corpus import stopwords


stop_words_eng = stopwords.words('english')
stop_words_maori = [
    'i', 'o', 'ki', 'ka', 'ko', 'me', 'atu', 'kia', 'nei', 'ai', 'mō', 'kua', 'kei', 'ake', 'tonu', 'nō', 'mo', 'rāua',
    'ahakoa', 'mōna', 'te', 'tēnei', 'ētahi', 'tōna', 'tana', 'taku', 'ōna', 'aku', 'tōku', 'tāna', 'enei', 'toku',
    'era', 'tōu', 'tōnā', 'heoi', 'aue', 'ä', 'rātou', 'koutou', 'matou', 'kōrua', 'ngā', 'he', 'nga', 'e', 'a', 'anō',
    'nā', 'oku', 'tētahi', 'tērā', 'tetahi', 'kaua', 'kāhore', 'kīhai', 'ehara', 'ngātahi', 'rawa', 'rānei', 'tēnā',
    'mātou', 'iho', 'ēnei', 'tenei', 'kāore', 'no', 'na', 'tātou', 'ma', 'engari', 'aha', 'kē', 'ra', 'ano', 'ahau',
    'arā', 'ratou', 'pea', 'āhua', 'aua', 'rātau', 'āta', 'tona', 'ke', 'konei', 'koia', 'tatou', 'kaore', 'āna',
    'tētehi', 'ērā', 'etahi', 'ngä', 'tera', 'pēhea', 'ranei', 'mātau', 'raua', 'hai', 'katahi', 'otirā', 'kāre',
    'nāna', 'inā', 'anei', 'ētehi', 'kahore', 'ahua', 'ēngari', 'ōku', 'tāku', 'ināianei', 'māua', 'ē', 'ratau', 'tāua',
    'ngaa', 'otira', 'kāti', 'tē', 'aia', 'āu', 'whea', 'tetehi', 'māku', 'tö', 'mai', 'tā', 'töna', 'ana', 'nana',
    'au', 'kihai', 'torutoru', 'tāu', 'tëtahi', 'ā', 'hei', 'hoki', 'to', 'mā', 'rā', 'koe', 'katoa', 'tō', 'tino',
    'noa', 'taua', 'kore', 'ō', 'ta', 'ina', 'tä', 'tå'
]

stop_words = set(stop_words_eng + [unidecode.unidecode(x) for x in stop_words_maori])


def escape_text(text_raw, lower=True):
    text_raw = text_raw.strip()
    text_raw = re.sub('\n+', ' ', text_raw)
    text_raw = regex.sub(r'[^\p{Latin} ]', u'', text_raw)
    text_raw = re.sub(' +', ' ', text_raw)
    if lower:
        return text_raw.lower()
    return text_raw


def clean_exp_and_remove_stopwords(exp, deacc=True, lower=True, remove_stop_words=True):
    text_raw = escape_text(exp, lower)
    if deacc:
        text_raw = unidecode.unidecode(text_raw)
    if remove_stop_words:
        words = text_raw.split(' ')
        text_raw = ' '.join([word for word in words if word not in stop_words])
    return text_raw


def clean_exp_and_return_arr(exp, remove_stop_words=False):
    text_raw = escape_text(exp)
    tokens = gensim.utils.simple_preprocess(text_raw.lower(), deacc=True)
    if remove_stop_words:
        retval = []
        for token in tokens:
            if token not in stop_words:
                retval.append(token)

        return retval
    else:
        return tokens


def calc_freq(words, word_freqs):
    for word in words:
        if word in stop_words:
            continue
        if word in word_freqs:
            word_freqs[word] += 1
        else:
            word_freqs[word] = 1


def count_words(sentence):
    return len(escape_text(sentence).split(' '))


def calc_unigram_freq(words, word_freqs):
    for word in words:
        if word in word_freqs:
            word_freqs[word] += 1
        else:
            word_freqs[word] = 1


def calc_bigram_freq(words, word_freqs):
    for i in range(len(words) - 1):
        word1 = words[i]
        word2 = words[i+1]
        bigram = word1 + ' ' + word2

        if bigram in word_freqs:
            word_freqs[bigram] += 1
        else:
            word_freqs[bigram] = 1


def indices(lst, element):
    result = []
    offset = -1
    while True:
        try:
            offset = lst.index(element, offset+1)
            result.append(offset)
        except ValueError:
            break
    return result


def get_word_windows(keyword, original_words, window_length):
    searching_frame = deepcopy(original_words)
    searching_frame_offset = 0

    pairs = []
    while True:
        try:
            keyword_index = searching_frame.index(keyword)
        except ValueError:
            break

        keyword_index_inset = searching_frame_offset + keyword_index

        window_1_start = keyword_index_inset - window_length - 1
        if window_1_start < 0:
            window_1_start = 0

        window_1_ends = keyword_index_inset
        if window_1_ends < 0:
            window_1_ends = 0

        window_2_ends = keyword_index_inset + window_length
        window_2_ends = min(window_2_ends, len(original_words)) + 1

        window_1 = original_words[window_1_start:window_1_ends]
        window_2 = original_words[keyword_index_inset+1:window_2_ends]

        pair = window_1, window_2
        pairs.append(pair)

        searching_frame_offset += keyword_index + 1

        searching_frame = original_words[searching_frame_offset:]

    return pairs


def search_all_keywords(word_list, keywords, definitives):
    retval = {}
    for keyword in keywords:
        while True:
            try:
                index = word_list.index(keyword)
                if index > 0:
                    prev_word = word_list[index - 1]
                    if prev_word in definitives:
                        keyword_defs = retval.get(keyword, None)
                        if keyword_defs is None:
                            keyword_defs = {}
                            retval[keyword] = keyword_defs
                        keyword_defs[prev_word] = keyword_defs.get(prev_word, 0) + 1

                word_list = word_list[index + 1:]
            except ValueError:
                break
    return retval
