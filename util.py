import re

import gensim
import regex
import unidecode
from nltk.corpus import stopwords

stop_words = stopwords.words('english')
stop_words.extend(
    ['e', 'he', 'tetahi', 'amo', 'ano', 'ki', 'katoa', 'tetahi', 'a', 'hoki', 'kaore', 'kahore', 'kare', 'kore',
     'ehara', 'hei', 'kei', 'i', 'i te mea', 'na te mea', 'no te meaki te mea', 'kua', 'mua', 'raua tahi', 'engari',
     'I', 'kihai', 'kaore', 'kare', 'kahore', 'raro', 'ia', 'torutoru', 'mo', 'ma', 'ia', 'ia', 'konei', 'anei', 'ina',
     'ana', 'ona', 'ia ano', 'ia', 'ia ano', 'tana', 'ana', 'tona ', 'ona', 'pehea', 'au', 'ahau', 'maku', 'mehemea',
     'mena', 'me', 'me', 'au', 'ahau', 'ano', 'nuinga', 'taku', 'taku', 'aku', 'toku', 'oku', 'au ano', 'ahau ano',
     'kao', 'kao', 'ehara', 'o', 'a', 'kotahi', 'noa iho', 'anake', 'ranei', 'atu', 'to taua', 'to tatou', 'to tatau',
     'to maua', 'to matou', 'o taua', 'o tatou', 'o tatau', 'o maua', 'o matou', 'taua ano', 'tatou ano', 'tatau ano',
     'maua ano', 'matou ano matau ano', 'orite', 'ia', 'me', 'me kaua', 'na reira', 'no reira', 'na', 'na', 'etahi',
     'he', 'i', 'tena', 'tera', 'taua', 'koia', 'ko ia', 'koina', 'koia tena', 'koira', 'te', 'nga', 'ta raua',
     'ta ratou', 'to raua', 'to ratou', 'to ratau', 'a raua', 'a ratou', 'o raua', 'o ratou', 'o ratau', 'raua ',
     'ratou', 'ratau', 'rau ano', 'ratou ano', 'ratau ano', 'kona', 'kora', 'ko', 'enei', 'tenei', 'ena', 'era', 'hoki',
     'raro', 'tae noa', 'kia', 'runga', 'tino', 'maua', 'matou', 'taua', 'tatou', 'tatau', 'aha', 'ina', 'kia', 'ka',
     'no te', 'hea', 'tehea', 'ehea', 'ko wai', 'na wai', 'ma wai', 'na wai', 'no wai', 'he aha ai', 'e kore', 'koe',
     'tou', 'tau', 'ou', 'au', 'koe ano', 'koutou ano'])


def escape_text(text_raw, lower=True):
    text_raw = text_raw.strip()
    text_raw = re.sub('\n+', ' ', text_raw)
    text_raw = regex.sub(r'[^\p{Latin} ]', u'', text_raw)
    text_raw = re.sub(' +', ' ', text_raw)
    if lower:
        return text_raw.lower()
    return text_raw


def clean_exp_and_remove_stopwords(exp, deacc=True, lower=True):
    text_raw = escape_text(exp, lower)
    if deacc:
        text_raw = unidecode.unidecode(text_raw)
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
