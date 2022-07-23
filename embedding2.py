import os.path
import pickle
import time
from logging import warning

from progress.bar import Bar
from pymongo import MongoClient
import pandas as pd
import numpy as np
import scipy.spatial as sp

from AbstractCommand import AbstractCommand
from util import get_word_windows, clean_exp_and_remove_stopwords

reference_xlsx = 'files/both_together_10_words.xlsx'
embedding_file = 'embedding-None-None.csv'


class Expression:
    def __init__(self, term, before, after, is_geo):
        assert len(before) + len(after) > 0, 'Expression is invalid'
        self.term = term
        self.before = before
        self.after = after
        self.avg_embbedding = None
        self.is_geo = is_geo

    def get_full_expr(self):
        return ' '.join(self.before + [self.term.core] + self.after)

    def calc_avg_embedding(self, keywords, embeddings):
        matrix = []
        for word in self.before + self.after:
            word_index = keywords.get(word, None)
            if word_index is None:
                continue
            word_embedding = embeddings[word_index]
            matrix.append(word_embedding)

        matrix = np.array(matrix, dtype=float)
        self.avg_embbedding = matrix.mean(axis=0)
        if np.any(np.isnan(self.avg_embbedding)):
            raise ValueError("matrix is empty")


class Term:
    def __init__(self, core):
        self.core = core
        self.expressions = []


def get_terms(cache_dir, type, keyword, window_size):
    if window_size is None or window_size == 10:
        cache_file = os.path.join(cache_dir, '{}.{}.pkl'.format(keyword, type))
    else:
        cache_file = os.path.join(cache_dir, '{}-{}.{}.pkl'.format(keyword, window_size, type))
    if not os.path.isfile(cache_file):
        raise FileNotFoundError('File {} not found'.format(cache_file))

    with open(cache_file, 'rb') as f:
        return pickle.load(f)


def save_terms(cache_dir, type, term, window_size):
    if window_size is None or window_size == 10:
        cache_file = os.path.join(cache_dir, '{}.{}.pkl'.format(term.core, type))
    else:
        cache_file = os.path.join(cache_dir, '{}-{}.{}.pkl'.format(term.core, window_size, type))
    with open(cache_file, 'wb') as f:
        pickle.dump(term, f)


def extract_reference_terms(cache_dir):
    keywords = []
    xl = pd.ExcelFile(reference_xlsx)
    for keyword in xl.sheet_names:
        cache_file = os.path.join(cache_dir, '{}.ref.pkl'.format(keyword))
        if os.path.isfile(cache_file):
            keywords.append(keyword)
            continue
        df = xl.parse(keyword)
        term = Term(keyword)
        keywords.append(keyword)
        for row_num, row in df.iterrows():
            expression = clean_exp_and_remove_stopwords(row['concatenated expression'])
            is_geo = str(row['geo/or not']) == '1'
            words = expression.strip().split(' ')
            try:
                keyword_ind = words.index(keyword)
                ex = Expression(term, before=words[:keyword_ind], after=words[keyword_ind + 1:], is_geo=is_geo)
            except ValueError as e:
                ex = Expression(term, before=words, after=[], is_geo=is_geo)
            term.expressions.append(ex)
        with open(cache_file, 'wb') as f:
            pickle.dump(term, f)
    xl.close()
    return keywords


def get_embeddings():
    embedding_cache = embedding_file + '.pkl'

    if os.path.isfile(embedding_cache):
        with open(embedding_cache, 'rb') as f:
            d = pickle.load(f)
            return d['keywords'], d['embeddings']
    embeddings = []
    keywords = []
    bar = Bar('Read embedding from {} into numpy matrix'.format(embedding_file))
    with open(embedding_file, 'r') as f:
        for line in f:
            line_parts = line.strip().split(',')
            keyword = line_parts[0]
            embedding = list(map(float, [x for x in line_parts[1:] if x != '']))
            keywords.append(keyword)
            embeddings.append(embedding)
            bar.next()
    bar.finish()
    embeddings = np.array(embeddings, dtype=float)
    with open(embedding_cache, 'wb') as f:
        pickle.dump(dict(embeddings=embeddings, keywords=keywords), f)
    return keywords, embeddings


def extract_db_expressions(cache_dir, keywords: list, window_size: int):
    already_exist = []
    for keyword in keywords:
        if window_size == 10:
            cache_file = os.path.join(cache_dir, '{}.db.pkl'.format(keyword))
        else:
            cache_file = os.path.join(cache_dir, '{}-{}.db.pkl'.format(keyword, window_size))
        if os.path.isfile(cache_file):
            already_exist.append(keyword)

    if len(already_exist) == len(keywords):
        return

    new_terms = {}

    client = MongoClient('mongodb://admin:6677028xxbbkat@localhost:27017/whenua')
    mydb = client.whenua
    mycol = mydb.whenua
    docs = mycol.find({}, {'Text_Raw': 1, '_id': 0})
    bar = Bar('Reading all documents from db', max=docs.count())

    for doc_num, doc in enumerate(docs):
        sentence = doc['Text_Raw']
        sentence = clean_exp_and_remove_stopwords(sentence)
        original_words = sentence.split(' ')

        for keyword in keywords:
            if keyword in already_exist:
                continue
            term = new_terms.get(keyword, None)
            if term is None:
                term = Term(keyword)
                new_terms[keyword] = term
            pairs = get_word_windows(keyword, original_words, window_size)
            for before20, after20 in pairs:
                if len(' '.join(before20 + after20).strip()) == 0:
                    continue
                ex = Expression(term, before20, after20, None)
                term.expressions.append(ex)
        bar.next()
    bar.finish()

    for term in new_terms.values():
        if window_size == 10:
            cache_file = os.path.join(cache_dir, '{}.db.pkl'.format(term.core))
        else:
            cache_file = os.path.join(cache_dir, '{}-{}.db.pkl'.format(term.core, window_size))
        print('Saving ' + cache_file)
        with open(cache_file, 'wb') as f:
            pickle.dump(term, f)


def extract_most_similar(cache_dir, keywords, window_size):
    output_file = 'files/embedding2-{}.output.xlsx'.format(window_size)
    writer = pd.ExcelWriter(output_file, engine='xlsxwriter')
    for keyword in keywords:
        term_ref = get_terms(cache_dir, 'ref', keyword, None)
        if window_size == 10:
            rows_cache = os.path.join(cache_dir, 'rows.{}.pkl'.format(keyword))
        else:
            rows_cache = os.path.join(cache_dir, 'rows.{}-{}.pkl'.format(keyword, window_size))
        if os.path.isfile(rows_cache):
            continue

        reference_matrix = []
        rows = []

        bar = Bar('Extracting most similar for term {}'.format(keyword), max=len(term_ref.expressions))

        for expr in term_ref.expressions:
            reference_matrix.append(expr.avg_embbedding)
        reference_matrix = np.array(reference_matrix, dtype=float)

        term_db = get_terms(cache_dir, 'db', keyword, window_size)
        for expr_ind, expr in enumerate(term_db.expressions):
            if np.any(np.isnan(expr.avg_embbedding)):
                warning('Expression ' + expr.get_full_expr() + ' has nan avg embedding')
                row = [expr.get_full_expr(), 'N/A', 0, '']
                rows.append(row)
                bar.next()
                continue
            expr_embedding = np.array([expr.avg_embbedding], dtype=float)
            similarities = 1 - sp.distance.cdist(expr_embedding, reference_matrix, 'cosine')
            similarities = similarities.squeeze()
            score = None
            closest_expr = None
            is_geos = []
            sort_inds = np.flip(np.argsort(similarities))
            for sort_ind in sort_inds:
                similarity = similarities[sort_ind]
                if score is not None and score != similarity:
                    break
                score = similarity
                closest_expr = term_ref.expressions[sort_ind]
                is_geos.append(1 if closest_expr.is_geo else 0)
            is_geos = np.array(is_geos)
            is_geo = False if len(is_geos) == 0 else is_geos.mean() >= 0.5
            row = [expr.get_full_expr(), 1 if is_geo else 0, score, closest_expr.get_full_expr()]
            rows.append(row)
            bar.next()
        bar.finish()
        with open(rows_cache, 'wb') as f:
            print('Saving a cache for term {} at {}'.format(keyword, rows_cache))
            pickle.dump(rows, f)

        df = pd.DataFrame(columns=['Expression', 'geo/or not', 'Score', 'Most similar'], data=rows)
        df.to_excel(writer, sheet_name=keyword, index=None)

    writer.save()
    print('Finished extracting most similar terms and score. Output is at ' + output_file)


class Command(AbstractCommand):
    
    def __init__(self):
        super(Command, self).__init__(__file__)

    def run(self, window_size):
        embedded_terms, embeddings = get_embeddings()

        embedded_terms = {k: i for i, k in enumerate(embedded_terms)}

        ref_keywords = extract_reference_terms(self.cache_dir)
        extract_db_expressions(self.cache_dir, ref_keywords, window_size)

        bar = Bar('Calculating avg embedding', max=len(ref_keywords),
                  suffix='%(index)d/%(max)d Elapsed %(elapsed)ds - ETA %(eta)ds')

        for keyword in ref_keywords:
            term_ref = get_terms(self.cache_dir, 'ref', keyword, None)
            term_db = get_terms(self.cache_dir, 'db', keyword, window_size)
            for exp in term_ref.expressions + term_db.expressions:
                if exp.avg_embbedding is None:
                    exp.calc_avg_embedding(embedded_terms, embeddings)
            bar.next()

            save_terms(self.cache_dir, 'ref', term_ref, None)
            save_terms(self.cache_dir, 'db', term_db, window_size)
        bar.finish()

        extract_most_similar(self.cache_dir, ref_keywords, window_size)


if __name__ == '__main__':
    command = Command()
    start = time.time()
    command.run(window_size=5)
    end = time.time()
    print('Took {} seconds'.format(end - start))
