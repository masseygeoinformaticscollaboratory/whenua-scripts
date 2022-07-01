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
    def __init__(self, sheet_name):
        self.core = sheet_name
        self.reference_expressions = []
        self.all_expressions = []


def get_reference_terms():
    terms = {}

    xl = pd.ExcelFile(reference_xlsx)
    for keyword in xl.sheet_names:
        df = xl.parse(keyword)
        term = Term(keyword)
        terms[keyword] = term
        for row_num, row in df.iterrows():
            expression = clean_exp_and_remove_stopwords(row['concatenated expression'])
            is_geo = str(row['geo/or not']) == '1'
            words = expression.strip().split(' ')
            try:
                keyword_ind = words.index(keyword)
                ex = Expression(term, before=words[:keyword_ind], after=words[keyword_ind + 1:], is_geo=is_geo)
            except ValueError as e:
                ex = Expression(term, before=words, after=[], is_geo=is_geo)
            term.reference_expressions.append(ex)
    xl.close()
    return terms


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


def get_all_expressions(terms: dict):
    client = MongoClient('mongodb://admin:6677028xxbbkat@localhost:27017/whenua')
    mydb = client.whenua
    mycol = mydb.whenua
    docs = mycol.find({}, {'Text_Raw': 1, '_id': 0})
    bar = Bar('Reading all documents from db', max=docs.count())

    for doc_num, doc in enumerate(docs):
        sentence = doc['Text_Raw']
        sentence = clean_exp_and_remove_stopwords(sentence)
        original_words = sentence.split(' ')

        for keyword, term in terms.items():
            pairs = get_word_windows(keyword, original_words, 10)
            for before20, after20 in pairs:
                if len(' '.join(before20 + after20).strip()) == 0:
                    continue
                ex = Expression(term, before20, after20, None)
                term.all_expressions.append(ex)
        bar.next()
    bar.finish()


def extract_most_similar(cache_dir, terms):
    output_file = 'files/embedding2.output.xlsx'
    writer = pd.ExcelWriter(output_file, engine='xlsxwriter')
    for term_ind, term in enumerate(terms.values()):
        rows_cache = os.path.join(cache_dir, 'embedding2.rows.{}.pkl'.format(term.core))
        if os.path.isfile(rows_cache):
            with open(rows_cache, 'rb') as f:
                rows = pickle.load(f)
        else:
            rows = []

        reference_matrix = []

        bar = Bar('Extracting most similar for term {}'.format(term.core), max=len(term.all_expressions))

        for expr in term.reference_expressions:
            reference_matrix.append(expr.avg_embbedding)
        reference_matrix = np.array(reference_matrix, dtype=float)

        bar.next(len(rows))
        for expr_ind, expr in enumerate(term.all_expressions):
            if expr_ind < len(rows):
                continue
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
                closest_expr = term.reference_expressions[sort_ind]
                is_geos.append(1 if closest_expr.is_geo else 0)
            is_geos = np.array(is_geos)
            is_geo = False if len(is_geos) == 0 else is_geos.mean() >= 0.5
            row = [expr.get_full_expr(), 1 if is_geo else 0, score, closest_expr.get_full_expr()]
            rows.append(row)

            if (expr_ind > 0 and expr_ind % 10000 == 0) or expr_ind == len(term.all_expressions) - 1:
                with open(rows_cache, 'wb') as f:
                    print('Saving a cache for term {} at {}'.format(term.core, rows_cache))
                    pickle.dump(rows, f)
            bar.next()
        bar.finish()

        df = pd.DataFrame(columns=['Expression', 'geo/or not', 'Score', 'Most similar'], data=rows)
        df.to_excel(writer, sheet_name=term.core, index=None)

    writer.save()
    print('Finished extracting most similar terms and score. Output is at ' + output_file)


class Command(AbstractCommand):
    
    def __init__(self):
        super(Command, self).__init__(__file__)

    def run(self, limit, method):
        if method not in ['normal', 'tfidf']:
            raise Exception('Unknown method {}'.format(method))
        keywords, embeddings = get_embeddings()

        keywords = {k: i for i, k in enumerate(keywords)}
        cache_file = self.get_cache_path('embedding2.terms.pkl')
        if os.path.isfile(cache_file):
            with open(cache_file, 'rb') as f:
                terms = pickle.load(f)
        else:
            terms = get_reference_terms()
            get_all_expressions(terms)

            with open(cache_file, 'wb') as f:
                pickle.dump(terms, f)

        num_calc = 0
        for term in terms.values():
            num_calc += len(term.all_expressions) + len(term.reference_expressions)

        bar = Bar('Calculating avg embedding', max=num_calc,
                  suffix='%(index)d/%(max)d Elapsed %(elapsed)ds - ETA %(eta)ds')
        for term in terms.values():
            for exp in (term.reference_expressions + term.all_expressions):
                if exp.avg_embbedding is None:
                    exp.calc_avg_embedding(keywords, embeddings)
                bar.next()
        bar.finish()

        with open(cache_file, 'wb') as f:
            pickle.dump(terms, f)

        extract_most_similar(self.cache_dir, terms)

if __name__ == '__main__':
    limit = None
    method = 'normal'

    command = Command()
    start = time.time()
    command.run(limit, method)
    end = time.time()
    print('Took {} seconds'.format(end - start))
