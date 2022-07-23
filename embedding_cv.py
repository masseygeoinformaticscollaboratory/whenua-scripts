import os.path
import pickle
import time
from logging import warning

import numpy as np
import pandas as pd
import scipy.spatial as sp
from progress.bar import Bar

from AbstractCommand import AbstractCommand
from embedding2 import get_embeddings, extract_reference_terms, extract_db_expressions, get_terms, save_terms
from util import clean_exp_and_remove_stopwords

reference_xlsx = 'files/both_together_10_words.xlsx'
embedding_file = 'embedding-None-None.csv'


def perf_measure(expected_values, actual_values):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for expected, actual in zip(expected_values, actual_values):
        if expected:
            if actual:
                TP += 1
            else:
                FN += 1
        else:
            if actual:
                FP += 1
            else:
                TN += 1

    return TP, FP, TN, FN


def cross_validate(cache_dir, ref_keywords, k=1):
    output_file = 'files/embedding_cv.xlsx'
    writer = pd.ExcelWriter(output_file, engine='xlsxwriter')

    reference_matrix = []
    corresponding_exps = []
    for keyword in ref_keywords:
        term_ref = get_terms(cache_dir, 'ref', keyword, None)

        for expr in term_ref.expressions:
            reference_matrix.append(expr.avg_embbedding)
            corresponding_exps.append(expr)
    reference_matrix = np.array(reference_matrix, dtype=float)

    expr_ind = 0
    expected = []
    actual = []

    for keyword in ref_keywords:
        rows = []
        term_ref = get_terms(cache_dir, 'ref', keyword, None)
        bar = Bar('Extracting most similar for term {}'.format(keyword), max=len(term_ref.expressions),
                  suffix='%(index)d/%(max)d Elapsed %(elapsed)ds - ETA %(eta)ds')
        for expr in term_ref.expressions:
            if np.any(np.isnan(expr.avg_embbedding)):
                warning('Expression ' + expr.get_full_expr() + ' has nan avg embedding')
                row = [expr.get_full_expr(), expr.is_geo, 'N/A', 0, '', '']
                rows.append(row)
                bar.next()
                expr_ind += 1
                continue

            expr_embedding = np.array([expr.avg_embbedding], dtype=float)
            similarities = 1 - sp.distance.cdist(expr_embedding, reference_matrix, 'cosine')
            similarities = similarities.squeeze()

            closest_expr = None
            sort_inds = np.flip(np.argsort(similarities))
            k_matches = sort_inds[:k]
            scores = similarities[k_matches]
            scores_is_geo = []
            closest_exprs = [corresponding_exps[x] for x in k_matches]
            is_geos = []
            for score, closest_expr in zip(scores, closest_exprs):
                if closest_expr.is_geo:
                    is_geos.append(1)
                    scores_is_geo.append(score)
                else:
                    is_geos.append(0)
            score = np.mean(scores_is_geo)

            is_geos = np.array(is_geos)
            is_geo = False if len(is_geos) == 0 else is_geos.mean() >= 0.5
            row = [expr.get_full_expr(), 1 if expr.is_geo else 0, 1 if is_geo else 0, score]
            row.append(closest_expr.get_full_expr())
            row.append(closest_expr.term.core)
            rows.append(row)
            expected.append(expr.is_geo)
            actual.append(is_geo)

            expr_ind += 1
            bar.next()
        bar.finish()

        columns = ['Expression', 'original geo/or not', 'validated geo/or not', 'Score', 'Most similar', 'Term']

        df = pd.DataFrame(columns=columns, data=rows)
        df.to_excel(writer, sheet_name=keyword, index=None)

    writer.save()

    tp, fp, tn, fn = perf_measure(expected, actual)
    precision = tp / (fp + tp)
    recall = tp / (fn + tp)
    accuracy = (tp + tn) / (tp + fn + tn + fp)
    f1 = 2 * precision * recall / (precision + recall)

    print('Finished extracting most similar terms and score. Output is at ' + output_file)
    print('TP={} FP={} TN={} FN={} Precision={} Recall={} Accuracy={} F1={}'.format(tp, fp, tn, fn, precision, recall, accuracy, f1))


class Command(AbstractCommand):

    def __init__(self):
        super(Command, self).__init__(__file__)

    def run(self, limit, method):
        if method not in ['normal', 'tfidf']:
            raise Exception('Unknown method {}'.format(method))
        embedded_terms, embeddings = get_embeddings()

        embedded_terms = {k: i for i, k in enumerate(embedded_terms)}

        ref_keywords = extract_reference_terms(self.cache_dir)

        # bar = Bar('Calculating avg embedding for reference terms', max=len(ref_keywords),
        #           suffix='%(index)d/%(max)d Elapsed %(elapsed)ds - ETA %(eta)ds')
        #
        # for keyword in ref_keywords:
        #     term_ref = get_terms(self.cache_dir, 'ref', keyword)
        #     for exp in term_ref.expressions:
        #         if exp.avg_embbedding is None:
        #             exp.calc_avg_embedding(embedded_terms, embeddings)
        #     bar.next()
        #
        #     save_terms(self.cache_dir, 'ref', term_ref)
        # bar.finish()
        #
        # bar = Bar('Calculating avg embedding for all terms', max=len(all_terms),
        #           suffix='%(index)d/%(max)d Elapsed %(elapsed)ds - ETA %(eta)ds')
        #
        # for keyword in all_terms:
        #     term_db = get_terms(self.cache_dir, 'db', keyword)
        #     for exp in term_db.expressions:
        #         if exp.avg_embbedding is None:
        #             exp.calc_avg_embedding(embedded_terms, embeddings)
        #     bar.next()
        #
        #     save_terms(self.cache_dir, 'db', term_db)
        # bar.finish()

        cross_validate(self.cache_dir, ref_keywords, 3)


if __name__ == '__main__':
    limit = None
    method = 'normal'

    command = Command()
    start = time.time()
    command.run(limit, method)
    end = time.time()
    print('Took {} seconds'.format(end - start))
