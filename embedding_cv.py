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
