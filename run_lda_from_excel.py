import argparse
import os
import pickle
import time

import gensim
import gensim.corpora as corpora
import pandas as pd
from gensim.models import LdaModel
from progress.bar import Bar

from AbstractCommand import AbstractCommand
from ldamallet import LdaMallet
from util import clean_exp_and_remove_stopwords


class Command(AbstractCommand):

    def __init__(self):
        super(Command, self).__init__(__file__)

    def run(self, args):
        source = args.source
        method = args.method

        if method not in ['normal', 'tfidf', 'hierarchical']:
            raise Exception('Unknown method {}'.format(method))

        corpus_cache = self.get_cache_path('corpus-{}.pkl'.format(source.replace(os.path.sep, '__')))
        print('Cache to save is {}'.format(corpus_cache))

        if os.path.isfile(corpus_cache):
            with open(corpus_cache, 'rb') as f:
                cache = pickle.load(f)
                corpus = cache['corpus']
                id2word = cache['id2word']
        else:
            data = []
            xl = pd.ExcelFile(source)
            row_count = 0
            dfs = {}
            bar = Bar('Parsing excel file {}'.format(source), max=len(xl.sheet_names))
            for keyword in xl.sheet_names:
                df = xl.parse(keyword)
                dfs[keyword] = df
                row_count += df.shape[0]
                bar.next()
            bar.finish()
            xl.close()

            bar = Bar('Reading rows from {}'.format(source), max=row_count)
            for keyword, df in dfs.items():
                for row_num, row in df.iterrows():
                    if row['Maori percentage'] >= 95 and row['geo/or not'] == 1:
                        cleaned_exp = row['Expression']
                        data.append(cleaned_exp.strip().split(' '))
                    bar.next()
            bar.finish()

            # Create Dictionary
            id2word = corpora.Dictionary(data)
            # Term Document Frequency
            corpus = [id2word.doc2bow(text) for text in data]

            print('Now saving the corpus to {}'.format(corpus_cache))

            with open(corpus_cache, 'wb') as f:
                pickle.dump({'corpus': corpus, 'id2word': id2word}, f)

        if method == 'tfidf':
            tfidf = gensim.models.TfidfModel(corpus)
            corpus = tfidf[corpus]

        for num_topics in range(10, 410, 10):
            if method == 'normal':
                save_to = 'lda.{}.model'.format(num_topics)
                output_file_name = 'lda.{}.csv'.format(num_topics)
            elif method == 'tfidf':
                save_to = 'lda-tfidf-{}.model'.format(num_topics)
                output_file_name = 'lda-tfidf.{}.csv'.format(num_topics)
            elif method == 'hierarchical':
                save_to = 'lda-hierarchical-{}.model'.format(num_topics)
                output_file_name = 'lda-hierarchical.{}.csv'.format(num_topics)
            else:
                raise Exception('Unknown method {}'.format(method))

            save_to = self.get_cache_path(save_to)
            output_file_name = self.get_cache_path(output_file_name)

            if not os.path.isfile(save_to):
                print('Now running the model with number of topics = {}, method of LDA = {}'.format(num_topics, method))

                # Build LDA model
                if method in ['normal', 'tfidf']:
                    lda_model = gensim.models.LdaMulticore(corpus=corpus, id2word=id2word, num_topics=num_topics)
                else:
                    mallet_home = os.path.join(self.root_dir, 'mallet')
                    mallet_path = os.path.join(mallet_home, 'bin', 'mallet')
                    os.environ.update(dict(MALLET_HOME=mallet_home))
                    lda_model = LdaMallet(mallet_path, corpus, num_topics, id2word=id2word)

                print('Save model to {}'.format(save_to))
                lda_model.save(save_to)
            else:
                if not os.path.isfile(output_file_name):
                    lda_model = LdaModel.load(save_to)

            if not os.path.isfile(output_file_name):
                shown_topics = lda_model.show_topics(num_topics=num_topics)
                with open(output_file_name, 'w') as f:
                    for topic_num, topic_model in shown_topics:
                        f.write(str(topic_num))
                        f.write(',')
                        f.write(topic_model)
                        f.write('\n')
                print('Output saved to {}'.format(output_file_name))

            # Print the Keyword in the 10 topics
            # pprint(lda_model.print_topics())
            # pprint(lda_model.show_topics(formatted=True))
            # doc_ldas = lda_model[corpus]
            #
            # for i, doc_lda in enumerate(doc_ldas, 1):
            #     if i > 10:
            #         break
            #     print('----------------------------------------')
            #     print('Document #{}\'s topic model probabilities are '.format(i))
            #     print(doc_lda)
            #     print('----------------------------------------')
            #     print('')
            #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', action='store', dest='source', default=None)
    parser.add_argument('--method', action='store', dest='method', default='normal', type=str)
    args = parser.parse_args()

    command = Command()
    start = time.time()
    command.run(args)
    end = time.time()
    print('Took {} seconds'.format(end - start))


if __name__ == '__main__':
    main()
