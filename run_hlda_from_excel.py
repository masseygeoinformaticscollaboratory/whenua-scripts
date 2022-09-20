import argparse
import logging
import os
import pickle
import time

import gensim.corpora as corpora
import pandas as pd
from gensim.models import LdaModel
from progress.bar import Bar

from AbstractCommand import AbstractCommand
from hldamallet import HLdaMallet

logging.basicConfig(level=logging.NOTSET)


class Command(AbstractCommand):

    def __init__(self):
        super(Command, self).__init__(__file__)

    def run(self, args):
        source = args.source
        limit = args.limit
        manual = args.manual

        data_cache = self.get_cache_path('data-{}.pkl'.format(source.replace(os.path.sep, '__')))
        print('Data cache is {}'.format(data_cache))

        if os.path.isfile(data_cache):
            with open(data_cache, 'rb') as f:
                data = pickle.load(f)
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

            print('Now saving the data to {}'.format(data_cache))
            with open(data_cache, 'wb') as f:
                pickle.dump(data, f)

        if limit is None:
            corpus_cache = self.get_cache_path('corpus-{}.pkl'.format(source.replace(os.path.sep, '__')))
        else:
            corpus_cache = self.get_cache_path('corpus-{}-{}.pkl'.format(source.replace(os.path.sep, '__'), limit))

        print('Corpus cache is {}'.format(corpus_cache))
        if os.path.isfile(corpus_cache):
            with open(corpus_cache, 'rb') as f:
                cache = pickle.load(f)
                corpus = cache['corpus']
                id2word = cache['id2word']
        else:
            data = data if limit is None else data[:limit]
            # Create Dictionary
            id2word = corpora.Dictionary(data)
            # Term Document Frequency
            corpus = [id2word.doc2bow(text) for text in data]

            print('Now saving the corpus to {}'.format(corpus_cache))

            with open(corpus_cache, 'wb') as f:
                pickle.dump({'corpus': corpus, 'id2word': id2word}, f)

        save_to = self.get_cache_path('hlda-{}.model'.format(limit))
        mallet_input_file_name = self.get_cache_path('hlda-input{}.txt'.format(limit))
        mallet_output_file_name = self.get_cache_path('hlda-input{}.mallet'.format(limit))
        output_state_file_name = self.get_cache_path('hlda-state{}.txt'.format(limit))
        blei_vocab_file_name = self.get_cache_path('hlda-blei{}.voc'.format(limit))
        blei_doc_file_name = self.get_cache_path('hlda-blei{}.doc'.format(limit))

        if not os.path.isfile(blei_vocab_file_name):
            with open(blei_vocab_file_name, 'w') as f:
                for token in id2word.token2id.keys():
                    f.write('{}\n'.format(token))

        if not os.path.isfile(blei_doc_file_name):
            with open(blei_doc_file_name, 'w') as f:
                for doc in corpus:
                    f.write(str(len(doc)))
                    for id_, val in doc:
                        f.write(' {}:{}'.format(id_, val))
                    f.write('\n')

        if not os.path.isfile(save_to):
            mallet_home = os.path.join('/Users/naflaki/workspace/Mallet')
            mallet_path = os.path.join(mallet_home, 'bin', 'mallet')
            os.environ.update(dict(MALLET_HOME=mallet_home))
            lda_model = HLdaMallet(mallet_path, None, 20, id2word=id2word)

            lda_model.corpus_to_mallet_input(corpus, mallet_input_file_name)
            print("Corpus preprocessed into " + mallet_input_file_name)

            lda_model.input_to_mallet(mallet_input_file_name, mallet_output_file_name)
            print("Corpus converted into " + mallet_output_file_name)
        else:
            lda_model = LdaModel.load(save_to)

        if manual:
            cmd = lda_model.get_train_cmd(mallet_output_file_name, output_state_file_name)
            print('Execute the following command to create gibbs sampling yourself using mallet')
            print('---------BEGIN--------')
            print(cmd)
            print('----------END---------')
            print('Execute the following command to create gibbs sampling yourself using blei\'s hlda')
            print('---------BEGIN--------')
            print('./main gibbs {} settings.txt output.txt'.format(blei_doc_file_name))
            print('----------END---------')
        else:
            lda_model.train(mallet_output_file_name, output_state_file_name)
            print("Model states saved to " + mallet_output_file_name)
            print('Save model to {}'.format(save_to))
            lda_model.save(save_to)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', action='store', dest='source', default=None)
    parser.add_argument('--limit', action='store', dest='limit', default=None, type=int)
    parser.add_argument('--manual', action='store_true', dest='manual', default=False)
    args = parser.parse_args()

    command = Command()
    start = time.time()
    command.run(args)
    end = time.time()
    print('Took {} seconds'.format(end - start))


if __name__ == '__main__':
    main()
