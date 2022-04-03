import re
import time

import gensim
import regex
from bson import ObjectId
from progress.bar import Bar
from pymongo import MongoClient

from util import clean_exp_and_return_arr, calc_freq


class Command:

    def run(self, limit):
        word_freqs = {}
        client = MongoClient('mongodb://admin:6677028xxbbkat@hpc-mongodb01.massey.ac.nz:27017/whenua')
        mydb = client.whenua
        mycol = mydb.AllData

        ids_file = 'doc.ids'

        current_doc_ind = 0
        bar = Bar('Reading all documents from db')
        with open(ids_file, 'r') as idf:
            while True:
                if limit is not None and current_doc_ind > limit:
                    break
                current_doc_id = idf.readline()
                if current_doc_id == '':
                    break

                current_doc_id = current_doc_id.strip()

                current_doc_ind += 1
                docs = mycol.find({'_id': ObjectId(current_doc_id)}, {'Text_Raw': 1, '_id': 0})
                if docs.count() == 0:
                    continue
                doc = docs[0]

                sentence = doc['Text_Raw']
                words = clean_exp_and_return_arr(sentence)
                calc_freq(words, word_freqs)
                bar.next()
        bar.finish()

        output_file = 'files/xlsx/word_freqs.csv'

        bar = Bar('Exporting word frequency...', max=len(word_freqs))
        with open(output_file, 'w') as f:
            f.write('Word\t')
            f.write('Frequency\n')
            for word in list(word_freqs.keys()):
                count = word_freqs[word]
                f.write(word)
                f.write('\t')
                f.write(str(count))
                f.write('\n')
                bar.next()
        bar.finish()

        print('Exported to {}'.format(output_file))


if __name__ == '__main__':
    limit = None

    command = Command()
    start = time.time()
    command.run(limit)
    end = time.time()
    print('Took {} seconds'.format(end - start))
