import os
import time

import fasttext
from bson import ObjectId
from progress.bar import Bar
from pymongo import MongoClient

from util import clean_exp_and_remove_stopwords


class Command:

    def run(self, limit):
        fasttext_input_file = 'fasttext-input-{}.txt'.format(limit)
        print('Input file to save is {}'.format(fasttext_input_file))
        model_file_name = 'fasttext-{}.bin'.format(limit)
        embedding_output_filename = 'embedding-{}.csv'.format(limit)

        if not os.path.isfile(fasttext_input_file):
            client = MongoClient('mongodb://admin:6677028xxbbkat@hpc-mongodb01.massey.ac.nz:27017/whenua')
            mydb = client.whenua
            mycol = mydb.AllData

            ids_file = 'doc.ids'

            current_doc_ind = 0
            bar = Bar('Reading all documents from db')
            with open(ids_file, 'r') as idf:
                with open(fasttext_input_file, 'w') as inf:
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

                        exp = doc['Text_Raw']
                        cleaned_exp = clean_exp_and_remove_stopwords(exp)
                        inf.write(cleaned_exp)
                        inf.write(' . ')
                        bar.next()
            bar.finish()

            print('--------------------------------------------------------------------')
            print('Finished exporting to file.')

        if not os.path.isfile(model_file_name):
            print('Model file {} not found, creating'.format(model_file_name))

            model = fasttext.train_unsupervised(fasttext_input_file, model='cbow', dim=300, minCount=1,
                                                ws=20, epoch=10, wordNgrams=2, loss='hs')
            model.save_model(model_file_name)
        else:
            model = fasttext.load_model(model_file_name)

        bar = Bar('Now exporting embedding to {}'.format(embedding_output_filename), max=len(model.words))
        with open(embedding_output_filename, 'w') as f:
            for word in model.words:
                embedding = model[word]
                f.write(word)
                f.write(',')
                for val in embedding:
                    f.write(str(val))
                    f.write(',')
                f.write('\n')
                bar.next()
            bar.finish()


if __name__ == '__main__':
    limit = None

    command = Command()
    start = time.time()
    command.run(limit)
    end = time.time()
    print('Took {} seconds'.format(end - start))
