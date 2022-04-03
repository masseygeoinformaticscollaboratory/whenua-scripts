import os
import time

import gensim
import gensim.corpora as corpora
from bson import ObjectId
from gensim.models import LdaModel
from progress.bar import Bar
from pymongo import MongoClient

from util import clean_exp_and_return_arr


class Command:

    def run(self, limit, method):
        if method not in ['normal', 'tfidf']:
            raise Exception('Unknown method {}'.format(method))

        client = MongoClient('mongodb://admin:6677028xxbbkat@hpc-mongodb01.massey.ac.nz:27017/whenua')
        mydb = client.whenua
        mycol = mydb.AllData

        ids_file = 'doc.ids'

        current_doc_ind = 0
        bar = Bar('Reading all documents from db')
        data = []
        with open(ids_file, 'r') as f:
            while True:
                if limit is not None and current_doc_ind > limit:
                    break
                current_doc_id = f.readline()
                if current_doc_id == '':
                    break

                current_doc_id = current_doc_id.strip()

                current_doc_ind += 1
                docs = mycol.find({'_id': ObjectId(current_doc_id)}, {'Text_Raw': 1, '_id': 0})
                if docs.count() == 0:
                    continue
                doc = docs[0]
                exp = doc['Text_Raw']
                cleaned_exp = clean_exp_and_return_arr(exp, True)
                data.append(cleaned_exp)
                bar.next()
        bar.finish()

        # Create Dictionary
        id2word = corpora.Dictionary(data)
        # Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in data]

        if method == 'tfidf':
            tfidf = gensim.models.TfidfModel(corpus)
            corpus = tfidf[corpus]

        for num_topics in range(20, 30, 10):
            if method == 'normal':
                save_to = 'lda2.{}.model'.format(num_topics)
                output_file_name = 'lda2.{}.csv'.format(num_topics)
            else:
                save_to = 'lda-tfidf2-{}.model'.format(num_topics)
                output_file_name = 'lda-tfidf2.{}.csv'.format(num_topics)

            if not os.path.isfile(save_to):
                print('Now running the model with number of topics = {}, method of LDA = {}'.format(num_topics, method))

                # Build LDA model
                lda_model = gensim.models.LdaMulticore(corpus=corpus, id2word=id2word, num_topics=num_topics)

                print('Save model to {}'.format(save_to))
                lda_model.save(save_to)

            else:
                lda_model = LdaModel.load(save_to)

            shown_topics = lda_model.show_topics(num_topics=num_topics)
            with open(output_file_name, 'w') as f:
                for topic_num, topic_model in shown_topics:
                    f.write(str(topic_num))
                    f.write(',')
                    f.write(topic_model)
                    f.write('\n')
            print('Output saved to {}'.format(output_file_name))


if __name__ == '__main__':
    limit = None
    method = 'normal'

    command = Command()
    start = time.time()
    command.run(limit, method)
    end = time.time()
    print('Took {} seconds'.format(end - start))
