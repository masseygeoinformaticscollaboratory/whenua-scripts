import pandas as pd
from bson import ObjectId
from progress.bar import Bar
from pymongo import MongoClient

from util import stop_words, calc_freq, calc_unigram_freq, calc_bigram_freq


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
                words = sentence.split(' ')
                calc_freq(words, word_freqs)
                calc_unigram_freq(words, word_freqs)
                calc_bigram_freq(words, word_freqs)
                bar.next()
        bar.finish()

        total_count = 0
        total_count_without_stopword = 0
        unique_count = 0
        unique_count_without_stopword = 0
        for word in list(word_freqs.keys()):
            count = word_freqs[word]
            total_count += count
            unique_count += 1
            if word in stop_words:
                del word_freqs[word]
            else:
                total_count_without_stopword += count
                unique_count_without_stopword += 1

        df = pd.read_excel('files/xlsx/generic_geographic_features_listing.xlsx', sheet_name='Sheet 1', na_values='')
        output_df = pd.DataFrame(columns=['Maori word', 'Frequency'])

        bar = Bar('Read and write to excel', max=df.shape[0])
        for row_ind, row in df.iterrows():
            maori_word = row['Maori Term']
            frequency = word_freqs.get(maori_word, 0)
            output_df.loc[row_ind] = [maori_word, frequency]
            bar.next()
        bar.finish()

        output_file = 'files/xlsx/generic_geographic_features_listing_freqs.xlsx'
        writer = pd.ExcelWriter(output_file)
        output_df.to_excel(excel_writer=writer, sheet_name='Sheet1', index=None)
        writer.save()
