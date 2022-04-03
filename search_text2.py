import re
import time
from collections import OrderedDict
from copy import deepcopy

import pandas as pd
from bson import ObjectId
from progress.bar import Bar
from pymongo import MongoClient

from util import escape_text


def search_all_keywords(word_list, keywords, definitives):
    retval = {}
    for keyword in keywords:
        while True:
            try:
                index = word_list.index(keyword)
                if index > 0:
                    prev_word = word_list[index - 1]
                    if prev_word in definitives:
                        keyword_defs = retval.get(keyword, None)
                        if keyword_defs is None:
                            keyword_defs = {}
                            retval[keyword] = keyword_defs
                        keyword_defs[prev_word] = keyword_defs.get(prev_word, 0) + 1

                word_list = word_list[index + 1:]
            except ValueError:
                break
    return retval


def escape_text_for_metadata(text_raw):
    text_raw = text_raw.strip()
    text_raw = re.sub('\s+', ' ', text_raw)
    return text_raw


class Command:

    def get_word_windows(self, keyword, keywords, definitives, original_words, window_length):
        searching_frame = deepcopy(original_words)
        searching_frame_offset = 0

        pairs = []
        while True:
            definitive = None
            try:
                keyword_index = searching_frame.index(keyword)
                if keyword_index > 0:
                    word_before = searching_frame[keyword_index - 1]
                    if word_before in definitives:
                        definitive = word_before

            except ValueError:
                break

            keyword_index_inset = searching_frame_offset + keyword_index

            window_1_start = keyword_index_inset - window_length - 1
            if window_1_start < 0:
                window_1_start = 0

            window_1_ends = keyword_index_inset - 1
            if window_1_ends < 0:
                window_1_ends = 0

            window_2_ends = keyword_index_inset + window_length
            window_2_ends = min(window_2_ends, len(original_words)) + 1

            if definitive is not None:
                window_1 = original_words[window_1_start:window_1_ends]
                window_2 = original_words[keyword_index_inset+1:window_2_ends]

                window_1_keywords_found = search_all_keywords(window_1, keywords, definitives)
                window_2_keywords_found = search_all_keywords(window_2, keywords, definitives)

                count = 0
                found_terms = []

                for _k, _defs in window_1_keywords_found.items():
                    for _def, _count in _defs.items():
                        term = _def + ' ' + _k
                        count += _count
                        found_terms.append(term)

                for _k, _defs in window_2_keywords_found.items():
                    for _def, _count in _defs.items():
                        term = _def + ' ' + _k
                        count += _count
                        found_terms.append(term)

                pair = (' '.join(window_1), ' '.join(window_2), definitive, count, found_terms)
                pairs.append(pair)

            searching_frame_offset += keyword_index + 1

            searching_frame = original_words[searching_frame_offset:]

        return pairs

    def run(self, limit):
        file_loc = 'files/xlsx/word_windows_list.xlsx'
        client = MongoClient('mongodb://admin:6677028xxbbkat@hpc-mongodb01.massey.ac.nz:27017/whenua')
        mydb = client.whenua
        mycol = mydb.AllData

        df = pd.read_excel(file_loc, sheet_name='Sheet1', na_values='')

        result = {}
        keywords = []
        definitives = []
        for row_num, row in df.iterrows():
            keyword = row['Maori word']
            definitive = row['Definitive']
            keywords.append(keyword)
            if isinstance(definitive, str):
                definitives.append(definitive.lower())

        for keyword in keywords:
            result[keyword] = []

        ids_file = 'doc.ids'

        current_doc_ind = 0
        bar = Bar('Reading all documents from db')

        metadata_columns = OrderedDict()
        with open(ids_file, 'r') as f:
            while True:
                if limit is not None and current_doc_ind > limit:
                    break
                current_doc_id = f.readline()
                if current_doc_id == '':
                    break

                current_doc_id = current_doc_id.strip()

                current_doc_ind += 1
                docs = mycol.find({'_id': ObjectId(current_doc_id)}, {'_id': 0})
                if docs.count() == 0:
                    continue
                doc = docs[0]

                metadata = {}
                for attr_name, attr_val in doc.items():
                    if attr_name in ['Text_Raw', '_id']:
                        continue

                    if isinstance(attr_val, str):
                        attr_val = escape_text_for_metadata(attr_val)
                    else:
                        attr_val = str(attr_val)

                    if attr_name not in metadata_columns:
                        col_ind = len(metadata_columns)
                        metadata_columns[attr_name] = col_ind
                    else:
                        col_ind = metadata_columns[attr_name]

                    metadata[col_ind] = attr_val

                sentence = doc['Text_Raw']
                sentence = escape_text(sentence)
                original_words = sentence.split(' ')
                for keyword, extracted_rows in result.items():
                    pairs = self.get_word_windows(keyword, keywords, definitives, original_words, 20)
                    for before20, after20, _def, count, found_terms in pairs:
                        extracted_rows.append((before20, after20, _def, count, found_terms, metadata))
                bar.next()
        bar.finish()

        output_file = 'files/xlsx/word_and_definitive_metadata_windows_list_output2.csv'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('Definitive\tKeyword\t20 words before\t20 words after\tCount\tFound terms\t')
            for attr_name in metadata_columns.keys():
                f.write(attr_name)
                f.write('\t')
            f.write('\n')

            for keyword, extracted_rows in result.items():
                bar = Bar('Write keyword {}'.format(keyword), max=len(extracted_rows))
                for before20, after20, _def, count, found_terms, metadata in extracted_rows:
                    f.write(_def)
                    f.write('\t')
                    f.write(keyword)
                    f.write('\t')
                    f.write(before20)
                    f.write('\t')
                    f.write(after20)
                    f.write('\t')
                    f.write(str(count))
                    f.write('\t')
                    f.write('*'.join(found_terms))
                    f.write('\t')

                    for attr_name, col_ind in metadata_columns.items():
                        attr_val = metadata.get(col_ind, '')
                        f.write(attr_val)
                        f.write('\t')

                    f.write('\n')

                    bar.next()
                bar.finish()

        print('Finish writing to file ' + output_file)


if __name__ == '__main__':
    limit = None

    command = Command()
    start = time.time()
    command.run(limit)
    end = time.time()
    print('Took {} seconds'.format(end - start))
