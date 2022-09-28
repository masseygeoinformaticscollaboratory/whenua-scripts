import argparse
import math
import time
from logging import warning

import pandas as pd
import numpy as np
import unidecode

from AbstractCommand import AbstractCommand
from embedding2 import get_embeddings
from util import escape_text
import scipy.spatial as sp


class Word:
    def __init__(self, val):
        self.val = val
        self.embedding = None

    def get_embedding(self, keywords, embeddings):
        word_index = keywords.get(self.val, None)
        if word_index is None:
            raise Exception('Word {} not found'.format(self.val))
        self.embedding = embeddings[word_index]
        if self.embedding is None:
            raise Exception('Word {} embedding not found'.format(self.val))

    def __str__(self):
        return self.val


class Group:
    def __init__(self, group_name):
        self.group_name = group_name
        self.words = []
        self.embedding = None

    def calc_avg_embedding(self):
        matrix = []
        for word in self.words:
            matrix.append(word.embedding)
        matrix = np.array(matrix, dtype=float)
        self.embedding = matrix.mean(axis=0)


class Command(AbstractCommand):

    def __init__(self):
        super(Command, self).__init__(__file__)

    @staticmethod
    def extract_groups_and_words(source):
        xl = pd.ExcelFile(source)
        row_count = 0
        dfs = {}
        for keyword in xl.sheet_names:
            df = xl.parse(keyword)
            dfs[keyword] = df
            row_count += df.shape[0]
        xl.close()

        groups = {}
        group = None
        all_words = {}

        for keyword, df in dfs.items():
            for row_num, row in df.iterrows():
                group_name = row['Group name']
                word = row['Word']

                if not isinstance(group_name, str) and math.isnan(group_name):
                    group_name = ''

                if not isinstance(word, str) and math.isnan(word):
                    word = ''

                if group_name.strip() != '':
                    group = Group(group_name)
                    groups[group_name] = group
                elif word.strip() != '':
                    word = escape_text(word, lower=True)
                    word = unidecode.unidecode(word)
                    if ' ' in word:
                        warning('Word "{}" is invalid. Skipped'.format(word))
                        continue

                    w = all_words.get(word, None)
                    if w is None:
                        w = Word(word)
                        all_words[word] = w
                    else:
                        warning('Word {} already exists in previous group'.format(word))
                    group.words.append(w)

        return groups, all_words

    @staticmethod
    def extract_embedding(groups, all_words):
        embedded_terms, embeddings = get_embeddings()
        embedded_terms = {term: index for index, term in enumerate(embedded_terms)}
        for w in all_words.values():
            w.get_embedding(embedded_terms, embeddings)

        for g in groups.values():
            g.calc_avg_embedding()

    @staticmethod
    def calc_similarity_word_to_groups(word, matrix, word_group_ind):
        matrix_excl_word_group = np.concatenate((matrix[:word_group_ind, :], matrix[word_group_ind+1:, :])).mean(axis=0)
        between = 1 - sp.distance.cdist(word.embedding.reshape(1, -1), matrix_excl_word_group.reshape(1, -1), 'cosine')[0][0]
        within = 1 - sp.distance.cdist(word.embedding.reshape(1, -1), matrix[word_group_ind, :].reshape(1, -1), 'cosine')[0][0]
        return between, within

    @staticmethod
    def calc_avg_similarity_word_to_others(word_ind, matrix, group_member_indx):
        start_ind = group_member_indx[0]
        end_ind = group_member_indx[-1]
        matrix_excl_word_group = np.concatenate((matrix[:start_ind, word_ind], matrix[end_ind+1:, word_ind]))
        matrix_of_word_group = np.concatenate((matrix[start_ind:word_ind, word_ind], matrix[word_ind+1:end_ind+1, word_ind]))

        between = matrix_excl_word_group.mean()
        within = matrix_of_word_group.mean()
        return between, within

    def run(self, args):
        source = args.source
        groups, all_words = self.extract_groups_and_words(source)
        self.extract_embedding(groups, all_words)

        word_matrices = []
        word_vals = []

        for group_name, group in groups.items():
            if len(group.words) == 0:
                continue
            for word in group.words:
                word_matrices.append(word.embedding)
                word_vals.append(word.val)

        word_matrices = np.array(word_matrices)
        word_dists = sp.distance.cdist(word_matrices, word_matrices, 'cosine')
        word_similarities = 1 - word_dists

        group_matrices = []
        group_vals = []
        for group_name, g in groups.items():
            if len(g.words) == 0:
                continue
            group_vals.append(group_name)
            group_matrices.append(g.embedding)

        group_matrices = np.array(group_matrices)
        group_dists = sp.distance.cdist(group_matrices, group_matrices, 'cosine')
        group_similarities = 1 - group_dists

        # group_ind = -1
        # bw_wt_mat = []
        # for group in groups.values():
        #     if len(group.words) == 0:
        #         continue
        #     group_ind += 1
        #     for word in group.words:
        #         between, within = self.calc_similarity_word_to_groups(word, group_matrices, group_ind)
        #         bw_wt_mat.append([between, within])

        group_ind = 0
        bw_wt_mat = []
        for group in groups.values():
            if len(group.words) == 0:
                continue
            group_member_inds = [group_ind + x for x in range(len(group.words))]
            for word_ind, word in enumerate(group.words):
                between, within = self.calc_avg_similarity_word_to_others(word_ind + group_ind, word_similarities, group_member_inds)
                bw_wt_mat.append([between, within])

            group_ind += len(group.words)

        bw_wt_mat = np.array(bw_wt_mat, dtype=float)

        writer = pd.ExcelWriter('files/group-similariries-bw-wt.xlsx', engine='xlsxwriter')

        bw_wt_data = np.concatenate((np.array(word_vals, dtype='O').reshape(-1, 1), bw_wt_mat), axis=1)
        bw_wt_df = pd.DataFrame(columns=['', 'Between', 'Within'], data=bw_wt_data)
        bw_wt_df.to_excel(writer, sheet_name='Between, within', index=None)

        word_data = np.concatenate((np.array(word_vals, dtype='O').reshape(-1, 1), word_similarities), axis=1)
        word_df = pd.DataFrame(columns=[' '] + word_vals, data=word_data)
        word_df.to_excel(writer, sheet_name='Word similarities', index=None)

        group_data = np.concatenate((np.array(group_vals, dtype='O').reshape(-1, 1), group_similarities), axis=1)
        group_df = pd.DataFrame(columns=[' '] + group_vals, data=group_data)
        group_df.to_excel(writer, sheet_name='Group similarities', index=None)

        writer.save()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', action='store', dest='source', default=None)
    args = parser.parse_args()

    command = Command()
    start = time.time()
    command.run(args)
    end = time.time()
    print('Took {} seconds'.format(end - start))


if __name__ == '__main__':
    main()
