import argparse
import math
import os.path
import time
from logging import warning

import pandas as pd
import numpy as np
import unidecode

from AbstractCommand import AbstractCommand
from util import escape_text


class Group:
    def __init__(self, group_name):
        self.group_name = group_name
        self.words = []

    def __str__(self):
        return '{}: [{}]'.format(self.group_name, ','.join(self.words))
    

class Command(AbstractCommand):

    def __init__(self):
        super(Command, self).__init__(__file__)

    def extract_groups_and_words_from_excel(self, source):
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

                    group.words.append(word)
        groups = [x for x in groups.values() if len(x.words) > 0]
        for g in groups:
            g.words = list(set(g.words))
        return groups

    def extract_groups_and_words_from_lda(self, source, thresh):
        groups = []
        with open(source, 'r') as f:
            for line in f:
                line = line.strip()
                line_no, formula = line.split(',')
                terms = formula.split(' + ')
                group = Group('Model {}'.format(line_no))
                for term in terms:
                    weight, word = term.split('*')
                    weight = float(weight)
                    if weight >= thresh:
                        word = word.replace('"', '')
                        word = escape_text(word, lower=True)
                        word = unidecode.unidecode(word)
                        group.words.append(word)
                if len(group.words) > 0:
                    groups.append(group)
        return groups

    def calc_stats(self, g1: Group, g2: Group) -> (float, float):
        all_words_concat = g1.words + g2.words
        union = set(all_words_concat)
        intersect_count = len(all_words_concat) - len(union)
        overlap = intersect_count / min(len(g1.words), len(g2.words))
        jaccard = intersect_count / len(union)
        return jaccard, overlap

    def get_stats(self, matrix):
        max = np.max(matrix)
        sum = np.sum(matrix)
        size_incl_zero = matrix.size
        size_non_zero = np.count_nonzero(matrix)
        avg0 = sum / size_incl_zero
        avg1 = sum / size_non_zero

        return max, avg0, avg1

    def get_dfs(self, ntopics, thresh, excel_groups):
        lda_source = "cache/run_lda_from_excel/lda-hierarchical.{}.csv".format(ntopics)
        if not os.path.isfile(lda_source):
            return None
        lda_groups = self.extract_groups_and_words_from_lda(lda_source, thresh)
        jaccard_matrix = np.ndarray((len(excel_groups), len(lda_groups)), dtype=float)
        overlap_matrix = np.ndarray((len(excel_groups), len(lda_groups)), dtype=float)

        headers = [g.group_name for g in excel_groups]
        row_headers = [g.group_name for g in lda_groups]

        for g1_ind, g1 in enumerate(excel_groups):
            for g2_ind, g2 in enumerate(lda_groups):
                jaccard, overlap = self.calc_stats(g1, g2)
                jaccard_matrix[g1_ind][g2_ind] = jaccard
                overlap_matrix[g1_ind][g2_ind] = overlap

        jaccard_data = np.concatenate(
            (np.array(row_headers).astype('O').reshape((1, -1)), jaccard_matrix.astype('O')), axis=0).T
        jaccard_df = pd.DataFrame(columns=[' '] + headers, data=jaccard_data)

        overlap_data = np.concatenate(
            (np.array(row_headers).astype('O').reshape((1, -1)), overlap_matrix.astype('O')), axis=0).T
        overlap_df = pd.DataFrame(columns=[' '] + headers, data=overlap_data)
        
        return jaccard_df, overlap_df, self.get_stats(jaccard_matrix), self.get_stats(overlap_matrix)

    def get_stat_df(self, stat_name, row, ntopics_list):
        header = [''] + [str(x) for x in ntopics_list]
        row = [stat_name] + row
        data = np.array(row, dtype=object).reshape((1, len(row)))
        return pd.DataFrame(columns=header, data=data)

    def write_multiple_dfs(self, dfs, sheet_name, writer, merge=False, transpose=False):
        if merge:
            global_df = dfs[0]
            for df in dfs[1:]:
                global_df = global_df.append(df)

            if transpose:
                global_df = global_df.transpose()

            global_df.to_excel(writer, sheet_name=sheet_name, header=False, index=True)

        else:
            start_row = 0
            for df in dfs:
                df.to_excel(writer, sheet_name=sheet_name, index=None, startrow=start_row)
                start_row += len(df.index) + 2

    def run(self, args):
        excel_groups = self.extract_groups_and_words_from_excel(args.group_source)
        dfs = {}
        jmax_matrix = []
        javg0_matrix = []
        javg1_matrix = []
        omax_matrix = []
        oavg0_matrix = []
        oavg1_matrix = []
        for ntopics in range(10, 410, 10):
            results = self.get_dfs(ntopics, args.thresh, excel_groups)
            if results is None:
                continue

            jaccard_df, overlap_df, (jmax, javg0, javg1), (omax, oavg0, oavg1) = results
            dfs[ntopics] = jaccard_df, overlap_df
            jmax_matrix.append(jmax)
            javg0_matrix.append(javg0)
            javg1_matrix.append(javg1)
            omax_matrix.append(omax)
            oavg0_matrix.append(oavg0)
            oavg1_matrix.append(oavg1)

        output_file = 'files/sim_overlap-{}.xlsx'.format(args.thresh)
        writer = pd.ExcelWriter(output_file, engine='xlsxwriter')

        stats_dfs = [
            self.get_stat_df('Max Jaccard', jmax_matrix, list(dfs.keys())),
            self.get_stat_df('Avg (incl 0) Jaccard', javg0_matrix, list(dfs.keys())),
            self.get_stat_df('Avg (excl 0) Jaccard', javg1_matrix, list(dfs.keys())),
            self.get_stat_df('Max Overlap', omax_matrix, list(dfs.keys())),
            self.get_stat_df('Avg (incl 0) Overlap', oavg0_matrix, list(dfs.keys())),
            self.get_stat_df('Avg (excl 0) Overlap', oavg1_matrix, list(dfs.keys())),
        ]

        self.write_multiple_dfs(stats_dfs, 'Stats', writer, True, True)

        for ntopics, (jaccard_df, overlap_df) in dfs.items():
            sheet_name = 'lda-{}'.format(ntopics)
            self.write_multiple_dfs([jaccard_df, overlap_df], sheet_name, writer)

        writer.save()
        print('Result exported to ' + output_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group-source', action='store', dest='group_source', required=True, type=str)
    parser.add_argument('--thresh', action='store', dest='thresh', default=0.0, type=float)
    args = parser.parse_args()

    command = Command()
    start = time.time()
    command.run(args)
    end = time.time()
    print('Took {} seconds'.format(end - start))


if __name__ == '__main__':
    main()
