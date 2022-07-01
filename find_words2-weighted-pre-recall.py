from collections import OrderedDict
from logging import warning

import numpy as np
import pandas as pd
from progress.bar import Bar

from util import escape_text

# expression_file = 'files/xlsx/Precisiion-Recall our corpus.xlsx'
expression_file = 'files/xlsx/Precisiion-Recall our corpus - 10.xlsx'
# expression_file = 'files/xlsx/Expressions for precision-recall.xlsx'

terms_file = 'files/xlsx/word_windows_list.xlsx'

weights_file = 'files/xlsx/weights.xlsx'

raw_output_file = 'files/xlsx/Precisiion-Recall our corpus - 10 -raw - gte.xlsx'
# raw_output_file = 'files/xlsx/Precisiion-Recall their corpus-raw.xlsx'

# summary_output_file = 'files/xlsx/Precisiion-Recall their corpus-summary.xlsx'
summary_output_file = 'files/xlsx/Precisiion-Recall our corpus - 10 -summary - gte.xlsx'

term_df = pd.read_excel(terms_file)
weights_df = pd.read_excel(weights_file)

terms = []
weights = {}
definitives = ['***']

weight_missing = set()


for row_num, row in term_df.iterrows():
    term = row['Maori word']
    terms.append(term)
    definitive = row['Definitive']
    if isinstance(definitive, str):
        definitives.append(definitive.lower())


for row_num, row in weights_df.iterrows():
    term = row['Maori word']
    weight = int(row['Likert – HM'])
    weights[term] = weight


# output_dfs = OrderedDict()
headings = ['concatenated expression', 'core term index', 'def+term count', 'def+terms found', 'term count', 'terms and indices found', 'is geo', 'score']


def calc_stats(df, col_name, thresh):
    df['TP'] = (df[col_name] >= thresh).astype(int) * (df['is geo'] == 1).astype(int)
    df['FP'] = (df[col_name] >= thresh).astype(int) * (df['is geo'] == 0).astype(int)
    df['TN'] = (df[col_name] < thresh).astype(int) * (df['is geo'] == 0).astype(int)
    df['FN'] = (df[col_name] < thresh).astype(int) * (df['is geo'] == 1).astype(int)

    tp = df['TP'].sum()
    fp = df['FP'].sum()
    tn = df['TN'].sum()
    fn = df['FN'].sum()

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_score = 2 * precision * recall / (precision + recall)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return precision, recall, accuracy, f_score


class ExpressionRow:
    def __init__(self):
        pass


class ExpressionSheet:
    def __init__(self, sheet_name):
        self.expressions = []
        self.sheet_name = sheet_name
        pass


def main():

    expression_lists = []

    xl = pd.ExcelFile(expression_file)
    for sheet_num, sheet_name in enumerate(xl.sheet_names):
        df = xl.parse(sheet_name)
        bar = Bar('Processing worksheet {}'.format(sheet_name), max=df.shape[0])
        weight_adjustment = 1 / 7

        el = ExpressionSheet(sheet_name)
        expression_lists.append(el)

        for row_num, row in df.iterrows():
            ori_expression = row['concatenated expression']
            try:
                is_geo = int(row['geo/or not'])
            except ValueError:
                print('Error at sheet {} and row {}'.format(sheet_name, row_num))
                raise
            expression = escape_text(ori_expression)
            found_terms = {}
            def_and_term_found = []
            term_only_found = {}
            score = 0

            try:
                individual_words = expression.split(' ')
                core_term_index = individual_words.index(sheet_name)

                for term in terms:
                    if term == sheet_name:
                        continue
                    individual_words = expression.split(' ')
                    found_definitives = {}
                    found_terms[term] = found_definitives
                    search_offset = 0
                    while True:
                        try:
                            search_index = individual_words.index(term)
                            if search_index > 0:
                                weight = weights.get(term, None)
                                if weight is not None:
                                    term_index = search_offset + search_index
                                    if term not in term_only_found:
                                        term_only_found[term] = []
                                    term_only_found[term].append(term_index)
                                    score += (weight_adjustment * weight / (abs(term_index - core_term_index)))
                                    # score += weight_adjustment * weight
                                else:
                                    weight_missing.add(term)

                                for definitive in definitives:

                                    # This is the case where we only want to search for the term and not including definitive
                                    if definitive != '***' and individual_words[search_index - 1] == definitive:
                                        found_definitives[definitive] = found_definitives.get(definitive, 0) + 1

                            individual_words = individual_words[search_index + 1:]
                            search_offset += search_index + 1

                        except ValueError:
                            break

                for term, found_definitives in found_terms.items():
                    for definitive, count in found_definitives.items():
                        if definitive != '***':
                            def_and_term_found.append(definitive + ' ' + term)
            except ValueError:
                core_term_index = 'N/A'

            er = ExpressionRow()
            er.expression = ori_expression
            er.core_term_index = core_term_index
            er.num_def_and_terms_found = len(def_and_term_found)
            er.def_and_term_found = ';'.join(def_and_term_found)
            er.num_term_only_found = len(term_only_found)
            er.term_and_indices_found = ';'.join(['{}: {}'.format(term, ','.join(map(str, indices))) for term, indices in term_only_found.items() if len(indices) > 0])  # terms and indices found
            er.score = score
            er.is_geo = is_geo

            el.expressions.append(er)

            bar.next()
        bar.finish()

    for el in expression_lists:
        score_df = pd.DataFrame(columns=headings)
        expressions = el.expressions
        bar = Bar('Creating worksheet {}'.format(el.sheet_name), max=len(expressions))
        for row_num, expression in enumerate(el.expressions):
            score_df.loc[row_num] = [
                expression.expression,  # concatenated expression
                expression.core_term_index,  # core term index
                expression.num_def_and_terms_found,  # def+term count
                expression.def_and_term_found,  # def+terms found
                expression.num_term_only_found,  # term count
                expression.term_and_indices_found,  # terms and indices found
                expression.is_geo,
                expression.score
            ]
            bar.next()
        bar.finish()
        el.df = score_df

    raw_writer = pd.ExcelWriter(raw_output_file, engine='xlsxwriter')

    summary = OrderedDict()

    for el in expression_lists:
        print('Calculating for {}'.format(el.sheet_name))

        if 'score' not in summary:
            summary['score'] = OrderedDict()

        if 'def+term count' not in summary:
            summary['def+term count'] = OrderedDict()

        if 'term count' not in summary:
            summary['term count'] = OrderedDict()

        for row_num, thresh in enumerate(np.arange(0.00, 1, 0.02)):
            if thresh not in summary['score']:
                summary['score'][thresh] = OrderedDict()

            summary['score'][thresh][el.sheet_name] = calc_stats(el.df, 'score', thresh)
            # Write each dataframe to a different worksheet.
            if thresh == 0.5:
                el.df.to_excel(raw_writer, sheet_name=el.sheet_name, index=None)

        for row_num, thresh in enumerate([0, 1, 2, 3, 4, 5]):
            if thresh not in summary['def+term count']:
                summary['def+term count'][thresh] = OrderedDict()
            summary['def+term count'][thresh][el.sheet_name] = calc_stats(el.df, 'def+term count', thresh)

        for row_num, thresh in enumerate([0, 1, 2, 3, 4, 5]):
            if thresh not in summary['term count']:
                summary['term count'][thresh] = OrderedDict()
            summary['term count'][thresh][el.sheet_name] = calc_stats(el.df, 'term count', thresh)

    summary_writer = pd.ExcelWriter(summary_output_file, engine='xlsxwriter')

    for measure, result_by_thresh in summary.items():
        summary_df = pd.DataFrame(columns=['Threshold', 'Term', 'Prec', 'Recall', 'Accu', 'Fscore'])

        row_num = 0
        for thresh, result_by_term in result_by_thresh.items():
            tmp_df = pd.DataFrame(columns=['Prec', 'Recall', 'Accu', 'Fscore'])
            tmp_row_num = 0
            for sheet_name, (precision, recall, accuracy, f_score) in result_by_term.items():
                summary_df.loc[row_num] = [thresh, sheet_name, precision, recall, accuracy, f_score]
                tmp_df.loc[tmp_row_num] = [precision, recall, accuracy, f_score]

                row_num += 1
                tmp_row_num += 1

            # For average
            avg_prec = tmp_df['Prec'].mean(skipna=True)
            avg_recall = tmp_df['Recall'].mean(skipna=True)
            avg_accuracy = tmp_df['Accu'].mean(skipna=True)
            avg_f_score = tmp_df['Fscore'].mean(skipna=True)

            # Average all terms
            summary_df.loc[row_num] = [thresh, 'Average', avg_prec, avg_recall, avg_accuracy, avg_f_score]
            row_num += 1

            # Add an empty line
            summary_df.loc[row_num] = [' '] * summary_df.head().shape[1]
            row_num += 1

        summary_df.to_excel(summary_writer, sheet_name=measure, index=None)
    summary_writer.save()
    print('Results written to {}'.format(summary_output_file))

    # Close the Pandas Excel writer and output the Excel file.
    raw_writer.save()
    print('Raw results written to {}'.format(raw_output_file))

    # print('================SUMMARY================')
    # print('{}\t{}\t{}\t{}\t{}'.format('Term', 'Prec', 'Recall', 'Accu', 'Fscore'))
    # for sheet_name, (precision, recall, accuracy, f_score) in summary.items():
    #     print('{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}'.format(sheet_name, precision, recall, accuracy, f_score))
    #
    return

    result_df_headings = ['Threshold', 'Precision', 'Recall', 'Accuracy', 'F-Score']

    score_output_file = 'files/xlsx/precision-recall-by-score.xlsx'
    score_writer = pd.ExcelWriter(score_output_file, engine='xlsxwriter')

    def_and_term_output_file = 'files/xlsx/precision-recall-by-def-and-term.xlsx'
    def_and_term_writer = pd.ExcelWriter(def_and_term_output_file, engine='xlsxwriter')

    term_output_file = 'files/xlsx/precision-recall-by-term.xlsx'
    term_writer = pd.ExcelWriter(term_output_file, engine='xlsxwriter')

    for el in expression_lists:
        print('Calculating for {}'.format(el.sheet_name))

        score_df = pd.DataFrame(columns=result_df_headings)
        for row_num, thresh in enumerate(np.arange(0.01, 1, 0.1)):
            precision, recall, accuracy, f_score = calc_stats(el.df, 'score', thresh)
            score_df.loc[row_num] = [thresh, precision, recall, accuracy, f_score]

        def_and_term_df = pd.DataFrame(columns=result_df_headings)
        for row_num, thresh in enumerate(np.arange(0, 10, 1)):
            precision, recall, accuracy, f_score = calc_stats(el.df, 'def+term count', thresh)
            def_and_term_df.loc[row_num] = [thresh, precision, recall, accuracy, f_score]

        term_df = pd.DataFrame(columns=result_df_headings)
        for row_num, thresh in enumerate(np.arange(0, 10, 1)):
            precision, recall, accuracy, f_score = calc_stats(el.df, 'term count', thresh)
            term_df.loc[row_num] = [thresh, precision, recall, accuracy, f_score]

        el.score_stats = score_df
        el.def_and_term_stats = def_and_term_df
        el.term_stats = term_df

        # Write each dataframe to a different worksheet.
        score_df.to_excel(score_writer, sheet_name=el.sheet_name, index=None)
        def_and_term_df.to_excel(def_and_term_writer, sheet_name=el.sheet_name, index=None)
        term_df.to_excel(term_writer, sheet_name=el.sheet_name, index=None)

    # Close the Pandas Excel writer and output the Excel file.
    score_writer.save()
    print('Results written to {}'.format(score_output_file))

    def_and_term_writer.save()
    print('Results written to {}'.format(def_and_term_output_file))

    term_writer.save()
    print('Results written to {}'.format(term_output_file))

    if len(weight_missing) > 0:
        warning('The following terms has no weight')
        warning(','.join(weight_missing))


if __name__ == '__main__':
    main()
