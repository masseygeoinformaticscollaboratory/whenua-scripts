import pandas as pd
import numpy as np
from progress.bar import Bar

from util import indices


def shorten(exp, term, winlen_ori, winlen_new):
    words = exp.split(' ')
    last_ind = len(words) - 1
    term_inds = indices(words, term)
    if len(term_inds) == 0:
        raise Exception('What the hell???')

    if len(term_inds) == 1:
        best_ind = term_inds[0]
    else:
        term_inds = np.array(term_inds)
        diffs = np.abs(term_inds + 1 - len(words) / 2)
        smallest_dif_ind = np.argmin(diffs)
        best_ind = term_inds[smallest_dif_ind]

    start_ind = max(0, best_ind - winlen_new) if best_ind < winlen_ori else best_ind - winlen_new
    end_ind = last_ind if best_ind == last_ind else min(last_ind + 1, best_ind + 1 + winlen_new)

    if len(words[start_ind:best_ind]) > winlen_new:
        raise Exception('Here 1')

    if start_ind > best_ind:
        raise Exception('Here 2')

    if len(words[best_ind+1:end_ind]) > winlen_new:
        raise Exception('Here 3')

    if end_ind < best_ind + 1:
        raise Exception('Here 4')

    new_words = words[start_ind:best_ind] + words[best_ind:end_ind]
    return ' '.join(new_words)


def main(w20file, w8file):
    writer = pd.ExcelWriter(w8file, engine='xlsxwriter')
    headings = ['concatenated expression', 'geo/or not']
    w20xl = pd.ExcelFile(w20file)
    for sheet_name in w20xl.sheet_names:
        df = w20xl.parse(sheet_name)
        output_df = pd.DataFrame(columns=headings)
        bar = Bar('Processing worksheet {}'.format(sheet_name), max=df.shape[0])
        for row_num, row in df.iterrows():
            expression = row['concatenated expression']
            is_geo = row['geo/or not']

            new_expression = shorten(expression, sheet_name, 20, 10)
            output_df.loc[row_num] = [new_expression, is_geo]
            bar.next()
        bar.finish()
        output_df.to_excel(writer, sheet_name=sheet_name, index=None)
    writer.save()
    print('Result saved to ' + w8file)


if __name__ == '__main__':
    word_window_20_input_file = 'files/xlsx/Precisiion-Recall our corpus.xlsx'
    word_window_8_input_file = 'files/xlsx/Precisiion-Recall our corpus - 10.xlsx'
    main(word_window_20_input_file, word_window_8_input_file)
