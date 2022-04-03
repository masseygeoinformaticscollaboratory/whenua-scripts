import pandas as pd
from progress.bar import Bar

word_window_file = 'files/xlsx/word_windows_list_output.csv'

terms_file = 'files/xlsx/word_windows_list.xlsx'

output_file = 'files/xlsx/word_windows_list_term_searched2.csv'

term_df = pd.read_excel(terms_file)

terms = []

for row_num, row in term_df.iterrows():
    term = row['Maori word']
    terms.append(term)


with open(output_file, 'w') as f:
    f.write('Keyword\t20 words before\t20 words after\tCount\tFound terms\n')

    word_window_df = pd.read_csv(word_window_file, delimiter='\t')
    bar = Bar('Count term appearances', max=word_window_df.shape[0])

    for row_num, row in word_window_df.iterrows():
        before20 = row['20 words before']
        after20 = row['20 words after']
        keyword = row['Keyword']
        count = 0
        found_terms = []

        combined = ''
        if not isinstance(before20, str):
            before20 = ''

        if not isinstance(after20, str):
            after20 = ''

        combined = before20 + ' ' + after20
        combined = combined.strip()

        individual_words = combined.split(' ')

        for term in terms:
            try:
                individual_words.index(term)
                exists = True
                found_terms.append(term)
            except ValueError:
                exists = False

            if exists:
                count += 1

        f.write(keyword)
        f.write('\t')
        f.write(before20)
        f.write('\t')
        f.write(after20)
        f.write('\t')
        f.write(str(count))
        f.write('\t')
        f.write('*'.join(found_terms))
        f.write('\n')

        bar.next()
    bar.finish()

print('Output written to ' + output_file)
