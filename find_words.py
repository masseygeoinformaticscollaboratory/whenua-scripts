import pandas as pd
from progress.bar import Bar

word_window_file = 'files/xlsx/word_and_definitive_windows_list_output.csv'

terms_file = 'files/xlsx/word_windows_list.xlsx'

output_file = 'files/xlsx/word_and_definitive_windows_list_term_searched.csv'

term_df = pd.read_excel(terms_file)

terms = []
definitives = ['***']

for row_num, row in term_df.iterrows():
    term = row['Maori word']
    terms.append(term)
    definitive = row['Definitive']
    if isinstance(definitive, str):
        definitives.append(definitive)


with open(output_file, 'w') as f:
    f.write('Keyword\t20 words before\t20 words after\tCount\tFound terms\n')

    word_window_df = pd.read_csv(word_window_file, delimiter='\t')
    bar = Bar('Count term appearances', max=word_window_df.shape[0])

    for row_num, row in word_window_df.iterrows():
        before20 = row['20 words before']
        after20 = row['20 words after']
        keyword = row['Keyword']
        total_count = 0

        combined = ''
        if not isinstance(before20, str):
            before20 = ''

        if not isinstance(after20, str):
            after20 = ''

        combined = before20 + ' ' + after20
        combined = combined.strip()

        found_terms = {}

        for term in terms:
            individual_words = combined.split(' ')
            found_definitives = {}
            found_terms[term] = found_definitives
            while True:
                try:
                    search_index = individual_words.index(term)
                    if search_index > 0:
                        for definitive in definitives:

                            # This is the case where we only want to search for the term and not including definitive
                            if definitive == '***':
                                found_definitives[definitive] = found_definitives.get(definitive, 0) + 1

                            elif individual_words[search_index - 1] == definitive:
                                found_definitives[definitive] = found_definitives.get(definitive, 0) + 1

                    individual_words = individual_words[search_index + 1:]

                except ValueError:
                    break

        found_terms_arr = []
        for term, found_definitives in found_terms.items():
            for definitive, count in found_definitives.items():
                total_count += count
                if definitive == '***':
                    term_definitive = term
                else:
                    term_definitive = definitive + ' ' + term
                found_terms_arr.append(term_definitive)

        f.write(keyword)
        f.write('\t')
        f.write(before20)
        f.write('\t')
        f.write(after20)
        f.write('\t')
        f.write(str(total_count))
        f.write('\t')
        f.write('*'.join(found_terms_arr))
        f.write('\n')

        bar.next()
    bar.finish()

print('Output written to ' + output_file)
