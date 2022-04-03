from logging import warning

import pandas as pd
from progress.bar import Bar

from util import escape_text

expression_file = 'files/xlsx/Expressions for precision-recall.xlsx'

terms_file = 'files/xlsx/word_windows_list.xlsx'

weights_file = 'files/xlsx/weights.xlsx'

output_file = 'files/xlsx/Expressions-for-precision-recall-output-weighted.xlsx'


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
headings = ['concatenated expression', 'core term index', 'def+term count', 'def+terms found', 'term count', 'terms and indices found', 'score']
# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter(output_file, engine='xlsxwriter')

xl = pd.ExcelFile(expression_file)
for sheet_name in xl.sheet_names:
    df = xl.parse(sheet_name)
    output_df = pd.DataFrame(columns=headings)
    bar = Bar('Processing worksheet {}'.format(sheet_name), max=df.shape[0])
    weight_adjustment = 1 / 7

    for row_num, row in df.iterrows():
        ori_expression = row['concatenated expression']
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
                term_only_found[term] = []
                search_offset = 0
                while True:
                    try:
                        search_index = individual_words.index(term)
                        if search_index > 0:
                            weight = weights.get(term, None)
                            if weight is not None:
                                term_index = search_offset + search_index
                                term_only_found[term].append(term_index)
                                score += (weight_adjustment * weight / (abs(term_index - core_term_index)))
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

        df_row = [
            ori_expression,  # concatenated expression
            core_term_index,  # core term index
            len(def_and_term_found),  # def+term count
            ';'.join(def_and_term_found),  # def+terms found
            len(term_only_found),  # term count
            ';'.join(['{}: {}'.format(term, ','.join(map(str, indices))) for term, indices in term_only_found.items() if len(indices) > 0]),  # terms and indices found
            score
        ]

        output_df.loc[row_num] = df_row
        bar.next()
    bar.finish()

    # Write each dataframe to a different worksheet.
    output_df.to_excel(writer, sheet_name=sheet_name, index=None)

# Close the Pandas Excel writer and output the Excel file.
writer.save()

print('Results written to {}'.format(output_file))

if len(weight_missing) > 0:
    warning('The following terms has no weight')
    warning(','.join(weight_missing))

