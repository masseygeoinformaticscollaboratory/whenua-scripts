import pandas as pd
from progress.bar import Bar

from util import escape_text

expression_file = 'files/xlsx/Expressions for precision-recall.xlsx'

terms_file = 'files/xlsx/word_windows_list.xlsx'

output_file = 'files/xlsx/Expressions-for-precision-recall-output.xlsx'

term_df = pd.read_excel(terms_file)

terms = []
definitives = ['***']


for row_num, row in term_df.iterrows():
    term = row['Maori word']
    terms.append(term)
    definitive = row['Definitive']
    if isinstance(definitive, str):
        definitives.append(definitive.lower())


# output_dfs = OrderedDict()
headings = ['concatenated expression', 'def+term count', 'def+terms found', 'term count', 'terms found']
# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter(output_file, engine='xlsxwriter')

xl = pd.ExcelFile(expression_file)
for sheet_name in xl.sheet_names:
    df = xl.parse(sheet_name)
    output_df = pd.DataFrame(columns=headings)
    bar = Bar('Processing worksheet {}'.format(sheet_name), max=df.shape[0])

    for row_num, row in df.iterrows():
        ori_expression = row['concatenated expression']
        expression = escape_text(ori_expression)

        found_terms = {}

        for term in terms:
            individual_words = expression.split(' ')
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

        def_and_term_found = []
        term_only_found = []

        for term, found_definitives in found_terms.items():
            for definitive, count in found_definitives.items():
                if definitive == '***':
                    term_only_found.append(term)
                else:
                    def_and_term_found.append(definitive + ' ' + term)

        df_row = [ori_expression, len(def_and_term_found), ';'.join(def_and_term_found), len(term_only_found), ';'.join(term_only_found)]
        output_df.loc[row_num] = df_row
        bar.next()
    bar.finish()

    # Write each dataframe to a different worksheet.
    output_df.to_excel(writer, sheet_name=sheet_name, index=None)

# Close the Pandas Excel writer and output the Excel file.
writer.save()


