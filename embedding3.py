import argparse
import os.path
import pickle
import time
from logging import warning

import numpy as np
import pandas as pd
import scipy.spatial as sp
import unidecode
from progress.bar import Bar

from AbstractCommand import AbstractCommand
from embedding2 import get_embeddings, extract_reference_terms, extract_db_expressions, get_terms, save_terms
from taumahi import Taumahi
from util import clean_exp_and_remove_stopwords

reference_xlsx = 'files/both_together_19_words.xlsx'
embedding_file = 'embedding-None-None.csv'


def get_all_terms():
    terms = ['whenua', 'roto', 'rawa', 'take', 'rua', 'puta', 'ara', 'utu', 'tuku', 'kaupapa', 'wai', 'wa', 'motu',
             'wāhi', 'rere', 'rohe', 'awa', 'pā', 'huarahi', 'moana', 'tai', 'papa', 'one', 'haka', 'taumata', 'pae',
             'puke', 'putanga', 'uta', 'mata', 'kōhanga', 'toi', 'pū', 'puna', 'nohoanga', 'moutere', 'tauranga',
             'kawa', 'waha', 'takiwā', 'koi', 'koro', 'ranga', 'ngahere', 'teitei', 'whakahua', 'horo', 'peka', 'tara',
             'toka', 'tomo', 'puni', 'whanga', 'rae', 'io', 'para', 'maka', 'ngutu', 'pakihi', 'manga', 'pāti',
             'takutai', 'tihi', 'pari', 'paru', 'aka', 'roma', 'tuawhenua', 'hiwi', 'whāiti', 'parihaka', 'taiwhanga',
             'tūāpapa', 'repo', 'ipu', 'kōhatu', 'urupa', 'raupapa', 'āpiti', 'papawai', 'rarawa', 'arero', 'puia',
             'tahataha', 'pukenga', 'waiariki', 'mania', 'rei', 'piko', 'huanui', 'wao', 'koraha', 'riu', 'hū',
             'muriwai', 'kirikiri', 'ngutuawa', 'tawa', 'koko', 'tārua', 'mātāpuna', 'pakihiwi', 'paeroa', 'kakari',
             'te nuku', 'tōpito', 'awarua', 'whirinaki', 'ngā papa', 'tātahi', 'tapoko', 'awaawa', 'rea', 'tahatika',
             'matamata', 'pohatu', 'ākau', 'taiwhenua', 'moka', 'kikī', 'huahua', 'ripa', 'kopi', 'wairere', 'raorao',
             'taratara', 'mato', 'tuahiwi', 'pokohiwi', 'wahapū', 'maioro', 'kira', 'pākihi', 'whakarua', 'raetihi',
             'whārua', 'kokoru', 'haupapa', 'koutu', 'kokorutanga', 'onepū', 'mārua', 'rapaki', 'paparahi', 'ngāwhā',
             'tahā', 'horohoro', 'taumutu', 'wāpu', 'te koko', 'tāhuna', 'kāpiti', 'awakeri', 'pōhatu', 'mānia', 'ripo',
             'ngae', 'toitoi', 'mātārae', 'pahī', 'paripari', 'mauka', 'awapuni', 'waipara', 'pūaha', 'tairua',
             'tahitahi', 'pūkawa', 'tāheke', 'kahiwi', 'takau', 'hawai', 'ngaruru', 'kūrae', 'rehutai', 'hongere',
             'mātā', 'kahupapa', 'pīnakitanga', 'mitimiti', 'kōmata', 'koeko', 'kumete', 'hīrere', 'hāpua', 'arawai',
             'hawe', 'tokatoka', 'kōtihitihi', 'taieri', 'rerewē', 'taukaka', 'tawhā', 'tahatai', 'ika whenua',
             'pūwaha', 'tītōhea', 'kauanga', 'makatea', 'motuiti', 'hāroto', 'ngahu', 'tarahanga', 'harapaki', 'mātātā',
             'taiheke', 'nonoti', 'tāpere', 'kāpeka', 'kōawaawa', 'ahi tipua', 'taiari', 'taupae', 'pāraharaha',
             'aupaki', 'reporepo', 'torouka', 'pakohu', 'kōtihi', 'kopia', 'taihua', 'arapiki', 'kūititanga', 'awakari',
             'taone matua', 'kuinga', 'hūhi', 'kakaritanga', 'pūau', 'pūroto', 'korio', 'ririno', 'tahawai', 'pīpīwai',
             'tauwharenga', 'kahaka', 'matiri', 'pararahi', 'tāpuhipuhi', 'parehua', 'tāwhārua', 'kororipo', 'kaurapa',
             'kaimanga', 'ararua', 'nohoaka', 'ara kūiti', 'karahiwi', 'nukuao']

    return set([unidecode.unidecode(x) for x in terms])


def perf_measure(expected_values, actual_values):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for expected, actual in zip(expected_values, actual_values):
        if expected:
            if actual:
                TP += 1
            else:
                FN += 1
        else:
            if actual:
                FP += 1
            else:
                TN += 1

    return TP, FP, TN, FN


def cross_validate(cache_dir, ref_keywords, k=1):
    output_file = 'files/embedding_cv.xlsx'
    writer = pd.ExcelWriter(output_file, engine='xlsxwriter')

    reference_matrix = []
    corresponding_exps = []
    for keyword in ref_keywords:
        term_ref = get_terms(cache_dir, 'ref', keyword, None)

        for expr in term_ref.expressions:
            if len(expr.avg_embbedding) == 0:
                continue
            reference_matrix.append(expr.avg_embbedding)
            corresponding_exps.append(expr)
    reference_matrix = np.array(reference_matrix, dtype=float)

    expr_ind = 0
    expected = []
    actual = []

    for keyword in ref_keywords:
        rows = []
        term_ref = get_terms(cache_dir, 'ref', keyword, None)
        bar = Bar('Extracting most similar for term {}'.format(keyword), max=len(term_ref.expressions),
                  suffix='%(index)d/%(max)d Elapsed %(elapsed)ds - ETA %(eta)ds')
        for expr in term_ref.expressions:
            if len(expr.avg_embbedding) == 0:
                warning('Expression ' + expr.get_full_expr() + ' has nan avg embedding')
                row = [expr.get_full_expr(), expr.is_geo, 'N/A', 0, '', '']
                rows.append(row)
                bar.next()
                expr_ind += 1
                continue

            expr_embedding = np.array([expr.avg_embbedding], dtype=float)
            similarities = 1 - sp.distance.cdist(expr_embedding, reference_matrix, 'cosine')
            similarities = similarities.squeeze()

            closest_expr = None
            sort_inds = np.flip(np.argsort(similarities))
            k_matches = sort_inds[:k]
            scores = similarities[k_matches]
            scores_is_geo = []
            closest_exprs = [corresponding_exps[x] for x in k_matches]
            is_geos = []
            for score, closest_expr in zip(scores, closest_exprs):
                if closest_expr.is_geo:
                    is_geos.append(1)
                    scores_is_geo.append(score)
                else:
                    is_geos.append(0)
            score = np.mean(scores_is_geo)

            is_geos = np.array(is_geos)
            is_geo = False if len(is_geos) == 0 else is_geos.mean() >= 0.5
            row = [expr.get_full_expr(), 1 if expr.is_geo else 0, 1 if is_geo else 0, score]
            row.append(closest_expr.get_full_expr())
            row.append(closest_expr.term.core)
            rows.append(row)
            expected.append(expr.is_geo)
            actual.append(is_geo)

            expr_ind += 1
            bar.next()
        bar.finish()

        columns = ['Expression', 'original geo/or not', 'validated geo/or not', 'Score', 'Most similar', 'Term']

        df = pd.DataFrame(columns=columns, data=rows)
        df.to_excel(writer, sheet_name=keyword, index=None)

    writer.save()

    tp, fp, tn, fn = perf_measure(expected, actual)
    precision = tp / (fp + tp)
    recall = tp / (fn + tp)
    accuracy = (tp + tn) / (tp + fn + tn + fp)
    f1 = 2 * precision * recall / (precision + recall)

    print('Finished extracting most similar terms and score. Output is at ' + output_file)
    print('TP={} FP={} TN={} FN={} Precision={} Recall={} Accuracy={} F1={}'.format(tp, fp, tn, fn, precision, recall, accuracy, f1))


def _replace_score_with_maori_percentage_if_necessary(taumahi, rows):
    if taumahi is None:
        return rows

    for row in rows:
        expr = row[0]
        _, _, _, _, maori_percentage = taumahi.tiki_ōrau(expr)
        row[2] = maori_percentage

    return rows


def extract_most_similar(cache_dir, extract_type, ref_keywords, all_terms, window_size, score_type, k, combined):
    if score_type == 'maori_percentage':
        taumahi = Taumahi(verbose=False)
        score_column_name = 'Maori percentage'

    else:
        taumahi = None
        score_column_name = 'Score'

    columns = ['Expression', 'geo/or not', score_column_name]
    if k == 1:
        columns += ['Most similar', 'Term']

    if combined:
        output_file = 'files/embedding-{}-{}.k={}-{}-combined.xlsx'.format(extract_type, window_size, k, score_type)
    else:
        output_file = 'files/embedding-{}-{}.k={}-{}.xlsx'.format(extract_type, window_size, k, score_type)

    remaining_db_terms = []
    cached_terms = []
    for keyword in all_terms:
        if window_size == 10:
            cache_file = os.path.join(cache_dir, 'rows.{}.k={}.pkl'.format(keyword, k))
        else:
            cache_file = os.path.join(cache_dir, 'rows.{}-{}.k={}.pkl'.format(keyword, window_size, k))
        if not os.path.isfile(cache_file):
            remaining_db_terms.append(keyword)
        else:
            cached_terms.append(keyword)

    reference_matrix = []
    corresponding_exps = []
    for keyword in ref_keywords:
        term_ref = get_terms(cache_dir, 'ref', keyword, None)

        for expr in term_ref.expressions:
            if len(expr.avg_embbedding) == 0:
                continue
            reference_matrix.append(expr.avg_embbedding)
            corresponding_exps.append(expr)
    reference_matrix = np.array(reference_matrix, dtype=float)

    ws = {}

    bar = Bar('Read cached term', max=len(cached_terms),
              suffix='%(index)d/%(max)d Elapsed %(elapsed)ds - ETA %(eta)ds')
    for keyword in cached_terms:
        if window_size == 10:
            cache_file = os.path.join(cache_dir, 'rows.{}.k={}.pkl'.format(keyword, k))
        else:
            cache_file = os.path.join(cache_dir, 'rows.{}-{}.k={}.pkl'.format(keyword, window_size, k))
        with open(cache_file, 'rb') as f:
            rows = pickle.load(f)

        bar.next()
        rows = _replace_score_with_maori_percentage_if_necessary(taumahi, rows)
        df = pd.DataFrame(columns=columns, data=rows)
        ws[keyword] = df
    bar.finish()

    for keyword in remaining_db_terms:
        if window_size == 10:
            cache_file = os.path.join(cache_dir, 'rows.{}.k={}.pkl'.format(keyword, k))
        else:
            cache_file = os.path.join(cache_dir, 'rows.{}-{}.k={}.pkl'.format(keyword, window_size, k))
        rows = []
        term_db = get_terms(cache_dir, 'db', keyword, window_size)
        bar = Bar('Extracting most similar for term {}'.format(keyword), max=len(term_db.expressions),
                  suffix='%(index)d/%(max)d Elapsed %(elapsed)ds - ETA %(eta)ds')
        for expr_ind, expr in enumerate(term_db.expressions):
            expr_full = expr.get_full_expr()
            if len(expr.avg_embbedding) == 0:
                # warning('Expression ' + expr_full + ' has nan avg embedding')
                # row = [expr_full, 'N/A', 0]
                # if k == 1:
                #     row.append('')
                #     row.append('')
                # rows.append(row)
                bar.next()
                continue
            expr_embedding = np.array([expr.avg_embbedding], dtype=float)
            similarities = 1 - sp.distance.cdist(expr_embedding, reference_matrix, 'cosine')
            similarities = similarities.squeeze()

            score = None
            closest_expr = None
            is_geos = []
            sort_inds = np.flip(np.argsort(similarities))
            if k == 1:
                for sort_ind in sort_inds:
                    similarity = similarities[sort_ind]
                    if score is not None and score != similarity:
                        break
                    score = similarity
                    closest_expr = corresponding_exps[sort_ind]
                    is_geos.append(1 if closest_expr.is_geo else 0)
            else:
                k_matches = sort_inds[:k]
                scores = similarities[k_matches]
                scores_is_geo = []
                closest_exprs = [corresponding_exps[x] for x in k_matches]
                is_geos = []
                for score, closest_expr in zip(scores, closest_exprs):
                    if closest_expr.is_geo:
                        is_geos.append(1)
                        scores_is_geo.append(score)
                    else:
                        is_geos.append(0)
                score = np.mean(scores_is_geo)

            if taumahi is not None:
                _, _, _, _, score = taumahi.tiki_ōrau(expr_full)

            is_geos = np.array(is_geos)
            is_geo = False if len(is_geos) == 0 else is_geos.mean() >= 0.5
            row = [expr_full, 1 if is_geo else 0, score]
            if k == 1:
                row.append(closest_expr.get_full_expr())
                row.append(closest_expr.term.core)
            rows.append(row)
            bar.next()
        bar.finish()
        with open(cache_file, 'wb') as f:
            print('Saving a cache for term {} at {}'.format(keyword, cache_file))
            pickle.dump(rows, f)

        rows = _replace_score_with_maori_percentage_if_necessary(taumahi, rows)
        df = pd.DataFrame(columns=columns, data=rows)
        ws[keyword] = df

    writer = pd.ExcelWriter(output_file, engine='xlsxwriter')
    if combined:
        dfs = []
        xlsx_bar = Bar('Processing all tabs into one tab', max=len(ws))
        for sheet_name, df in ws.items():
            df['Keyword'] = sheet_name
            dfs.append(df)
            xlsx_bar.next()
        xlsx_bar.finish()

        combined_df = pd.concat(dfs)
        worksheets = combined_df.groupby(np.arange(len(combined_df.index)) // 1000000)

        xlsx_bar = Bar('Exporting to Excel {}'.format(output_file), max=len(worksheets))
        for (sheet_num, df) in worksheets:
            df.to_excel(writer, sheet_name='Sheet {}'.format(sheet_num + 1), index=None)
            xlsx_bar.next()
        xlsx_bar.finish()
    else:
        xlsx_bar = Bar('Exporting to Excel {}'.format(output_file), max=len(ws))
        for sheet_name, df in ws.items():
            df.to_excel(writer, sheet_name=sheet_name, index=None)
            xlsx_bar.next()
        xlsx_bar.finish()
    writer.save()
    print('Finished extracting most similar terms and score. Output is at ' + output_file)


class Command(AbstractCommand):

    def __init__(self):
        super(Command, self).__init__(__file__)

    def run(self, args):
        extract_type = args.extract_type
        window_size = args.window_size
        k = args.k
        score_type = args.score_type
        recalculate_embedding = args.recalculate_embedding
        combined = args.combined

        embedded_terms, embeddings = get_embeddings()

        embedded_terms = {term: index for index, term in enumerate(embedded_terms)}

        ref_keywords = extract_reference_terms(self.cache_dir, reference_xlsx)
        if extract_type == 'ref':
            all_terms = ref_keywords
        elif extract_type == 'all':
            all_terms = get_all_terms()
        else:
            all_terms = []

        extract_db_expressions(self.cache_dir, all_terms, window_size)
        bar = Bar('Calculating avg embedding for reference terms', max=len(ref_keywords),
                  suffix='%(index)d/%(max)d Elapsed %(elapsed)ds - ETA %(eta)ds')

        for keyword in ref_keywords:
            need_save = False
            term_ref = get_terms(self.cache_dir, 'ref', keyword, None)
            for exp in term_ref.expressions:
                if exp.avg_embbedding is None or recalculate_embedding:
                    exp.calc_avg_embedding(embedded_terms, embeddings)
                    need_save = True
            bar.next()

            if need_save:
                save_terms(self.cache_dir, 'ref', term_ref, None)
        bar.finish()

        bar = Bar('Calculating avg embedding for all terms', max=len(all_terms),
                  suffix='%(index)d/%(max)d Elapsed %(elapsed)ds - ETA %(eta)ds')

        for keyword in all_terms:
            need_save = False
            term_db = get_terms(self.cache_dir, 'db', keyword, window_size)
            for exp in term_db.expressions:
                if exp.avg_embbedding is None or recalculate_embedding:
                    exp.calc_avg_embedding(embedded_terms, embeddings)
                    need_save = True
            bar.next()

            if need_save:
                save_terms(self.cache_dir, 'db', term_db, window_size)
        bar.finish()

        if extract_type == 'cv':
            cross_validate(self.cache_dir, ref_keywords, k)
        else:
            extract_most_similar(self.cache_dir, extract_type, ref_keywords, all_terms, window_size, score_type, k, combined)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--extract-type', dest='extract_type', action='store', type=str, default='all',
                        choices=['all', 'ref', 'cv'])
    parser.add_argument('--score-type', dest='score_type', action='store', type=str, default='similarity',
                        choices=['similarity', 'maori_percentage'])
    parser.add_argument('--window', dest='window_size', action='store', type=int, required=True)
    parser.add_argument('--k', dest='k', action='store', type=int, required=True)
    parser.add_argument('--recalculate-embedding', dest='recalculate_embedding', action='store_true', default=False)
    parser.add_argument('--combined', dest='combined', action='store_true', default=False)
    args = parser.parse_args()

    command = Command()
    start = time.time()
    command.run(args)
    end = time.time()
    print('Took {} seconds'.format(end - start))


if __name__ == '__main__':
    main()
