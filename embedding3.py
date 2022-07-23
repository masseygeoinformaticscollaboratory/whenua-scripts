import argparse
import os.path
import pickle
import time
from logging import warning

import numpy as np
import pandas as pd
import scipy.spatial as sp
from progress.bar import Bar

from AbstractCommand import AbstractCommand
from embedding2 import get_embeddings, extract_reference_terms, extract_db_expressions, get_terms, save_terms
from taumahi import Taumahi
from util import clean_exp_and_remove_stopwords

reference_xlsx = 'files/both_together_10_words.xlsx'
embedding_file = 'embedding-None-None.csv'

all_terms = ['whenua', 'roto', 'rawa', 'take', 'rua', 'puta', 'ara', 'utu', 'tuku', 'kaupapa', 'wai', 'wa', 'motu',
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


all_terms = set([clean_exp_and_remove_stopwords(x) for x in all_terms])


def _replace_score_with_maori_percentage_if_necessary (taumahi, rows):
    if taumahi is None:
        return rows

    for row in rows:
        expr = row[0]
        _, _, _, _, maori_percentage = taumahi.tiki_ōrau(expr)
        row[2] = maori_percentage

    return rows


def extract_most_similar(cache_dir, ref_keywords, all_terms, window_size, score_type, k):
    if score_type == 'maori_percentage':
        taumahi = Taumahi(verbose=False)
        columns = ['Expression', 'geo/or not', 'Maori percentage']
    else:
        taumahi = None
        columns = ['Expression', 'geo/or not', 'Score']

    if k == 1:
        columns += ['Most similar', 'Term']

    output_file = 'files/embedding3-{}.k={}-{}.xlsx'.format(window_size, k, score_type)
    writer = pd.ExcelWriter(output_file, engine='xlsxwriter')
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
            reference_matrix.append(expr.avg_embbedding)
            corresponding_exps.append(expr)
    reference_matrix = np.array(reference_matrix, dtype=float)

    xlsx_bar = Bar('Exporting to Excel {}'.format(output_file), max=len(cached_terms) + len(remaining_db_terms))

    for keyword in cached_terms:
        if window_size == 10:
            cache_file = os.path.join(cache_dir, 'rows.{}.k={}.pkl'.format(keyword, k))
        else:
            cache_file = os.path.join(cache_dir, 'rows.{}-{}.k={}.pkl'.format(keyword, window_size, k))
        with open(cache_file, 'rb') as f:
            rows = pickle.load(f)

        rows = _replace_score_with_maori_percentage_if_necessary(taumahi, rows)
        df = pd.DataFrame(columns=columns, data=rows)
        df.to_excel(writer, sheet_name=keyword, index=None)
        xlsx_bar.next()

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
            if np.any(np.isnan(expr.avg_embbedding)):
                warning('Expression ' + expr_full + ' has nan avg embedding')
                row = [expr_full, 'N/A', 0]
                if k == 1:
                    row.append('')
                    row.append('')
                rows.append(row)
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
        df.to_excel(writer, sheet_name=keyword, index=None)
        xlsx_bar.next()

    xlsx_bar.finish()
    writer.save()
    print('Finished extracting most similar terms and score. Output is at ' + output_file)


class Command(AbstractCommand):

    def __init__(self):
        super(Command, self).__init__(__file__)

    def run(self, window_size, k, score_type):
        embedded_terms, embeddings = get_embeddings()

        embedded_terms = {term: index for index, term in enumerate(embedded_terms)}

        ref_keywords = extract_reference_terms(self.cache_dir)
        extract_db_expressions(self.cache_dir, all_terms, window_size)

        bar = Bar('Calculating avg embedding for reference terms', max=len(ref_keywords),
                  suffix='%(index)d/%(max)d Elapsed %(elapsed)ds - ETA %(eta)ds')

        for keyword in ref_keywords:
            term_ref = get_terms(self.cache_dir, 'ref', keyword, None)
            for exp in term_ref.expressions:
                if exp.avg_embbedding is None:
                    exp.calc_avg_embedding(embedded_terms, embeddings)
            bar.next()

            save_terms(self.cache_dir, 'ref', term_ref, None)
        bar.finish()

        bar = Bar('Calculating avg embedding for all terms', max=len(all_terms),
                  suffix='%(index)d/%(max)d Elapsed %(elapsed)ds - ETA %(eta)ds')

        for keyword in all_terms:
            term_db = get_terms(self.cache_dir, 'db', keyword, window_size)
            for exp in term_db.expressions:
                if exp.avg_embbedding is None:
                    exp.calc_avg_embedding(embedded_terms, embeddings)
            bar.next()

            save_terms(self.cache_dir, 'db', term_db, window_size)
        bar.finish()

        extract_most_similar(self.cache_dir, ref_keywords, all_terms, window_size, score_type, k)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--score-type', dest='score_type', action='store', type=str, default='similarity',
                        choices=['similarity', 'maori_percentage'])
    parser.add_argument('--window', dest='window_size', action='store', type=int, required=True)
    parser.add_argument('--k', dest='k', action='store', type=int, required=True)
    args = parser.parse_args()

    command = Command()
    start = time.time()
    command.run(window_size=args.window_size, k=args.k, score_type=args.score_type)
    end = time.time()
    print('Took {} seconds'.format(end - start))
