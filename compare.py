import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse
from db import InvertedIndex
from collections import Counter
from audfprint import audfprint_analyze
import rankaggregation as ra



parser = argparse.ArgumentParser("Builder")
parser.add_argument('--dataset', type=str, default='./datasets/fingerprints/gtzan.txt', help='dataset file')
parser.add_argument('--results', type=str, default='./exp/gtzan/', help='output directory')
parser.add_argument('--db_idx', type=int, default=0, help='database id')
parser.add_argument('--db_port', type=int, default=6379, help='database port')
parser.add_argument('--db_url', type=str, default='localhost', help='database url')
parser.add_argument('--tau', type=float, default=0, help='threshold value')
args = parser.parse_args()

CONV = 256/11025  # to convert audfprint frames to time


df_queries = pd.read_csv(os.path.join(args.results, 'queries_hashes.csv'))
df_filenames = pd.read_csv(os.path.join(args.results, 'queries.csv'))
df_dataset = pd.read_csv(os.path.join(args.dataset), header=None).reset_index()
df_dataset.columns = ['file_idx', 'filename']
df_dataset.set_index('filename', inplace=True)
df_final = pd.DataFrame(columns=['methd', 'query_idx', 'size', 'tau', 'payload_size', 'correct'])



tf_df = pd.read_csv(os.path.join(args.results, 'tf.csv'))
tf_df.set_index('file_idx', inplace=True)
index = InvertedIndex(args.db_url, args.db_port, args.db_idx)
index.load(args.results)


def calc_query_tfidf(query_idx):
    #conds = (df_queries.idx == query_idx) & (df_queries['size'] <= 9)
    terms_df = df_queries.loc[query_idx].reset_index(drop=False)
    tf_dict = Counter(terms_df.hash.tolist())
    terms_df['tf'] = terms_df.hash.apply(lambda x: tf_dict[x])
    #import code; code.interact(local=dict(globals(), **locals()))
    terms_df['idf'] = index.idfs[terms_df.hash].values  # terms_df.hash.apply(lambda x: index.idf(x))
    terms_df['tf_idf'] = terms_df['tf'] * terms_df['idf']
    terms = terms_df[terms_df.tf_idf > tau].hash.unique().tolist()
    terms_tfidf = terms_df[terms_df.tf_idf > tau].set_index(['hash']).loc[terms].tf_idf.tolist()
    dict_tfidf = dict(zip(terms, terms_tfidf))
    return terms, dict_tfidf, terms_tfidf


def calc_cand_tf_idf(cands):
    #import code; code.interact(local=dict(globals(), **locals()))
    cand_idxs = df_dataset.loc[cands.filename].file_idx.values
    cand_terms = tf_df.loc[cand_idxs].reset_index(drop=False)

    cand_terms['idf'] = index.idfs[cand_terms.term].values
    cand_terms['tf_idf'] = cand_terms.tf * cand_terms.idf
    return cand_terms

def calc_score_wang(query_idx, query_filename, df_results, terms):
    df_alignment = df_results.copy()
    query_hashes = audfprint_analyze.hashes_load(query_filename)
    df_q = pd.DataFrame(query_hashes, columns=['frame', 'hash'])
    df_q = df_q[df_q.hash.isin(terms)].set_index('hash')
    def calc_alignment(x):
        cand_hashes = audfprint_analyze.hashes_load(x)
        df_c = pd.DataFrame(cand_hashes, columns=['frame', 'hash'])
        df_c = df_c[df_c.hash.isin(terms)].set_index('hash')
        df_m = pd.merge(df_q, df_c, on='hash')
        df_m['offset'] = df_m['frame_y'] - df_m['frame_x']
        return df_m.groupby('offset').count().frame_x.max()
    df_alignment['score'] = df_alignment.filename.apply(calc_alignment)
    dfaln = df_alignment.sort_values('score', ascending=False)
    result_filename = dfaln.iloc[0].filename
    is_correct = int(query_filename == result_filename)
    df_final.loc[len(df_final)] = ['WNG', query_idx[0], query_idx[1], tau, len(df_results), is_correct]
    return dfaln

def calc_score_cnt(query_idx, query_filename, df_results, terms):
    df_alignment = df_results.copy()
    query_hashes = audfprint_analyze.hashes_load(query_filename)
    df_q = pd.DataFrame(query_hashes, columns=['frame', 'hash'])
    df_q = df_q[df_q.hash.isin(terms)].set_index('hash')
    def calc_alignment_cnt(x):
        cand_hashes = audfprint_analyze.hashes_load(x)
        df_c = pd.DataFrame(cand_hashes, columns=['frame', 'hash'])
        df_c = df_c[df_c.hash.isin(terms)].set_index('hash')
        df_m = pd.merge(df_q, df_c, on='hash')
        return len(df_m)
    df_alignment['score'] = df_alignment.filename.apply(calc_alignment_cnt)
    result_filename = df_alignment.sort_values('score', ascending=False).iloc[0].filename
    is_correct = int(query_filename == result_filename)
    df_final.loc[len(df_final)] = ['CNT', query_idx[0], query_idx[1], tau, len(df_results), is_correct]


def calc_score_cosine_sim(query_idx, query_filename, df_results):
    dd = calc_cand_tf_idf(df_results)
    df_itsc = dd.loc[dd.term.isin(terms)].reset_index(drop=True)
    df_itsc['q_tf_idf'] = df_itsc.term.apply(lambda x: dict_tfidf[x])
    df_itsc['tf_idf_prod'] = df_itsc['tf_idf'] * df_itsc['q_tf_idf']
    dot_products = df_itsc.groupby('file_idx').sum().tf_idf_prod

    dd['tf_idf_2'] = dd['tf_idf'] ** 2
    norm_a = np.sqrt(np.power(terms_tfidf, 2).sum())
    denom = norm_a * np.sqrt(dd.groupby('file_idx').sum().tf_idf_2)
    cosine_similarities = (dot_products / denom).reset_index()
    cosine_similarities.columns = ['file_idx', 'cosine_similarity']
    cosine_similarities.sort_values('cosine_similarity', ascending=False, inplace=True)
    best_idx = cosine_similarities.iloc[0].file_idx
    is_correct = int(df_dataset.loc[query_filename].file_idx==best_idx)
    df_final.loc[len(df_final)] = ['COS', query_idx[0], query_idx[1], tau, len(df_results), is_correct]
    return cosine_similarities


#import code; code.interact(local=dict(globals(), **locals()))
tau = args.tau
print("tau: ", tau)
#queries = df_queries.idx.unique()
#import code; code.interact(local=dict(globals(), **locals()))
df_queries = df_queries[df_queries['size'] <= 9]
df_queries.set_index(['idx', 'size'], inplace=True)
df_queries = df_queries.sort_index(level='idx')
#import code; code.interact(local=dict(globals(), **locals()))
query_indexes = df_queries.index.unique()
for query_idx in tqdm(query_indexes, total=len(query_indexes)):
    query_filename = df_filenames.loc[query_idx[0]].filename
    terms, dict_tfidf, terms_tfidf = calc_query_tfidf(query_idx)
    if len(terms) > 0:
        db_payload = list(index.search(terms))
    else:
        df_final.loc[len(df_final)] = ['WNG', query_idx[0], query_idx[1], tau, len(df_results), 0]
        df_final.loc[len(df_final)] = ['CNT', query_idx[0], query_idx[1], tau, len(df_results), 0]
        df_final.loc[len(df_final)] = ['COS', query_idx[0], query_idx[1], tau, len(df_results), 0]
        df_final.loc[len(df_final)] = ['WNG+COS', query_idx[0], query_idx[1], tau, len(df_results), 0]
        continue
    df_results = pd.DataFrame(db_payload, columns=['filename'])
    df_results.filename = df_results.filename.str.decode('utf-8')
    #cand = df_results.iloc[0].filename

    df1 = calc_score_cosine_sim(query_idx, query_filename, df_results)
    df2 = calc_score_wang(query_idx, query_filename, df_results, terms)
    calc_score_cnt(query_idx, query_filename, df_results, terms)
    df2['file_idx'] = df2.filename.apply(lambda x: df_dataset.loc[x].file_idx)
    #import code; code.interact(local=dict(globals(), **locals()))
    agg = ra.RankAggregator()
    best_borda = agg.borda([df1.file_idx.tolist(), df2.file_idx.tolist()])[0][0]
    is_correct_borda = int(df_dataset.loc[query_filename].file_idx==best_borda)
    df_final.loc[len(df_final)] = ['WNG+COS', query_idx[0], query_idx[1], tau, len(df_results), is_correct_borda]
    df_final.to_csv(os.path.join(args.results, 'results_comparison_%s.csv' % tau), index=False)


df_final.query_idx = df_final.query_idx.astype(int)
df_final.correct = df_final.correct.astype(int)
df_final.payload_size = df_final.payload_size.astype(int)
df_final.to_csv(os.path.join(args.results, 'results_comparison_%s.csv' % tau), index=False)
