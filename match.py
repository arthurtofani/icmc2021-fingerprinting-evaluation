import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse
from db import InvertedIndex
from collections import Counter
from audfprint import audfprint_analyze


parser = argparse.ArgumentParser("Builder")
parser.add_argument('--dataset', type=str, default='./datasets/fingerprints/gtzan.txt', help='dataset file')
parser.add_argument('--results', type=str, default='./exp/gtzan/', help='output directory')
parser.add_argument('--query_size', type=int, default=3, help='query size in seconds')
parser.add_argument('--db_idx', type=int, default=0, help='database id')
parser.add_argument('--db_port', type=int, default=6379, help='database port')
parser.add_argument('--db_url', type=str, default='localhost', help='database url')
args = parser.parse_args()

CONV = 256/11025  # to convert audfprint frames to time


df_queries = pd.read_csv(os.path.join(args.results, 'queries_hashes.csv'))
df_filenames = pd.read_csv(os.path.join(args.results, 'queries.csv'))
df_dataset = pd.read_csv(os.path.join(args.dataset), header=None).reset_index()
df_dataset.columns = ['file_idx', 'filename']
df_dataset.set_index('filename', inplace=True)
df_final = pd.DataFrame(columns=['query_size', 'query_idx', 'tau', 'payload_size', 'correct'])


#taus = list(np.arange(0.0, 7.0, 0.1)) + list(np.arange(7.0, 30.0, 0.5))
#taus += list(np.linspace(5.5, 5.8, 30))
#taus = sorted(np.unique(taus))

# taus = list(np.arange(0.0, 9.0, 0.5)) + list(np.arange(9.0, 30.0, 0.5))
taus = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 5.5, 6.5, 7.5, 8.5, 9.5, 4.5, 3.5, 11, 12, 13, 14, 15]
tf_df = pd.read_csv(os.path.join(args.results, 'tf.csv'))
tf_df.set_index('file_idx', inplace=True)
index = InvertedIndex(args.db_url, args.db_port, args.db_idx)
index.load(args.results)


def calc_query_tfidf(query_idx):
    conds = (df_queries.idx == query_idx) & (df_queries['size'] == query_size)
    terms_df = df_queries[conds].copy()
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


query_size = args.query_size

#taus = [t for t in taus if t > 4.5]
#taus += [0, 1, 2, 3, 4]
#import code; code.interact(local=dict(globals(), **locals()))
for tau in taus:
    print("tau: ", tau, "query_size", query_size)
    queries = df_queries.idx.unique()
    for query_idx in tqdm(queries, total=len(queries)):
        query_filename = df_filenames.loc[query_idx].filename
        terms, dict_tfidf, terms_tfidf = calc_query_tfidf(query_idx)
        if len(terms) == 0:
            df_final.loc[len(df_final)] = [query_size, query_idx, tau, 0, 0]
            continue
        db_payload = list(index.search(terms))
        df_results = pd.DataFrame(db_payload, columns=['filename'])
        df_results.filename = df_results.filename.str.decode('utf-8')
        #cand = df_results.iloc[0].filename
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
        df_final.loc[len(df_final)] = [query_size, query_idx, tau, len(df_results), is_correct]

    df_final.query_idx = df_final.query_idx.astype(int)
    df_final.correct = df_final.correct.astype(int)
    df_final.payload_size = df_final.payload_size.astype(int)
    df_final.to_csv(os.path.join(args.results, 'results.%s.csv' % query_size), index=False)

