import time
import os
import sys
import logging
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse
from db import InvertedIndex
import mmh3
from audfprint import audfprint_analyze


logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler()])
                    # logging.FileHandler("debug.log"),

parser = argparse.ArgumentParser("Builder")
parser.add_argument('--dataset', type=str, default='./datasets/fingerprints/gtzan.txt', help='dataset file')
parser.add_argument('--results', type=str, default='./exp/gtzan/', help='output directory')
args = parser.parse_args()

CONV = 256/11025


queries_df = pd.read_csv(args.dataset, header=None).sample(40).reset_index(drop=True)
queries_df.columns = ['filename']
queries_df.to_csv(os.path.join(args.results, 'queries.csv'), index=False)
query_hashes_df = pd.DataFrame(columns=['idx', 'size', 'frame', 'hash', 'timestamp'])
for query_size in [3, 6, 9, 12, 15]:
    for i, query in tqdm(queries_df.iterrows(), total=len(queries_df)):
        hashes = audfprint_analyze.hashes_load(query.filename)
        hashes_df = pd.DataFrame(hashes, columns=['frame', 'hash'])
        hashes_df['timestamp'] = hashes_df.frame * CONV
        query_timecut = hashes_df[hashes_df.timestamp < (hashes_df.timestamp.max() - query_size)].sample().timestamp.values[0]
        query_df = hashes_df[(hashes_df.timestamp >= query_timecut) & (hashes_df.timestamp < (query_timecut + query_size))].copy()
        query_df['idx'] = i
        query_df['size'] = query_size
        query_hashes_df = query_hashes_df.append(query_df)

query_hashes_df.to_csv(os.path.join(args.results, 'queries_hashes.csv'), index=False)
print("ok")
#import code; code.interact(local=dict(globals(), **locals()))
