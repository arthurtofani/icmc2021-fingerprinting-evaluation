import time
import os
import sys
import logging
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
parser.add_argument('--db_idx', type=int, default=0, help='database id')
parser.add_argument('--db_port', type=int, default=6379, help='database port')
parser.add_argument('--db_url', type=str, default='localhost', help='database url')
args = parser.parse_args()

CONV = 256/11025  # to convert audfprint frames to time

df_tf = pd.DataFrame(columns=['file_idx', 'term', 'tf'])


def calc_tf_values(file_idx, hashes, df_tf):
    tf = pd.DataFrame(hashes, columns=['tf', 'term']).groupby(['term']).count().reset_index(drop=False)
    tf['file_idx'] = file_idx
    return tf


index = InvertedIndex(args.db_url, args.db_port, args.db_idx)
index.clear()
dataset = pd.read_csv(args.dataset, header=None)[0].values
for file_idx, file in tqdm(enumerate(dataset), total=len(dataset)):
    hashes = audfprint_analyze.hashes_load(file)
    tf_values = calc_tf_values(file_idx, hashes, df_tf)
    df_tf = df_tf.append(tf_values)
    index.add(file, [b for a, b in hashes])

print("Persisting db info...")
df_tf.to_csv(os.path.join(args.results, 'tf.csv'), index=False)
index.persist(args.results)
print("Done!")


