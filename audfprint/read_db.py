import pandas as pd
from hash_table import HashTable
import numpy as np
from collections import defaultdict
from itertools import combinations
from scipy.sparse import lil_matrix
import os

res = '../results.csv'
try:
    os.remove(res)
except:
    pass

def file_info(names, x):
    try:
        r = names[x].split('/')[-1]
        f, ext = r.split('.')
        u, dt, hr, q, _ = f.split('_')
        return u, hr, q[1:-1], ext
    except IndexError:
        return None


def flush_dic(ht, dic):
    df = pd.DataFrame.from_dict(dic, orient='index', columns=['ct']).reset_index(drop=False)
    df = df[df.ct > 1]
    df['f1'] = df['index'].apply(lambda x: ht.names[x[0]])
    df['f2'] = df['index'].apply(lambda x: ht.names[x[1]])
    df = df[['f1', 'f2', 'ct']]
    try:
        df2 = pd.read_csv(res)
        #import code; code.interact(local=dict(globals(), **locals()))
        df2.append(df).to_csv(res, index=False)
    except:
        df.to_csv(res, index=False)
    #import code; code.interact(local=dict(globals(), **locals()))
    del(dic)
    return defaultdict(lambda: 0)

ht = HashTable('../db/db.pklz')
names = ht.names
nhashes = ht.table.shape[0]

dic = defaultdict(lambda: 0)
maxtimemask = (1 << ht.maxtimebits) - 1
hashmask = (1 << ht.hashbits) - 1

M = lil_matrix((len(names), len(names)))
for ix in range(nhashes):
    if (ix % 50000) == 0:
        print("flushing")
        dic = flush_dic(ht, dic)

    if (ix % 1000) == 0:
        rn = round((ix/nhashes)*100, 2)
        print(rn, len(dic.keys()))
        #if rn > 1.2:
        #    break
    tabvals = ht.table[ix, :]
    #import code; code.interact(local=dict(globals(), **locals()))
    file_ids = np.unique((tabvals >> ht.maxtimebits) - 1)
    try:
        file_ids = file_ids[ht.counts[file_ids] > 60]
    except IndexError:
        continue
    times = (tabvals & maxtimemask)
    if len(file_ids) > 1:
        for x in combinations(file_ids, 2):

            a, b = sorted(x)
            n1 = file_info(names, a)
            n2 = file_info(names, b)
            any_nil = ((n1 is None) or (n2 is None))
            if not any_nil:
                same_user = (n1[0] == n2[0])
                tdist = abs(int(n1[1])-int(n2[1]))
                same_q = n1[3] == n2[3]
                if (not same_user) and same_q and (tdist < 2000):  # < 1h
                    dic[a, b] += 1

#import code; code.interact(local=dict(globals(), **locals()))
## print("To file...")
## k = len(dic.keys())
## z = np.zeros([k, 3])
## for i, key in enumerate(dic.keys()):
##     z[i][0:2] = key
##     z[i][2] = dic[key]
##
## print("saving")
## z.dump('../Z.pkl')

#xx = np.load('../Z.pkl', allow_pickle=True)
