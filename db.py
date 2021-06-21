import redis
import pandas as pd
import pickle
from collections import defaultdict
import numpy as np
import os


class InvertedIndex():

    def __init__(self, url, port, db):
        self.db = redis.StrictRedis(host=url, port=port, db=db)
        self.DF = defaultdict(lambda: 0)
        #self.dic_tf = defaultdict(lambda: (defaultdict(lambda: 0)))
        self.docs = defaultdict(lambda: 0)

    def clear(self):
        self.db.flushdb()

    def add(self, doc, terms):
        for term in [int(x) for x in set(terms)]:
            self.db.sadd(term, doc)
            self.DF[term] += 1
            self.docs[doc] = 1
            #self.add_tf(doc, term)

    def persist(self, folder):
        with open(os.path.join(folder, 'DF.pickle'), 'wb') as handle:
            pickle.dump(dict(self.DF), handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(folder, 'docs.pickle'), 'wb') as handle:
            pickle.dump(dict(self.docs), handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, folder):
        with open(os.path.join(folder, 'DF.pickle'), 'rb') as handle:
            self.DF = defaultdict(int, pickle.load(handle))
        with open(os.path.join(folder, 'docs.pickle'), 'rb') as handle:
            self.docs = pickle.load(handle)

        N = len(self.docs.keys())
        df = pd.DataFrame.from_dict(self.DF, orient='index')
        df.columns = ['DF']
        self.idfs = np.log(N / (df['DF'] + 1))

    def idf(self, term):
        return self.idfs[term]

    def search(self, terms_list):
        return self.db.sunion(terms_list)
