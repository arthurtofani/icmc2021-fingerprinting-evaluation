from collections import Counter
from collections import defaultdict
from scipy.spatial.distance import cosine
import numpy as np


class CosineSimilarity:

    def __init__(self, query_terms, max_term, dic_idf):
        self.query_terms = query_terms
        self.max_term = max_term
        self.dic_idf = dic_idf
        self.dic_tf_query = dict(Counter(self.query_terms))  # tf for terms in query
        #self.N = N

    def dist(self, target_terms):
        dic_tf_target = dict(Counter(target_terms))      # tf for terms in target
        #norm_a_sum = norm_b_sum = 0
        #common_terms = set(self.query_terms).intersection(set(target_terms))
        #dot_product = sum([self.dic_tf_query[x] * dic_tf_target[x] * (self.dic_idf[x] ** 2) for x in common_terms])

        # TODO: redo below
        norm_query = np.zeros(self.max_term)
        norm_target = np.zeros(self.max_term)
        for i in self.query_terms:
            norm_query[i-1] = self.dic_tf_query[i] * self.dic_idf[i]
        for i in target_terms:
            norm_target[i-1] = dic_tf_target[i] * self.dic_idf[i]
        return cosine(norm_query, norm_target)
        #mag = np.sqrt((norm_query ** 2).sum()) * np.sqrt((norm_target ** 2).sum())
        #return dot_product / mag

