"""
  degree centraility
  lexrank
  lexrank continous
"""


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.sparse.csr import csr_matrix #need this if you want to save tfidf_matrix



from nltk.tokenize import word_tokenize
import itertools
import networkx as nx

from scipy.spatial.distance import cosine
import scipy.sparse

from eval import bleu

class LexRank():
    """
      version : BASIC
      init: pass the list of sentences as input
      (Same for single or multidocument)
      No preprocessing done here
    """

    def __init__(self, reference_filename, sentences, threshold=0.2):
        self.sentences = sentences
        self.reference_filename = reference_filename
        self.threshold = threshold

    def cosine_distance(self, v1, v2):

        dist =  cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))
        #print dist[0][0]
        return dist[0][0]

    def build_tf_idf_matrix(self):

        tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), stop_words = 'english')
        #print self.sentences
        #print type(self.sentences)
        #import sys;sys.exit(1)
        self.tfidf_matrix =  tf.fit_transform(self.sentences).toarray()


    def build_similarity_matrix(self):

        self.similarity_matrix = []
        for sent_vec1 in self.tfidf_matrix:
            row = []
            for sent_vec2 in self.tfidf_matrix:
                sim = self.cosine_distance(sent_vec1, sent_vec2)
                row.append(sim)
            self.similarity_matrix.append(row)
        #print len(self.similarity_matrix), len(self.similarity_matrix[0])
        #import sys; sys.exit()


    def build_graph(self):
        self.gr = nx.Graph() #initialize an undirected graph
        self.gr.add_nodes_from(self.sentences)

        for row_index, row in enumerate(self.similarity_matrix):
            for col_index, value in enumerate(row):
                if value > self.threshold:
                    self.gr.add_edge(self.sentences[row_index], self.sentences[col_index] )

    def computepagerank(self):
        page_ranks = nx.pagerank(self.gr)
        self.keysentences = []
        #self.keysentences = sorted(page_ranks, key=page_ranks.get, reverse=False)
        for key, value in sorted(page_ranks.iteritems(), key=lambda (k,v): (v,k)):
            self.keysentences.append(key)

    def eval(self):
        references = []
        score = bleu(self.reference_filename, self.keysentences)
        return score
