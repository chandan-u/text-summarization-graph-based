from algorithms import LexRank

from datautils import nlp_preprocessing

import numpy as np
clusters = nlp_preprocessing()


# basic lexrank
for threshold in np.arange(0, 1, 0.1):

    bleu_scores = []
    for clustername, sentences in clusters.iteritems():
        lexrank = LexRank(clustername, sentences, threshold)
        lexrank.build_tf_idf_matrix()
        lexrank.build_similarity_matrix()
        lexrank.build_graph()
        lexrank.computepagerank()
        score = lexrank.eval()

        bleu_scores.append(score)

    print "threshold: ", threshold
    print "scores: ", bleu_scores
    print "avg_scoreL", sum(bleu_scores)/len(bleu_scores)

    print "max_score", max(bleu_scores)
    print "min_score", min(bleu_scores)
