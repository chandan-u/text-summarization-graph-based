"""
  data loading and preprocessing scripts


"""
from nltk.tokenize import sent_tokenize

def documents(path="./data/Summarization-master/data"):

    import os
    for f in os.listdir(path):
        if f.endswith(".txt"):
            yield (open(os.path.join(path, f)), f)


def load_clusters():

    clusters = {}
    for file_handler,filename in documents():
        cluster = []
        for line in file_handler:

            if line.strip() != "#" or " ":
                # append all the lines in the cluster of documents as 1
                cluster.extend(sent_tokenize(line.strip()))
            else:
                continue

        clusters[filename] = cluster
    return clusters




def nlp_preprocessing():

    clusters = load_clusters()

    from nltk.corpus import stopwords
    from nltk import ngrams
    from nltk.tokenize import word_tokenize

    import re
    import string

    regex = re.compile('[%s]' % re.escape(string.punctuation))
    regexb=re.compile('b[\'\"]')
    stop_words = set(stopwords.words('english'))

    for filename, sentences in clusters.iteritems():
        for index, sentence in enumerate(sentences):

            # preprocess each sentence in cluster

            #print sentences
            try:
                # lower strings
                tokens = map(str.lower, word_tokenize(sentence))

                #remove punctuation
                tokens =  map(lambda token:regex.sub('', token), tokens)

                #filter empty strings
                tokens = filter(lambda token: token != '', tokens )

                # remove stopwords (remove stopwords at algorithm level, insteald of corpus)
                # Stopwords can be removed at vectorization (bag_of_words)/word_embedding
                # the original sentenec is preserved for calculation of bleu scores
                # Since we will be comparing against human written summaries
                # tokens = filter(lambda word: word not in stop_words, tokens)

                # can do stemming
                # Unigrams are generated after removal of stopwords
                if " ".join(tokens) != "":
                    clusters[filename][index] = " ".join(tokens)
            except BaseException, e:
                print "error", filename, index, sentence, e

    return clusters
nlp_preprocessing()
