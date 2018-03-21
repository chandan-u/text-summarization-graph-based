import nltk
import os
from nltk.tokenize import word_tokenize, sent_tokenize
import re
import string
regex = re.compile('[%s]' % re.escape(string.punctuation))
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
def bleu(reference_filename, hypothesis):
    #the maximum is bigram, so assign the weight into 2 half.


    # load references
    references = []
    for f in os.listdir("./data/Summarization-master/data"):
        if re.match(reference_filename.split(".")[0] + ".m", f):
            references.append([ line for line in open(os.path.join("./data/Summarization-master/data",f)) ])

    #print len(refrences), len(references[0]), len(keysentences)

    # count references
    #min_count = min([len(reference) for reference in references])

    #select_references = [ [word_tokenize(reference[i]) for reference in references] for i in range(min_count) ]

    # unigrams for references:
    processed_references = []
    max_ref = 0
    for reference in references:
        processed_reference = []
        #if len(reference) > max_ref:


        for line in reference:
            #tokens = map(str.lower, word_tokenize(line))
            tokens = word_tokenize(line)
            tokens =  map(lambda token:regex.sub('', token), tokens)
            tokens = filter(lambda token: token != '', tokens )
            #tokens = filter(lambda word: word not in stop_words, tokens)
            processed_reference.extend(tokens)
        processed_references.append(processed_reference)

    processed_hypothesis = []
    for line in hypothesis[0:5]:
        #tokens = map(str.lower, word_tokenize(line))
        tokens = word_tokenize(line)
        tokens =  map(lambda token:regex.sub('', token), tokens)
        processed_hypothesis.extend(tokens)

    #print processed_hypothesis
    #print processed_references[0]



    BLEUscore = nltk.translate.bleu_score.sentence_bleu( processed_references, processed_hypothesis)
    return BLEUscore





# from pyrouge import Rouge155
#
# r = Rouge155()
# r.system_dir = './data/'
# r.model_dir = 'path/to/model_summaries'
# r.system_filename_pattern = 'some_name.(\d+).txt'
# r.model_filename_pattern = 'some_name.[A-Z].#ID#.txt'
#
# output = r.convert_and_evaluate()
# print(output)
# output_dict = r.output_to_dict(output)
