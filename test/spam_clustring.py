# Author: James Shi
# License: BSD 3 clause

import sys
# from time import time

sys.path.append("../tools/")

from tools import LoadData, Stem_TfidfTransform, CosSimilarity

from pandas import DataFrame

# Load data
fpath = "../data/test_1000.txt"
data = LoadData(fpath)
print 'data shape:', data.shape

# Get tweets content
tweets = data[:, 4]

# Vectorizer
tweets_vect = Stem_TfidfTransform(tweets)
print 'tweets vector:'
print 'n_samples:%d, n_features:%d' % tweets_vect.shape


# Caculate cosin similarity
# If have caculated and saved, reading from csv files.
def get_similarity(tweets_vect):
    simis = []
    n_samples = len(tweets_vect)
    for i in xrange(n_samples):
        for j in xrange(i+1, n_samples):
            simis.append([i, j, CosSimilarity(tweets_vect[i], tweets_vect[j])])
    return simis


# Explor correlation
simis = get_similarity(tweets_vect)
df = DataFrame(simis)