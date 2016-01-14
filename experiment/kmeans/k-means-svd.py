# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Lars Buitinck <L.J.Buitinck@uva.nl>
# License: BSD 3 clause

from __future__ import print_function

# private tools
import sys
sys.path.append("..")

from tools.handledata import load_data

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.metrics import silhouette_score

from sklearn.cluster import KMeans, MiniBatchKMeans

import logging
from optparse import OptionParser
from time import time

import numpy as np
from pandas import DataFrame

import matplotlib.pyplot as plt

from collections import Counter

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


##########################################################
# Global variable / Configuration
test_res_dir = "test_result"


# parse commandline arguments
op = OptionParser()
op.add_option("--lsa",
              dest="n_components", type="int",
              help="Preprocess documents with latent semantic analysis.(LSA)")
op.add_option("--no-minibatch",
              action="store_false", dest="minibatch", default=True,
              help="Use ordinary k-means algorithm (in batch mode).")
op.add_option("--no-idf",
              action="store_false", dest="use_idf", default=True,
              help="Disable Inverse Document Frequency feature weighting.")
op.add_option("--use-hashing",
              action="store_true", default=False,
              help="Use a hashing feature vectorizer")
op.add_option("--n-features", type=int, default=10000,
              help="Maximum number of features (dimensions)"
                   " to extract from text.")
op.add_option("--verbose",
              action="store_true", dest="verbose", default=False,
              help="Print progress reports inside k-means algorithm.")

print(__doc__)


(opts, args) = op.parse_args()
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)

###############################################################################
# Here set params? Yes, I write it.
opts.minibatch = False
opts.n_components = 100  # For LSA, recommended value
# Using both minibatch and LSA, the result is not good. Using LSA is better
# than using minibatch.

###############################################################################
# Load data

fpath = "../../data/test_1000.txt"
dataset = load_data(fpath)
print('dataset shape: %d,%d' % dataset.shape)
# Get tweets content
data = dataset[:, 3]

# dataset = pd.read_csv("explor_data/data/test_8000.csv", sep="|")
# data = dataset['content']
# # remove hashtags
# data = remove_hashtag(data)

print("Extracting features from the training dataset"
      "using a sparse vectorizer")
t0 = time()
if opts.use_hashing:
    if opts.use_idf:
        # Perform an IDF normalization on the output of HashingVectorizer
        hasher = HashingVectorizer(n_features=opts.n_features,
                                   stop_words='english', non_negative=True,
                                   norm=None, binary=False)
        vectorizer = make_pipeline(hasher, TfidfTransformer())
    else:
        vectorizer = HashingVectorizer(n_features=opts.n_features,
                                       stop_words='english',
                                       non_negative=False, norm='l2',
                                       binary=False)
else:
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=opts.n_features,
                                 min_df=2, stop_words='english',
                                 use_idf=opts.use_idf)
X = vectorizer.fit_transform(data)  # more efficiently

print("done in %fs" % (time() - t0))
print("n_samples: %d, n_features: %d" % X.shape)
print()

n_samples = X.shape[0]
n_features = X.shape[1]

######################################################################
# Dimension Reduction, LSA(SVD)
if opts.n_components:
    print("Performing dimensionality reduction using LSA")
    t0 = time()
    # Vectorizer results are normalized, which makes KMeans behave as
    # spherical k-means for better results. Since LSA/SVD results are
    # not normalizd, we have to redo the normalization.
    if opts.n_components >= n_features:
        opts.n_components = int(n_features * 3 / 5)
    svd = TruncatedSVD(opts.n_components)
    normalizer = Normalizer(copy=False)  # in pipeline, must be no copy.
    lsa = make_pipeline(svd, normalizer)

    X = lsa.fit_transform(X)

    print("done in %fs" % (time() - t0))

    explained_variance = svd.explained_variance_ratio_.sum()
    print("Explained variance of the SVD step: {}%".format(
        int(explained_variance * 100)))
    print("n_samples: %d, n_features: %d" % X.shape)

    print()


###############################################################################
# Do the actual clustering
true_k = 2  # actually, I do not know how many clusters

if opts.minibatch:
    km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                         init_size=1000, batch_size=1000, verbose=opts.verbose)
else:
    km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1,
                verbose=opts.verbose)

print(km)
print()


###############################################################################
# How to find K. Using the method from Ch8.5 in Introduction to Data Mining.
# The result figure is NOT like the text book. There is NO
# obvious turning point.
# But, There is a turing point, when K is 3.
def should_iter(km=km):
    print("Should run k means many times?")
    n_iter = 100
    n_samples = X.shape[0]

    sses = []
    for ii in xrange(n_iter):
        km.n_clusters = 4
        # print("Clustering sparse data with %s" % km)
        km.fit(X)
        sses.append(km.inertia_)

    # show sses
    sses.sort()
    sses = map(lambda x: int(x) / 10 * 10, sses)  # 10 is the section size
    sse_count = Counter(sses)
    sse_sorted = sorted(sse_count.keys())
    cnt = map(sse_count.get, sse_sorted)
    print(sse_sorted)
    print(cnt)
    plt.bar(sse_sorted, cnt, width=10 - 1)
    plt.xlabel("SSE")
    plt.ylabel("Count")
    plt.title(("n_iter = %d, n_samples = %d" % (n_iter, n_samples)))
    plt.suptitle(("Should we run k means many times?"), fontweight='bold')
    plt.show()
    return sses  # for farther use


# get sihouette with specific SSE
def get_silhouette(km, sses):
    """
    Run k-means, until get sse in range.
    Use the result to caculate SC.

    return SC, list of clustring result
    """
    sse = np.median(sses)
    width = (max(sses) - min(sses)) / 10
    upp = sse + width
    low = sse - width

    cur_sse = low - 10

#    print('cur', cur_sse)
#    print('upp', upp)
#    print('low', low)

    cluster_labels = []
    # using closed interval in case of width=0
    while not (low <= cur_sse <= upp):
        km.fit(X)
        cluster_labels = km.labels_
        cur_sse = km.inertia_
#        print ('cur', cur_sse)

    # get sihouette
    silhouette_avg = silhouette_score(X, cluster_labels)
    return silhouette_avg, cluster_labels


# get SSEs and Silhouettes with diffirent K
def validate(km=km, n_iter=100, range_k=[2, 5]):
    """
    Get SSE and SC for each k.

    iter kmeans n_iter times to get a better sse
    for each k.

    rerun kmeans to get SC.

    return three lists of k, sse, SC
    """

    print("Estimate this model")
    time0 = time()
    k_sse = []
    silhouettes = []
    # range_k = range(2,30) ### set k's range
    for k in range_k:
        print('K', k)
        km.n_clusters = k

        # iter kmeans n_iter times
        sses = []
        for ii in xrange(n_iter):
            # print("Clustering sparse data with %s" % km)
            km.fit(X)
            sses.append(km.inertia_)
        sse = np.median(sses)  # set sse for this K
        k_sse.append(sse)

        # get SC
        sil_avg, cluster_labels = get_silhouette(km, sses)
        silhouettes.append(sil_avg)

        # save clustring result
        save_clustering_result(km, data)  # data is tweet content

    print("validation done in %fs" % (time() - time0))
    print("repeated %d times" % n_iter)
    print()
    print()
    return range_k, k_sse, silhouettes


# Plot SSE and Silhouette
def plot_sse_and_silhouette(ks, sses, silhouettes):
    import matplotlib.pyplot as plt
    sses = map(int, sses)

    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is SSE plot
    # ax1.set_xlim([])
    # ax1.set_ylim([])
    ax1.plot(ks, sses)
    ax1.scatter(ks, sses)
    ax1.set_xlabel("k")
    ax1.set_ylabel("SSE")
    ax1.set_ylim(0, max(sses))

    ax1.set_title("The SSE plot for the various clusters.")

    # The 2nd subplot is Silhouette plot
    # print(silhouettes)
    ax2.plot(ks, silhouettes)
    ax2.scatter(ks, silhouettes)
    ax2.set_xlabel("K")
    ax2.set_ylabel("Silhouettes")
    ax2.set_ylim(0, 1)
    ax2.set_title("The silhouette plot for the various clusters.")

    plt.suptitle("KMeans: %d samples, repeat %d times" % (n_samples, n_iter))
    plt.show()


def save_test_result(n, n_iter, ks, sses, scs):
    """
    Save test result into a csv file.

    n, iter_times, ks, sees, silhouttes is included.
    file name format: kmeans_test_res_{d}n_{d}times_{}.csv

    return None.

    #########################################
    bug: when I re run this program, the old data
    will be covered.
    fix: append timestamp to the tail of file name.
    """
    file_name = 'kmeans_test_res_{}n_{}times_{}.csv'.format(
                n, n_iter, int(time()))
    file_path = test_res_dir + '/' + file_name

    data = np.asarray([ks, sses, scs]).T  # bug: k will be change to float
    df = DataFrame(data, columns=['ks', 'sses', 'scs'])

    df.to_csv(file_path, index=False)


def save_clustering_result(km, tweets):
    """
    Save clustering result in to a csv file.

    file name format: kmeans_res_{}n_{}k_{}.csv
    return None.
    """
    labels = km.labels_
    k = km.n_clusters
    n = len(labels)
    file_name = "kmeans_res_{}n_{}k_{}.csv".format(n, k, int(time()))
    file_path = test_res_dir + '/' + file_name

    result_df = DataFrame()
    result_df['label'] = labels
    result_df['tweet'] = tweets

    # sort by label
    result_df.sort_index(by='label', inplace=True)
    # save to csv file
    result_df.to_csv(file_path, index=False)


if __name__ == '__main__':
    """
    Test
    """
    # should_iter() # YES
    n_iter = 10  # make sure more than 20
    range_k = range(2, 10)
    # ks, sses, silhouettes = validate(
    #     n_iter=n_iter, range_k=range_k)  # fast and enough
    # # k_sse = validate() # slow but more accurate
    # plot_sse_and_silhouette(ks, sses, silhouettes)
    #
    # save_test_result(n_samples, n_iter, ks, sses, silhouettes)
