# Author: James Shi
# License: BSD 3 clause

"""
1. Load data
2. Tfidf transform
3. Cosin similarity

"""

# TODO: decompose the left compose

from __future__ import print_function

import logging
import time

import pandas as pd
from pandas import DataFrame
from pytagcloud import create_tag_image, make_tags
from pytagcloud.lang.counter import get_tag_counts

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('tweetspamtools')


#################################################
# Show 2d matrix by dot and cross mark
def plot_2d_matrix(mat):
    """
    Plot 2d matrix by dot and cross mark.

    The max area can be plot is 30 * 100. If the
    shape of matrix mat is more than 30 * 100,
    It will only plot the left-top of mat.

    dot(.) represent zero values.
    corss mark(+) represent non-zero value.

    return: None
    """
    ary = mat.toarray()
    for row in range(min(30, ary.shape[0])):
        for col in range(min(100, ary.shape[1])):
            if ary[row][col]:
                print('+', sep='', end='')
            else:
                print('.', sep='', end='')
        print()


#################################################
# Visualization
def merge_tweets(labels_tweets):
    """
    Merge tweets for each cluster.
    labels_tweets: DataFrame
    return: a dict of lables and their text
    """
    pass
    labels = set(labels_tweets['label'])
    labels_text = dict()
    for label in labels:
        qry = "label == {}".format(label)
        res = labels_tweets.query(qry)
        tweets = list(res['tweet'])
        text = " ".join(tweets)
        labels_text[label] = text
    return labels_text


def visualize_clusters(file_path, output_dir="."):
    """
    Visualize clusters using tag cloud.
    file_path: input data from csv file.
    output_dir: where to save tagcloud figure.
    """
    input_fname = file_path.split("/")[-1][:-4]

    labels_tweets = pd.read_csv(file_path)
    labels_text = merge_tweets(labels_tweets)
    make_tag_cloud(labels_text, input_fname, output_dir)

    return labels_text    # for test


def make_tag_cloud(labels_text, input_fname, output_dir):
    """
    Make tag cloud for each label/cluster from their text.

    labels_text: dict
    input_fname: string.
    output_dir: string.
    return: None
    """
    print("Let's make tag cloud")
    for label, text in labels_text.iteritems():
        fig_name = input_fname + "_label{}.png".format(label)
        fig_path = output_dir + "/" + fig_name
        tags = make_tags(get_tag_counts(text), maxsize=80)
        create_tag_image(tags, fig_path, size=(900, 600))
        print("label {} finished".format(label))


#######################################################
# Merge cluster

def sort_tag_counts(tag_counts):
    """
    Sort tag counts.

    sort rules: sort by tag first then frequence.
    tag_counts: list of tuples. example:[('python', 2), ('abc', 1)]
    return: sorted tag_counts.
    """
    tag_counts.sort(key=lambda x: x[0])
    tag_counts.sort(key=lambda x: x[1], reverse=True)
    return tag_counts


def get_m_tags(text, m):
    """
    Get tags from this text. tags are the most
    frequent words.
    text: not None or empty string.
    m: the no. of tags, m > 0
    return: a list of tags and their frequence.
    """
    if m <= 0:
        raise AssertionError("m should be bigger than 0.")

    assert text  # text is none or empty

    text = text.strip()
    assert text  # no word in text

    tag_counts = get_tag_counts(text)
    sort_tag_counts(tag_counts)
    if m <= len(tag_counts):
        return tag_counts[:m]
    else:
        return tag_counts


def get_cluster_tag(text):
    """
    Get a tag to represent a cluster.

    the tag is the most frequent word in text.

    text: not None or empty string.
    return: string.
    """
    assert text  # text is None or empty

    text = text.strip()
    assert text  # no word in text

    tag_list = get_m_tags(text, 1)
    if tag_list:
        return tag_list[0][0]


def get_merge_rules(labels_tags):
    """
    Merge rule: if two cluster have the same tag,
    merge the cluster with greater label id to
    the cluster with less label id.

    :labels_tag: list of tuple. tuple example: (label, tag).
    :return: dict.  key is each cluster label, value is
    which cluster it will be merge to.
    """
    tags = list(set([item[1] for item in labels_tags]))
    tags.sort()

    # get labels for each tag
    tag_labels = {}
    for tag in tags:
        tag_labels[tag] = sorted([item[0] for item in labels_tags
                                  if item[1] == tag])

    # make a rule for each tag
    # v1 sub_2_cluster_rules
#    rules = {}
#    for labels in tag_labels.itervalues():
#        min_label = labels[0]
#        rules[min_label] = labels[1:]
#
#    sub_2_cluster = {}
#    for cluster, subs in rules.iteritems():
# sub_2_cluster[cluster] = cluster  # merge to itself
#        if subs:
#            for sub in subs:
#                sub_2_cluster[sub] = cluster
#
    # v2 sub_2_cluster_rules
    sub_2_cluster = {}
    for labels in tag_labels.itervalues():
        cluster = labels[0]
        sub_2_cluster[cluster] = cluster
        for sub in labels[1:]:
            sub_2_cluster[sub] = cluster
    return sub_2_cluster


def show_merge_rules(rules):
    """
    Show merge rules of clustering result.

    rules: dict. key is cluster, values is a list of sub-clusters.
    """
    # print(rules)
    print("merge rules:")
    for sub, clu in rules.iteritems():
        print("{} -> {}".format(sub, clu))
#    print("Cluster\tSub Cluster(s)")
#    for c,s in rules.iteritems():
#        print(c, end="\t")
#        for sub in s:
#            print(sub, end=" ")
#        print()
#


def show_tags_of_each_cluster(labels_text):
    """
    Show tags of each cluster.

    file_path: input data from csv file.
    return: DataFrame.columns = [label, tags]
    """
    m = 3  # the no. of tags representing a cluster
    print("tags of each cluster:")
    for label, text in labels_text.iteritems():
        print("label {}: {}".format(label, get_m_tags(text, m)))


def analyze_result(file_path, save=False, out_dir="."):
    """
    Analyzing the clustering result.
    1. show tags of each cluster.
    2. merge rules of result.

    file_path: clustering result csv file.
    save: default True, save the analyzing result.
    out_dir: where to save the analyzing result.
    """
    # input_fname = file_path.split("/")[-1][:-4]

    labels_tweets = pd.read_csv(file_path)
    labels_text = merge_tweets(labels_tweets)

    # show tags
    show_tags_of_each_cluster(labels_text)

    labels_tags = []
    for label, text in labels_text.iteritems():
        labels_tags.append((label, get_cluster_tag(text)))

    # show rules
    rules = get_merge_rules(labels_tags)
    print()
    show_merge_rules(rules)


def merge_cluster_result(file_path):
    """
    Merge clustering result based on merge rules.
    """
    # input_fname = file_path.split("/")[-1][:-4]

    labels_tweets = pd.read_csv(file_path)
    labels_text = merge_tweets(labels_tweets)

    # get  a tag for each cluster
    labels_tags = []
    for label, text in labels_text.iteritems():
        labels_tags.append((label, get_cluster_tag(text)))

    # get rules
    rules = get_merge_rules(labels_tags)

    if len(set(rules.keys())) is len(set(rules.values())):
        return False  # no need to merge

    # merge
    labels = list(labels_tweets['label'])
    new_labels = []
    for l in labels:
        new_labels.append(rules[l])

    # merge new labels and tweets
    merged_cluster_df = DataFrame()
    merged_cluster_df['label'] = new_labels
    merged_cluster_df['tweet'] = labels_tweets['tweet']

    # save merged result
    fp = file_path[:-4] + "_merge.csv"
    merged_cluster_df.to_csv(fp, index=False)

    return True  # merged


def timestamp_to_datehour(timestamp):
    """
    Transform timestamp to date and hour.

    :timestamp: string
    :return: string
    """
    t = time.localtime(float(timestamp))
    datehour = time.strftime("%Y%m%d%H", t)
    return datehour
