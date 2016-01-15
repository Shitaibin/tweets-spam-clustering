from __future__ import print_function

import pandas as pd
from pandas import DataFrame
from pytagcloud.lang.counter import get_tag_counts


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


# print("Cluster\tSub Cluster(s)")
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
