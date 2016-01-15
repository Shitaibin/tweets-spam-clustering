# Author: James Shi
# License: BSD 3 clause

from __future__ import print_function

import pandas as pd
from pytagcloud import create_tag_image, make_tags
from pytagcloud.lang.counter import get_tag_counts

from merge_cluster import merge_tweets


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

    return labels_text  # for test


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
