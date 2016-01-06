from __future__ import print_function

import sys
sys.path.append("../../tools/")

import tools
from tools import timestamp_to_datehour
from tools import load_data

from collections import Counter
from pprint import pprint
import pandas as pd
from pandas import DataFrame
from matplotlib import pyplot as plt


# file_path = "../../data/test_1000.txt"
file_path = "../../data/processed_20110607_spam_training.txt"
file_name = file_path.split('/')[-1][:-4]

fig_dir = "images/"
count_dir = "count/"
data_dir = "data/"


# count
def count_datehour_and_plot(data_datehour, file_name):
    # count
    count_result = Counter(data_datehour)
    # pprint(count_result)

    # better to check result
    count_df = DataFrame()
    count_df['datehour'] = count_result.keys()
    count_df['count'] = count_result.values()
    sorted_count_df = count_df.sort_index(by='datehour')
    # save
    sorted_count_df.to_csv(count_dir + file_name + ".csv",
                           index=False)

    # plot result
    fontsize = 8
    sorted_count_df.plot(x='datehour', y='count', kind='bar',
                         figsize=get_figsize(sorted_count_df.shape[0],
                                             fontsize),
                         fontsize=fontsize)
    fig_name = fig_dir + file_name
    # print(fig_name)
    plt.savefig(fig_name)

    # plt.show()


def get_figsize(n_datehour, fontsize):
    x_len = fontsize * n_datehour / 100.0 * 3
    y_len = int(x_len * 0.8)

    if x_len < 10:
        x_len = 10
    if y_len < 10:
        y_len = 8

    y_len = 8

    return (x_len, y_len)


# add datehour as a feature
def add_datehour_feature(dataset, datehour):
    dataset_df = DataFrame(dataset,
                           columns=['uid', 'timestamp',
                                    'degree', 'content',
                                    'url'])
    dataset_df['datehour'] = datehour
    return dataset_df


# save to csv file
def save_df_to_csv(data, file_name):
    data.to_csv(data_dir + file_name + ".csv", sep='|', index=False)


def load_new_data(file_name):
    dataset = pd.read_csv(data_dir + file_name + ".csv", sep='|')
    return dataset

if __name__ == "__main__":
    # load data and get datehour list.
    # dataset = load_data(file_path)
    # data_timestamp = dataset[:, 1]
    # data_datehour = [timestamp_to_datehour(e) for e in data_timestamp]

    # count_datehour_and_plot(data_datehour, file_name)

    # dataset_df = add_datehour_feature(dataset, data_datehour)

    # save_df_to_csv(dataset_df, file_name)

    dataset = load_new_data(file_name)
