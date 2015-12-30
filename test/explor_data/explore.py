from __future__ import print_function

import sys
sys.path.append("../../tools/")

import tools
from tools import timestamp_to_datehour
from tools import LoadData

from collections import Counter
from pprint import pprint
# import pandas
from pandas import DataFrame
from matplotlib import pyplot as plt

file_path = "../../data/processed_20110607_spam_testing.txt"
fig_dir = "images/"
count_dir = "count/"

# load data and get datehour list.
dataset = LoadData(file_path)
data_timestamp = dataset[:, 1]
data_datehour = [timestamp_to_datehour(e) for e in data_timestamp]

# count
count_result = Counter(data_datehour)
# pprint(count_result)

# better to check result
count_df = DataFrame()
count_df['datehour'] = count_result.keys()
count_df['count'] = count_result.values()
sorted_count_df = count_df.sort_index(by='datehour')
# save
sorted_count_df.to_csv(count_dir + file_path.split('/')[-1][:-4] + ".csv",
                     index=False)


def get_figsize(n_datehour, fontsize):
    x_len = fontsize * n_datehour / 100.0 * 3
    y_len = int(x_len * 0.8)

    if x_len < 10:
        x_len = 10
    if y_len < 10:
        y_len = 8

    y_len = 8

    return (x_len, y_len)

# plot result
n_datehour = sorted_count_df.shape[0]
fontsize = 8
sorted_count_df.plot(x='datehour', y='count', kind='bar',
                     figsize=get_figsize(sorted_count_df.shape[0], fontsize),
                     fontsize=fontsize)
figname = fig_dir + file_path.split('/')[-1][:-4]
# print(figname)
plt.savefig(figname)

# plt.show()
