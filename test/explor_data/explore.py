from __future__ import print_function

import sys
sys.path.append("../../tools/")

import tools
from tools import timestamp_to_datehour
from tools import LoadData

from collections import Counter
from pprint import pprint

file_path = "../../data/test_10000.txt"

# load data and get datehour list.
dataset = LoadData(file_path)
data_timestamp = dataset[:, 1]
data_datehour = [timestamp_to_datehour(e) for e in data_timestamp]

# count
count_result = Counter(data_datehour)
pprint(count_result)