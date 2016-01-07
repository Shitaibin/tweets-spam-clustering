# Author: James Shi
# License: BSD 3 clause

from __future__ import print_function

import sys
sys.path.append("../tools")

from tools import merge_cluster_result, analyze_result

import os

pre_fix = "kmeans_res_8432n"

files = os.listdir("test_result")

for f in files:
    if f.startswith(pre_fix) and "merge" not in f:
        sta = len(pre_fix) + 1
        idx = f.index('k', sta)
        k = f[sta: idx]
        fp = "test_result/" + f
        nfp = fp[:-4] + "_merge.csv"

        print("k = ", k)
        print("Before merge")
        analyze_result(fp)

        if merge_cluster_result(fp):
            # show analyzing info
            analyze_result(fp)
            print("\nAfter merge")
            analyze_result(nfp)
            print("--------------------------------------")
        else:
            print("There is no need to merge.")
