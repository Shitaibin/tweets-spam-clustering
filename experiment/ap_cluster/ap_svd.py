from __future__ import print_function


from sklearn.cluster import AffinityPropagation

# LSA/SVD

# cluster
p = -50
ap = AffinityPropagation(preference=p)
ap.fit(X)
n_clusters = len(ap.cluster_centers_indices_)
print("number of clusters: %d" % n_clusters)
