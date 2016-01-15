from scipy import dot, mat


def cos_similarity(vecx, vecy):
    """
    Using for cacluate cosin similarity of document vectors.

    document vector comes from tfidf. They, defaulty, are
    NORMAL. So, there size is 1.
    """
    # if vecx is list, no need transform to matrix
    # if type(vecx) == type([]):
    if isinstance(vecx, list):
        return float(dot(vecx, vecy))
    # if other type, we transform to matrix
    a = mat(vecx)
    b = mat(vecy)
    c = float(dot(a, b.T))
    return c
