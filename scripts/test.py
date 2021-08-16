def test():
    r"""Test the transformation of a bipartite graph to a collaboration complex."""
    biadjacency = coo_matrix([[1, 1, 1, 0], [1, 1, 0, 0], [1, 0, 0, 1], [0, 0, 1, 1],])
    number_citations = np.array([100, 50, 10, 4])
    indices = np.arange(biadjacency.shape[0])
    simplices, cochains, signals_top = bipart2simpcochain(
        biadjacency, number_citations, function=np.sum
    )

    cochains_true = [
        {
            frozenset({0}): 100 + 50 + 10,
            frozenset({1}): 100 + 50,
            frozenset({2}): 100 + 4,
            frozenset({3}): 10 + 4,
        },
        {
            frozenset({0, 1}): 100 + 50,
            frozenset({0, 2}): 100,
            frozenset({1, 2}): 100,
            frozenset({0, 3}): 10,
            frozenset({2, 3}): 4,
        },
        {frozenset({0, 1, 2}): 100},
    ]
    simplices_true = [
        {frozenset({0}): 0, frozenset({1}): 1, frozenset({2}): 2, frozenset({3}): 3},
        {
            frozenset({0, 1}): 0,
            frozenset({0, 2}): 1,
            frozenset({0, 3}): 2,
            frozenset({1, 2}): 3,
            frozenset({2, 3}): 4,
        },
        {frozenset({0, 1, 2}): 0},
    ]
    assert cochains == cochains_true
    assert simplices == simplices_true


test()
