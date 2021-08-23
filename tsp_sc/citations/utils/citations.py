import numpy as np


def process_papers(corpus):
    """First pass through all the papers."""

    papers = dict()
    edges = list()

    for file in tqdm(sorted(glob.glob(corpus))):
        with gzip.open(file, "rt") as file:

            for line in file:

                paper = json.loads(line)

                try:
                    year = paper["year"]
                except KeyError:
                    year = 0

                missing_authors = 0
                for author in paper["authors"]:
                    n_ids = len(author["ids"])
                    if n_ids == 0:
                        missing_authors += 1
                    elif n_ids == 1:
                        edges.append((paper["id"], author["ids"][0]))
                    else:
                        raise ValueError("No author should have multiple IDs.")

                # papers[paper_id_i] = [ [citation_1, .. , citation_n], num_out_citations, year_publication, num_missing_authors ]
                papers[paper["id"]] = (
                    paper["inCitations"],
                    len(paper["outCitations"]),
                    year,
                    missing_authors,
                )

    print(f"processed {len(papers):,} papers")
    print(f"collected {len(edges):,} paper-author links")

    return papers, edges


def count_citations(papers, years):
    """Second pass to check the publication year of the referencing papers."""

    years = np.array(years)

    # for each paper
    for pid, attributes in tqdm(papers.items()):

        missing_citations = 0
        counts = np.zeros_like(years)

        # for each in_citation
        for citation in attributes[0]:

            try:
                year = papers[citation][2]
            except KeyError:
                missing_citations += 1  # unknown paper
                continue

            if year != 0:
                counts += year < years
            else:
                missing_citations += 1  # unknown year

        papers[pid] = tuple(counts) + attributes[1:] + (missing_citations,)


def save(papers, edges, pickle_file):
    data = dict(edges=edges, papers=papers)
    with open(pickle_file, "wb+") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def count_papers_authors(papers, edges):
    print(f"  {len(papers):,} papers")
    print("  {:,} authors".format(edges["author"].nunique()))
    print(f"  {len(edges):,} paper-author links")


def load_papers_authors_bipartite_graph(path):
    """Load the full bipartite graph."""

    with open(path, "rb") as file:
        data = pickle.load(file)

    edges = pd.DataFrame(data["edges"], columns=["paper", "author"])
    edges["author"] = pd.to_numeric(edges["author"])

    cols_citations = [f"citations_{year}" for year in range(1994, 2024, 5)]

    cols = cols_citations + [
        "references",
        "year",
        "missing_authors",
        "missing_citations",
    ]

    papers = pd.DataFrame.from_dict(data["papers"], orient="index", columns=cols)
    papers.index.name = "paper"

    print("input data:")
    count(papers, edges)
    print("  {:,} missed links".format(papers["missing_authors"].sum()))
    print("  {:,} missed citations".format(papers["missing_citations"].sum()))

    return papers, edges


def drop_papers(paper_ids, papers_df, text):
    print(f"  drop {len(paper_ids):,} {text}")
    papers_df.drop(paper_ids, inplace=True)
    print(f"  {len(papers_df):,} papers remaining")


def drop_edges(edges_df, papers_df):
    keep = edges_df["paper"].isin(papers_df.index)
    print(f"  drop {len(edges_df) - keep.sum():,} edges from dropped papers")
    edges_df = edges_df[keep]
    print(f"  {len(edges_df):,} edges remaining")
    return edges_df


def clean(papers_df, edges_df):
    """Select a subset of the bipartite paper-author graph.

    Only drop papers. We want to keep all the authors from the selected papers.

    TODO: should we drop papers with missing in/out citations?
    There's probably citations S2 misses as well.
    """

    print("removing some papers:")

    # Papers that are not in the edge list, i.e., paper without identified authors.
    drop_papers(
        papers_df.index.difference(edges_df["paper"].unique()),
        papers_df,
        "papers without authors",
    )

    drop_papers(
        papers_df.index[papers_df["missing_authors"] != 0],
        papers_df,
        "papers with missing author IDs",
    )
    drop_papers(
        papers_df.index[papers_df["year"] == 0],
        papers_df,
        "papers without publication year",
    )
    drop_papers(
        papers_df.index[papers_df["references"] == 0],
        papers_df,
        "papers without references",
    )
    drop_papers(
        papers_df.index[papers_df["missing_citations"] != 0],
        papers_df,
        "papers with missing citations",
    )

    edges_df = drop_edges(edges_df, papers_df)

    # Papers written by too many authors (cap the simplex dimensionality).
    # Will also limit how fast the BFS grows.
    n_authors = 10
    size = edges_df.groupby("paper").count()
    drop_papers(
        size[(size > n_authors).values].index,
        papers_df,
        f"papers with more than {n_authors} authors",
    )

    edges_df = drop_edges(edges_df, papers_df)

    return papers_df, edges_df


def grow_network(seed, n_papers, edges_df):
    print(f"selecting at least {n_papers:,} papers around paper ID {seed}")
    new_papers = [seed]
    keep_papers = {seed}
    while len(keep_papers) < n_papers:
        print(f"  {len(keep_papers):,} papers currently selected")
        new_authors = edges_df["author"][edges_df["paper"].isin(new_papers)]
        new_papers = edges_df["paper"][edges_df["author"].isin(new_authors)]
        keep_papers = keep_papers.union(new_papers.values)
    print(f"  {len(keep_papers):,} papers selected")
    return keep_papers


def downsample(papers_df, edges_df):
    keep = grow_network(seed=papers_df.iloc[100].name, n_papers=5000, edges_df=edges_df)

    papers_df = papers_df.loc[papers_df.index.intersection(keep)]
    edges_df = edges_df[edges_df["paper"].isin(papers_df.index)]

    papers_df.sort_index(inplace=True)
    edges_df.sort_values("paper", inplace=True)
    edges_df.reset_index(drop=True, inplace=True)

    print("remaining data:")
    count(papers_df, edges_df)
    return papers_df, edges_df


def add_node_ids(papers, edges):
    """Generate authors table and node IDs."""

    # Create author table with node IDs.
    authors = pd.DataFrame(edges["author"].unique(), columns=["author"])
    authors.sort_values("author", inplace=True)
    authors.set_index("author", inplace=True, verify_integrity=True)
    authors["author_node_id"] = np.arange(len(authors))
    print(f"author table: {len(authors):,} authors")

    # Create paper node IDs.
    papers["paper_node_id"] = np.arange(len(papers))

    # Insert paper and author node IDs in the edge list.
    edges = edges.join(papers["paper_node_id"], on="paper", how="right")
    edges = edges.join(authors["author_node_id"], on="author", how="right")
    edges.sort_index(inplace=True)

    return papers, authors, edges


def save_paper_author_biadj(papers, authors, edges, biadj, folder):
    """Save the paper-author biadjacency sparse matrix as `*_adjacency.npz` and
    the paper feature matrix as `*_citations.npy`."""

    papers.drop("paper_node_id", axis=1, inplace=True)
    authors.drop("author_node_id", axis=1, inplace=True)
    edges.drop("paper_node_id", axis=1, inplace=True)
    edges.drop("author_node_id", axis=1, inplace=True)

    print("saving:")

    print("  paper table: {:,} papers, {:,} features".format(*papers.shape))
    papers.to_csv(os.path.join(folder, "papers.csv"))
    print("  edges table: {:,} edges".format(edges.shape[0]))
    edges.to_csv(os.path.join(folder, "paper_author_edges.csv"), index=False)

    print(
        "  biadjacency matrix: {:,} papers, {:,} authors, {:,} edges".format(
            *biadj.shape, biadj.nnz
        )
    )
    sparse.save_npz(os.path.join(folder, "paper_author_biadjacency.npz"), biadj)


def load_paper_author_biadj(path):
    biadjacency = sparse.load_npz(path)
    bipartite = nxb.from_biadjacency_matrix(biadjacency)

    print(f"{bipartite.number_of_edges():,} edges in the bipartite graph")
    print(f"connected: {nx.is_connected(bipartite)}")

    return bipartite


def project(bipartite):
    """Project the bipartite graph on both sides.

    Returns
    -------
    graph_papers : nx graph
        Graph where two papers are connected if they share an author.
    graph_authors : nx graph
        Graph where two authors are connected if they wrote a paper together.
    """

    nodes_papers = {n for n, d in bipartite.nodes(data=True) if d["bipartite"] == 0}
    nodes_authors = set(bipartite) - nodes_papers

    graph_papers = nxb.weighted_projected_graph(bipartite, nodes_papers)
    graph_authors = nxb.weighted_projected_graph(bipartite, nodes_authors)

    print(
        f"projection: {graph_papers.number_of_nodes():,} papers and {graph_papers.number_of_edges():,} edges"
    )
    print(
        f"projection: {graph_authors.number_of_nodes():,} authors and {graph_authors.number_of_edges():,} edges"
    )

    return graph_papers, graph_authors


def save_projected_graphs(
    graph_papers, graph_authors, adj_papers_path, adj_authors_path
):
    adjacency_papers = nx.to_scipy_sparse_matrix(graph_papers, dtype=np.int32)
    adjacency_authors = nx.to_scipy_sparse_matrix(graph_authors, dtype=np.int32)

    print(
        "adjacency matrix: {:,} papers, {:,} edges".format(
            adjacency_papers.shape[0], adjacency_papers.nnz // 2
        )
    )
    print(
        "adjacency matrix: {:,} authors, {:,} edges".format(
            adjacency_authors.shape[0], adjacency_authors.nnz // 2
        )
    )

    sparse.save_npz(adj_papers_path, adjacency_papers)
    sparse.save_npz(adj_authors_path, adjacency_authors)


def starting_node_random_walk(bipartite, weights_x, min_weight=100, max_dim=10):
    """
    Sample random node (paper) in X (from bipartite graph X-Y (papers, authors)) with the restriction that:
        - it does not connect to more than "max_dim" nodes in Y, i.e. it has less than max_dim authors
        - its weight is more than "min_weight", i.e. it has more than min_weight citations

    Parameters
    ----------
    bipartite : scipy sparse matrix
        bipartite collaboration graph X-Y

    weights_x : ndarray
        Array of size bipartite.shape[0], containing the weights on the node of X

    min_weight : float
        minimum weight of the sampled node

    max_dim : int
        maximum number of adjacent nodes in Y

    Returns
    -------
        start : starting node of the random walk
    """

    # convert to list of lists format
    adj_list = bipartite.tolil()

    rows = adj_list.rows

    indices_of_papers_with_at_least_min_weight = np.where(weights_x > min_weight)
    adj_list_papers_at_least_min_weight = rows[
        indices_of_papers_with_at_least_min_weight
    ]

    seeds_papers = []

    for index, authors in enumerate(adj_list_papers_at_least_min_weight):
        num_authors = len(authors)
        if num_authors < max_dim:
            # print(f'Paper {indices_of_papers_with_at_least_min_weight[0][index]} has {num_authors} authors and {weights_x[indices_of_papers_with_at_least_min_weight][index]} citations')
            seeds_papers.append(indices_of_papers_with_at_least_min_weight[0][index])

    copy_seed = np.copy(seeds_papers)
    shuffle(copy_seed)
    start = copy_seed[0]

    return int(start)


def subsample_node_x(
    start,
    adjacency_graph_x,
    bipartite,
    weights_x,
    min_weight=5,
    max_dim=10,
    length_walk=80,
):
    """"
        Downsample set of nodes (papers) X' of X (from bipartite graph X-Y (papers, authors)) such that each node connects to at most 10 nodes in Y
        (i.e. the paper has at most 10 authors) and its weights are at least 5 (i.e. the number of citation is at least 5).
        To ensure that the resulting bipartite graph X'-Y' is connected we downsampled X (with the above restrictions) by performing random walks on the X-graph.
        (eg performing random walks on the papers graph -restricted to papers that have at least 5 citations and at most 10 authors-
        where two papers are connected if they have at least one author in common)

        Parameters
        ----------
        adjacency_graph_x : scipy sparse matrix
            adjacency matrix of X (from the bipartite graph X-Y)

        bipartite : scipy sparse matrix
            bipartite collaboration graph X-Y

        weights_x : ndarray
            Array of size bipartite.shape[0], containing the weights on the node of X

        min_weight : float
            minimum weight of the sampled node, default 5

        max_dim : int
            maximum number of adjacent nodes in Y, default 10

        length_walk : int
            length of random walk with the above restrictions
        Returns
        -------
        p: array of the downsampled nodes in X = X'
    """

    adj_list = bipartite.tolil()
    rows = adj_list.rows

    G = nx.from_scipy_sparse_matrix(adjacency_graph_x)

    # first iteration
    neighborhood_nodes = get_neighborhood_plus_start(G, start)

    neighborhood_nodes_weights = weights_x[neighborhood_nodes]
    neighborhood_nodes_with_at_least_min_weight = neighborhood_nodes[
        np.where(neighborhood_nodes_weights >= min_weight)
    ]

    past_seeds = [start]

    for iterations in range(0, length_walk):

        new_seeds = []

        for index, authors in enumerate(rows[neighborhood_nodes]):
            num_authors = len(authors)
            if (
                num_authors < max_dim
                and weights_x[neighborhood_nodes[index]] >= min_weight
            ):
                new_seeds.append(neighborhood_nodes[index])

        unseen_seeds = list(set(new_seeds).difference(past_seeds))
        if len(unseen_seeds) <= 1:
            break

        start = unseen_seeds[np.argsort(weights_x[unseen_seeds])[-2]]

        neighborhood_nodes = get_neighborhood_plus_start(G, start)

        neighborhood_nodes_weights = weights_x[neighborhood_nodes]
        new_neighborhood_nodes_with_at_least_min_weight = neighborhood_nodes[
            np.where(neighborhood_nodes_weights >= min_weight)
        ]

        # update the nodes
        final = np.concatenate(
            (
                neighborhood_nodes_with_at_least_min_weight,
                new_neighborhood_nodes_with_at_least_min_weight,
            )
        )
        neighborhood_nodes_with_at_least_min_weight = np.unique(final)

        past_seeds.append(start)

    return neighborhood_nodes_with_at_least_min_weight


def get_neighborhood_plus_start(G, start):
    # get edges from start node to the neighbors
    neighbor_edges = list(
        nx.algorithms.traversal.breadth_first_search.bfs_edges(
            G, start, reverse=False, depth_limit=1
        )
    )

    B = nx.Graph()
    B.add_edges_from(neighbor_edges)

    neighborhood_nodes = np.array(B.nodes())
    return neighborhood_nodes


def bipart2simpcochain(
    bipartite, weights_x, indices_x=None, function=np.sum, dimension=3
):
    """From a collaboration bipartite graph X-Y and its weights on X to a
    collaboration simplicial complex and its collaboration values on the
    simplices.

    Parameters
    ----------
    bipartite : scipy sparse matrix
        Sparse matrix representing the collaboration bipartite graph X-Y.
    weights_x : ndarray
        Array of size bipartite.shape[0], containing the weights on the node of X
    function : callable
        Functions that will aggregate the features to build the k-cochains, default=np.sum
    indices_x : array
        Array of the indices of the X nodes to restrict to, default = all nodes of X
    dimension : int
        Maximal dimension of the simplicial complex.

    Returns
    -------
    simplices: list of dictionaries
        List of dictionaries, one per dimension k.
        The dictionary's keys are the k-simplices. The dictionary's values are their indices
    cochains:list of dictionaries
        List of dictionaries, one per dimension k.
        The dictionary's keys are the k-simplices. The dictionary's values are the k-cochains
    signals_top:
        Features for every maximal dimensional simplex
    """
    simplex_tree, signal_top = bipart2simplex(
        bipartite, weights_x, indices_x, dimension
    )
    simplices = extract_simplices(simplex_tree)
    cochains = build_cochains(simplex_tree, signal_top, function, dimension)
    return simplices, cochains, signal_top


def bipart2simplex(bipartite, weights_x, indices_x=None, dimension=3):
    """Build a Gudhi Simplex Tree from the bipartite graph X-Y by projection on Y and extract the
    features corresponding to maximal dimensional simplices.
    Parameters
    ----------
    bipartite : scipy sparse matrix
        bipartite collaboration graph X-Y
    weights_x : ndarray
        Array of size bipartite.shape[0], containing the weights on the node of X
    indices_x = array
        Array of the indices of the X nodes to restrict to, default: all nodes of X
    dimension: int
        maximal dimension of the simplicial complex = maximal number of individuals collaborating.
    Returns
    -------
    simplex_tree:
        Gudhi simplex tree.
    signals_top:
        Features for every maximal dimensional simplex.
    """
    signals_top = [dict() for _ in range(dimension + 1)]
    simplex_tree = gudhi.SimplexTree()

    if np.all(indices_x) == None:
        indices_x = np.arange(bipartite.shape[0])

    Al = bipartite.tolil()

    # for each paper, if the number of authors is <= the max accepted dimension then we insert it into the simplex complex
    for j, authors in enumerate(Al.rows[indices_x]):
        if len(authors) <= dimension + 1:
            k = len(authors)
            simplex_tree.insert(authors)
            # append the number of citations of the paper to the simplex
            # (which may already exist as it is indexed by the group of authors)
            signals_top[k - 1].setdefault(frozenset(authors), []).append(
                weights_x[indices_x][j]
            )
        else:
            continue

    return (simplex_tree, signals_top)


def extract_simplices(simplex_tree):
    """Create a list of simplices from a gudhi simplex tree."""
    simplices = [dict() for _ in range(simplex_tree.dimension() + 1)]
    for simplex, _ in simplex_tree.get_skeleton(simplex_tree.dimension()):
        k = len(simplex)
        simplices[k - 1][frozenset(simplex)] = len(simplices[k - 1])
    return simplices


def build_cochains(simplex_tree, signals_top, function=np.sum, dimension=3):
    """Build the k-cochains using the weights on X (from the X-Y bipartite graph)
     and a chosen aggregating function. Features are aggregated by the provided functions.
     The function takes as input a list of values and must return a single number.

    Parameters
    ----------
    simplex_tree :
        Gudhi simplex tree
    signals_top : ndarray
        Features for every maximal dimensional simplex = weights on the nodes of X (from bipartite graph X-Y)
    function : callable
        Functions that will aggregate the features to build the k-coachains, default=np.sum

    Returns
    -------
    signals : list of dictionaries
        List of dictionaries, one per dimension k.
        The dictionary's keys are the k-simplices. The dictionary's values are the k-cochains
    """
    assert len(signals_top) == dimension + 1
    cochains = [dict() for _ in range(simplex_tree.dimension() + 1)]

    for d in range(dimension, -1, -1):
        # simplex represents a group of authors, values are the number of citations they got in their papers
        # len(values) = # papers published together by the group of authors
        for simplex, values in signals_top[d].items():
            st = gudhi.SimplexTree()
            st.insert(simplex)
            for face, _ in st.get_skeleton(st.dimension()):
                # for each "sub-simplex" s contained in the simplex (e.g. if d=2 we have the simplex itself, the edges and the vertices)
                face = frozenset(face)
                # if a simplex has k+1 nodes then it is k-dimensional
                face_dim = len(face) - 1
                cochains[face_dim].setdefault(face, []).extend(signals_top[d][simplex])

    # each simplex now has a list containing the citations of all the simplices it is contained into

    for d in range(dimension, -1, -1):
        for simplex, values in signals_top[d].items():
            st = gudhi.SimplexTree()
            st.insert(simplex)
            for face, _ in st.get_skeleton(st.dimension()):
                face = frozenset(face)
                face_dim = len(face) - 1
                value = np.array(cochains[face_dim][face])
                cochains[face_dim][face] = int(function(value))

    # each simplex has a single value which is the aggregation of the signal on all the simplices it is contained into
    return cochains


def build_missing_values(simplices, percentage_missing_values, max_dim=10):
    """
    The functions randomly deletes a given percentage of the values of simplices in each dimension
    of a simplicial complex.

    Parameters
    ----------

    simplices: list of dictionaries
        List of dictionaries, one per dimension d. The size of the dictionary
        is the number of d-simplices. The dictionary's keys are sets (of size d
        + 1) of the vertices that constitute the d-simplices. The
        dictionary's values are the indexes of the simplices in the boundary
        and Laplacian matrices.

    percentage_missing_values: integer
        Percentage of values missing

    max_dim: integer
        maximal dimension of the simplices to be considered.

    Returns
    ----------
        missing_values: list of dictionaries
        List of dictionaries, one per dimension d. The dictionary's keys are the missing d-simplices.
        The dictionary's values are the indexes of the simplices in the boundary and Laplacian matrices.

    """
    missing_values = [dict() for _ in range(max_dim + 1)]

    for k in range(max_dim + 1):

        # randomly select percentage_missing_values % of the simplices of each dimension
        simplices_dim_k = list(simplices[k].keys())
        threshold_index = int(
            np.ceil((len(simplices_dim_k) / 100) * percentage_missing_values)
        )
        simplices_dim_k_copy = np.copy(simplices_dim_k)
        shuffle(simplices_dim_k_copy)
        lost_simplices = simplices_dim_k_copy[:threshold_index]

        # assign to each simplex in the missing simplices the same id it had before
        for simplex in lost_simplices:
            missing_values[k][simplex] = simplices[k][simplex]

    return missing_values


def build_damaged_dataset(cochains, missing_values, function=np.median):
    """
    The function replaces the missing values in the dataset with a value inferred
    from the known data (eg the missing values are replaced by the median
    or mean of the known values).

    Parameters
    ----------
    cochains: list of dictionaries
        List of dictionaries, one per dimension k.
        The dictionary's keys are the k-simplices. The dictionary's values are the k-cochains

    missing_values: list of dictionaries
        List of dictionaries, one per dimension k.
        The dictionary's keys are the missing k-simplices. The dictionary's values are their indices

    function: callable
        missing values are replaced by the function of the known values, defaut median

    Returns
    ----------
    damaged_dataset: list of dictionaries

        List of dictionaries, one per dimension d. The dictionary's keys are the d-simplices.
        The dictionary's values are the d-cochains where the damaged portion has been replaced
        by the given function value.

    """

    # obtain median value
    max_dim = len(missing_values)
    cochains_copy = np.copy(cochains)
    signal = []
    for k in range(max_dim):
        cochains_dim_k = list(cochains_copy[k].values())
        signal.append(cochains_dim_k)
    signal = np.array(signal)

    # medians[k] has the median number of citations for dimension k
    medians = []
    for k in range(max_dim):
        signals_dim_k = [signal[k][j] for j in range(len(signal[k]))]
        median_dim_k = function(signals_dim_k)
        medians.append(median_dim_k)

    # create input using median value for unknown values
    damaged_dataset = np.copy(cochains)

    for k in range(max_dim):
        lost_simplices = list(missing_values[k].keys())
        for simplex in lost_simplices:
            damaged_dataset[k][simplex] = medians[k]
    return damaged_dataset


def build_known_values(missing_values, simplices):
    """
    The functions return the not missing simplices and indices in each dimension

    Parameters
    ----------
    missing_values: list of dictionaries
        List of dictionaries, one per dimension d. The dictionary's keys are the missing d-simplices.
        The dictionary's values are the indexes of the simplices in the boundary
        and Laplacian matrices.

    simplices: list of dictionaries
        List of dictionaries, one per dimension d. The size of the dictionary
        is the number of d-simplices. The dictionary's keys are the sets (of size d
        + 1) of vertices that constitute the d-simplices. The
        dictionary's values are the indexes of the simplices in the boundary
        and Laplacian matrices.


    Returns
    ----------
    known_values: list of dictionaries
        List of dictionaries, one per dimension d. The dictionary's keys are not missing d-simplices.
        The dictionary's values are the indexes of the simplices in the boundary
        and Laplacian matrices.

    """
    max_dim = len(missing_values)
    known_values = [dict() for _ in range(max_dim + 1)]

    for k in range(max_dim):
        # take only the simplices which have not been removed
        known_simplices = list(set(simplices[k].keys()) - set(missing_values[k].keys()))

        for index, simplex in enumerate(known_simplices):
            known_values[k][simplex] = simplices[k][simplex]

    return known_values


def get_paths(path_params, data_params):
    starting_node, missing_value_ratio = (
        data_params["starting_node"],
        data_params["missing_value_ratio"],
    )
    paths = {k: v for k, v in path_params.items()}
    starting_node_const = "STARTINGNODE"
    missing_value_ratio_const = "MISSINGVALUERATIO"
    for path_name, path_value in path_params.items():
        if starting_node_const in path_value:
            paths[path_name] = path_value.replace(
                starting_node_const, starting_node
            ).replace(missing_value_ratio_const, str(missing_value_ratio))
    return paths
