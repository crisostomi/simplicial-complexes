import argparse
import os

from tsp_sc.common.io import load_config
from tsp_sc.citations.utils.citations import *


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
    random.shuffle(copy_seed)
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    cli_args = parser.parse_args()

    config = load_config(cli_args.config)

    paths = config["paths"]

    adjacency_papers = sparse.load_npz(paths["adj_papers_path"])

    adjacency = scipy.sparse.load_npz(paths["biadjacency_matrix_path"])

    papers_df_path = os.path.join(paths["raw"], "papers.csv")
    papers_df = pd.read_csv(papers_df_path, index_col=0)

    # shape (num_papers, )
    citations = np.array(papers_df["citations_2019"])

    # starting_node = 150250

    downsample = np.array([0])
    while downsample.shape[0] < 250:
        starting_node = starting_node_random_walk(
            adjacency, weights_x=citations, min_weight=100, max_dim=10
        )

        COMPLEX_FOLDER_START_NODE = os.path.join(
            paths["complex_folder"], str(starting_node)
        )
        print("The starting node of the random walk has ID {}".format(starting_node))

        downsample = subsample_node_x(
            start=starting_node,
            adjacency_graph_x=adjacency_papers,
            bipartite=adjacency,
            weights_x=citations,
            min_weight=5,
            max_dim=10,
            length_walk=200,
        )

        print(downsample.shape)
    os.makedirs(COMPLEX_FOLDER_START_NODE)
    np.save(os.path.join(COMPLEX_FOLDER_START_NODE, "downsampled.npy"), downsample)
