import pandas as pd
import argparse
import os
import scipy.sparse.linalg
from scipy import sparse

from tsp_sc.common.io import load_config
from tsp_sc.citations.utils.citations import *


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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    cli_args = parser.parse_args()

    config = load_config(cli_args.config)
    paths = config["paths"]

    bipartite = load_paper_author_biadj(paths["biadjacency_matrix_path"])

    graph_papers, graph_authors = project(bipartite)

    save_projected_graphs(
        graph_papers, graph_authors, paths["adj_papers_path"], paths["adj_authors_path"]
    )
