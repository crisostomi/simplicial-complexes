import argparse

from tsp_sc.common.io import load_config
from tsp_sc.citations.utils.citations import *


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


def clean(papers_df, edges_df):
    """
    Select a subset of the bipartite paper-author graph.
    Only drop papers. We want to keep all the authors from the selected papers.
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


def downsample(papers_df, edges_df):
    keep = grow_network(
        seed=papers_df.iloc[100].name, n_papers=30000, edges_df=edges_df
    )

    papers_df = papers_df.loc[papers_df.index.intersection(keep)]
    edges_df = edges_df[edges_df["paper"].isin(papers_df.index)]

    papers_df.sort_index(inplace=True)
    edges_df.sort_values("paper", inplace=True)
    edges_df.reset_index(drop=True, inplace=True)

    print("remaining data:")
    count(papers_df, edges_df)
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    cli_args = parser.parse_args()

    config = load_config(cli_args.config)
    paths = config["paths"]

    papers, edges = load_papers_authors_bipartite_graph(
        paths["paper_author_graph_full"]
    )

    papers, edges = clean(papers, edges)

    papers, edges = downsample(papers, edges)

    print(papers.head(n=3))
    print(edges.head(n=3))

    # map papers and authors to node ids

    papers, authors, edges = add_node_ids(papers, edges)

    print(papers.head(n=3))
    print(edges.head(n=3))
    print(authors.head(n=3))

    # build biadjacency matrix

    biadjacency = sparse.coo_matrix(
        (
            np.ones(len(edges), dtype=np.bool),
            (edges["paper_node_id"], edges["author_node_id"]),
        )
    )

    # save
    save_paper_author_biadj(papers, authors, edges, biadjacency, paths["raw"])
