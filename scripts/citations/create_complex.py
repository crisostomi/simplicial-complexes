import argparse

from tsp_sc.common.io import load_config
from tsp_sc.common.simplices import (
    build_laplacians,
    build_boundaries,
)
from tsp_sc.citations.utils.citations import *

parser = argparse.ArgumentParser()
parser.add_argument("config")
parser.add_argument("starting_node")
cli_args = parser.parse_args()

config = load_config(cli_args.config)
paths = config["paths"]


adjacency = sparse.load_npz(paths["biadjacency_matrix_path"])
starting_node = cli_args.starting_node

# starting_node = 150250 #Defferard
COMPLEX_FOLDER_START_NODE = os.path.join(paths["complex_folder"], starting_node)
downsample_papers = np.load(os.path.join(COMPLEX_FOLDER_START_NODE, "downsampled.npy"))

papers_df_path = os.path.join(paths["raw"], "papers.csv")
papers_df = pd.read_csv(papers_df_path, index_col=0)

# shape (num_papers, )
citations = np.array(papers_df["citations_2019"])

# Bipartite to cochains
simplices, cochains, signals_top = bipart2simpcochain(
    adjacency, citations, indices_x=downsample_papers, dimension=10
)

num_nodes = len(simplices[0])
num_edges = len(simplices[1])
num_triangles = len(simplices[2])
print(f"There are {num_nodes} nodes, {num_edges} edges and {num_triangles} triangles")

# save
cochains_path = os.path.join(COMPLEX_FOLDER_START_NODE, f"cochains.npy")
simplices_path = os.path.join(COMPLEX_FOLDER_START_NODE, f"simplices.npy")

np.save(cochains_path, cochains)
np.save(simplices_path, simplices)

boundaries = build_boundaries(simplices)

laplacians = build_laplacians(boundaries)

# save
boundaries_path = os.path.join(COMPLEX_FOLDER_START_NODE, f"boundaries.npy")
laplacians_path = os.path.join(COMPLEX_FOLDER_START_NODE, f"laplacians.npy")

np.save(laplacians_path, laplacians)
np.save(boundaries_path, boundaries)
