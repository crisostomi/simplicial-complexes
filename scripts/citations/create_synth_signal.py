import argparse
import numpy as np
from tsp_sc.common.io import *
from tsp_sc.common.misc import *
from tsp_sc.citations.utils.citations import get_paths
import pytorch_lightning as pl

pl.seed_everything(seed=42)

parser = argparse.ArgumentParser()
parser.add_argument("config")
parser.add_argument("--starting_node")
cli_args = parser.parse_args()

config = load_config(cli_args.config)

add_cli_args(config, cli_args)

path_params, data_params, run_params = (
    config["paths"],
    config["data"],
    config["run_params"],
)
considered_simplex_dim = data_params["considered_simplex_dim"]

device = "cuda" if torch.cuda.is_available else "cpu"
paths = get_paths(path_params, data_params)

boundaries = np.load(paths["boundaries"], allow_pickle=True).tolist()
Bs = [None] + boundaries[: considered_simplex_dim + 1]

laplacians = np.load(paths["laplacians"], allow_pickle=True).tolist()
signals = np.load(paths["citations"], allow_pickle=True).tolist()
simplices = np.load(paths["simplices"], allow_pickle=True).tolist()


U_basis_irr = [None for i in range(considered_simplex_dim + 1)]

for k in range(considered_simplex_dim + 1):
    Btk_upper = Bs[k + 1].transpose().todense()
    Bk_upper = Bs[k + 1].todense()

    BBt = Bk_upper @ Btk_upper
    eigvals, eigvecs = np.linalg.eig(BBt)
    U_basis_irr[k] = eigvecs

U_basis_sol = [None for i in range(considered_simplex_dim + 1)]

for k in range(1, considered_simplex_dim + 1):

    Btk = Bs[k].transpose().todense()
    Bk = Bs[k].todense()

    BtB = Btk @ Bk
    eigvals, eigvecs = np.linalg.eig(BtB)
    U_basis_sol[k] = eigvecs
