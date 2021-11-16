from scipy.sparse import coo_matrix
from scipy.linalg import null_space
import pytorch_lightning as pl
from tsp_sc.common.simplices import normalize_laplacian
from tsp_sc.common.misc import coo2tensor
from scipy.sparse import linalg


class TopologicalDataModule(pl.LightningDataModule):
    def __init__(self, paths, data_params):
        super(TopologicalDataModule, self).__init__()
        self.consider_last_dim_upper_adj = self.get_last_dim_upper_adj_flag(data_params)
        self.considered_simplex_dim = data_params["considered_simplex_dim"]
        self.max_simplex_dim = data_params["max_simplex_dim"]
        self.batch_size = data_params["batch_size"]

    def get_orthogonal_components(self):

        components = {
            "full": self.laplacians,
            "sol": self.get_solenoidal_component(),
            "irr": self.get_irrotational_component(),
            "har": None,
        }

        return components

    def get_harmonic_component(self):
        har = [null_space(L[0].todense()) for L in self.laplacians]

        for i, ker in enumerate(har):
            print(f"L{i} has kernel dimension {ker.shape[1]}")

        return har

    def get_irrotational_component(self):
        irr = [None for i in range(self.considered_simplex_dim + 1)]

        dimensions = (
            self.considered_simplex_dim + 1
            if self.consider_last_dim_upper_adj
            else self.considered_simplex_dim
        )

        for k in range(dimensions):
            irr_dim_k = []
            for i in range(self.num_complexes):
                Btk_upper = self.boundaries[k + 1][i].transpose().todense()
                Bk_upper = self.boundaries[k + 1][i].todense()

                BBt = Bk_upper @ Btk_upper
                irr_dim_k.append(coo_matrix(BBt))
            irr[k] = irr_dim_k

        return irr

    def get_solenoidal_component(self):
        sol = [None for i in range(self.considered_simplex_dim + 1)]

        for k in range(1, self.considered_simplex_dim + 1):
            sol_dim_k = []
            for i in range(self.num_complexes):
                Btk = self.boundaries[k][i].transpose().todense()
                Bk = self.boundaries[k][i].todense()

                BtB = Btk @ Bk
                sol_dim_k.append(coo_matrix(BtB))
            sol[k] = sol_dim_k

        return sol

    def normalize_components(self):
        for k in range(0, self.considered_simplex_dim + 1):
            lap_largest_eigenvalue = linalg.eigsh(
                self.components["full"][k][0],
                k=1,
                which="LM",
                return_eigenvectors=False,
            )[0]
            for comp in ["sol", "irr", "full"]:
                if self.components[comp][k] is not None:
                    for i in range(self.num_complexes):
                        normalized = normalize_laplacian(
                            self.components[comp][k][i],
                            lap_largest_eigenvalue,
                            half_interval=True,
                        )
                        self.components[comp][k][i] = coo2tensor(normalized)

    def get_last_dim_upper_adj_flag(self, data_params):
        return (
            True
            if data_params["considered_simplex_dim"] < data_params["max_simplex_dim"]
            else False
        )
