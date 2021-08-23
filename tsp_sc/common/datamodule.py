from scipy.sparse import coo_matrix
from scipy.linalg import null_space
import pytorch_lightning as pl
from tsp_sc.common.simplices import normalize_laplacian
from tsp_sc.common.misc import coo2tensor


class TopologicalDataModule(pl.LightningDataModule):
    def __init__(self, paths, data_params):
        super(TopologicalDataModule, self).__init__()
        self.consider_last_dim_upper_adj = self.get_last_dim_upper_adj_flag(data_params)
        self.considered_simplex_dim = data_params["considered_simplex_dim"]
        self.batch_size = data_params["batch_size"]

    def get_orthogonal_components(self):

        components = {
            "full": self.laplacians,
            "sol": self.get_solenoidal_component(),
            "irr": self.get_irrotational_component(),
            "har": self.get_harmonic_component(),
        }

        return components

    def get_harmonic_component(self):
        har = [null_space(L[0].todense()) for L in self.laplacians]

        for i, ker in enumerate(har):
            print(f"L{i} has kernel dimension {ker.shape[1]}")

        return har

    def get_irrotational_component(self):
        irr = [None for i in range(self.considered_simplex_dim + 1)]

        for k in range(self.considered_simplex_dim + 1):
            irr_batch = []
            for b in range(self.batch_size):
                print(k, b)
                Btk_upper = self.boundaries[k + 1][b].transpose().todense()
                Bk_upper = self.boundaries[k + 1][b].todense()

                BBt = Bk_upper @ Btk_upper
                irr_batch.append(coo_matrix(BBt))
            irr[k] = irr_batch

        if not self.consider_last_dim_upper_adj:
            for b in range(self.batch_size):
                irr[b][self.considered_simplex_dim] = None

        return irr

    def get_solenoidal_component(self):
        sol = [None for i in range(self.considered_simplex_dim + 1)]

        for k in range(1, self.considered_simplex_dim + 1):
            sol_batch = []
            for b in range(self.batch_size):
                Btk = self.boundaries[k][b].transpose().todense()
                Bk = self.boundaries[k][b].todense()

                BtB = Btk @ Bk
                sol_batch.append(coo_matrix(BtB))
            sol[k] = sol_batch

        return sol

    def normalize_components(self):
        for i in range(0, self.considered_simplex_dim + 1):
            for comp in ["sol", "irr", "full"]:
                if self.components[comp][i] is not None:
                    for b in range(self.batch_size):
                        normalized = normalize_laplacian(
                            self.components[comp][i][b], half_interval=True
                        )
                        self.components[comp][i][b] = coo2tensor(normalized)

    def get_last_dim_upper_adj_flag(self, data_params):
        return (
            True
            if data_params["considered_simplex_dim"] < data_params["max_simplex_dim"]
            else False
        )
