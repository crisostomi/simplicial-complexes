from tsp_sc.citations.models.citation_scnn import CitationSCNN
import torch.nn as nn
import torch
from tsp_sc.common.simplicial_convolution import (
    DeffSimplicialConvolution as SimplicialConvolution,
)


class DeffSCNN(CitationSCNN):
    def __init__(self, params):
        super().__init__(params)

        self.colors = params["colors"]
        self.filter_size = params["filter_size"]
        self.num_filters = params["num_filters"]
        self.variance = params["variance"]
        self.considered_simplex_dim = params["considered_simplex_dim"]

        self.num_layers = 3
        self.num_dims = self.considered_simplex_dim + 1
        self.layers = [f"l{i}" for i in range(1, self.num_layers + 1)]
        self.dims = [f"d{i}" for i in range(self.considered_simplex_dim + 1)]

        self.C = nn.ModuleDict(
            {
                layer: nn.ModuleDict({dim: nn.ModuleDict() for dim in self.dims})
                for layer in self.layers
            }
        )

        # layer 1
        self.C["l1"] = nn.ModuleDict(
            {
                dim: SimplicialConvolution(
                    filter_size=self.filter_size,
                    C_in=self.colors,
                    C_out=self.num_filters * self.colors,
                    variance=self.variance,
                )
                for dim in self.dims
            }
        )

        # layer 2
        self.C["l2"] = nn.ModuleDict(
            {
                dim: SimplicialConvolution(
                    filter_size=self.filter_size,
                    C_in=self.num_filters * self.colors,
                    C_out=self.num_filters * self.colors,
                    variance=self.variance,
                )
                for dim in self.dims
            }
        )

        self.C["l3"] = nn.ModuleDict(
            {
                dim: SimplicialConvolution(
                    filter_size=self.filter_size,
                    C_in=self.num_filters * self.colors,
                    C_out=self.colors,
                    variance=self.variance,
                )
                for dim in self.dims
            }
        )

        self.save_hyperparameters()

    def forward(self, xs, laplacians):
        """
        parameters:
            components: the full, lower and upper Laplacians
                        only uses the full
            xs: inputs
        """
        Ls = laplacians

        # 1st pass of convolutions
        outs = {layer: {dim: {} for dim in self.dims} for layer in self.layers}

        for dim in range(self.num_dims):
            outs["l1"][f"d{dim}"] = self.C["l1"][f"d{dim}"](Ls[dim], xs[dim])

        # 2nd pass of convolutions
        for dim in range(self.num_dims):
            outs["l2"][f"d{dim}"] = self.C["l2"][f"d{dim}"](
                Ls[dim], nn.LeakyReLU()(outs["l1"][f"d{dim}"])
            )

        # 3rd pass of convolutions
        for dim in range(self.num_dims):
            outs["l3"][f"d{dim}"] = self.C["l3"][f"d{dim}"](
                Ls[dim], nn.LeakyReLU()(outs["l2"][f"d{dim}"])
            )

        return [outs["l3"][dim] for dim in self.dims]

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def get_preds(self, batch):
        inputs = [batch[f"X{i}"] for i in range(self.num_dims)]

        components = self.get_components_from_batch(batch)
        laplacians = components["full"]

        preds = self(inputs, laplacians)

        return preds
