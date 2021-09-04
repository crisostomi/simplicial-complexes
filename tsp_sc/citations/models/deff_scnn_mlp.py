from tsp_sc.citations.models.citation_scnn import CitationSCNN
import torch.nn as nn
import torch
from tsp_sc.common.simplicial_convolution import (
    DeffSimplicialConvolution as SimplicialConvolution,
)


class DeffSCNNMLP(CitationSCNN):
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
        self.dims = [f"d{i}" for i in range(self.num_dims)]

        self.C = nn.ModuleDict(
            {
                f"l{i}": nn.ModuleDict(
                    {f"d{j}": nn.ModuleDict() for j in range(0, self.num_dims)}
                )
                for i in range(1, self.num_layers + 1)
            }
        )

        # layer 1
        self.C["l1"]["d0"] = nn.ModuleList(
            [
                SimplicialConvolution(
                    filter_size=self.filter_size,
                    C_in=self.colors,
                    C_out=self.num_filters * self.colors,
                    variance=self.variance,
                )
            ]
        )
        for dim in self.dims[1:]:
            self.C["l1"][dim] = nn.ModuleList(
                [
                    SimplicialConvolution(
                        filter_size=self.filter_size,
                        C_in=self.colors,
                        C_out=self.num_filters * self.colors,
                        variance=self.variance,
                    )
                    for i in range(2)
                ]
            )

        # layer 2
        self.C["l2"]["d0"] = nn.ModuleList(
            [
                SimplicialConvolution(
                    filter_size=self.filter_size,
                    C_in=self.num_filters * self.colors,
                    C_out=self.num_filters * self.colors,
                    variance=self.variance,
                )
            ]
        )

        for dim in self.dims[1:]:
            self.C["l2"][dim] = nn.ModuleList(
                [
                    SimplicialConvolution(
                        filter_size=self.filter_size,
                        C_in=self.num_filters * self.colors,
                        C_out=self.num_filters * self.colors,
                        variance=self.variance,
                    )
                    for i in range(2)
                ]
            )

        # layer 3
        self.C["l3"]["d0"] = nn.ModuleList(
            [
                SimplicialConvolution(
                    filter_size=self.filter_size,
                    C_in=self.num_filters * self.colors,
                    C_out=self.colors,
                    variance=self.variance,
                )
            ]
        )

        for dim in self.dims[1:]:
            self.C["l3"][dim] = nn.ModuleList(
                [
                    SimplicialConvolution(
                        filter_size=self.filter_size,
                        C_in=self.num_filters * self.colors,
                        C_out=self.colors,
                        variance=self.variance,
                    )
                    for i in range(2)
                ]
            )

        # aggregation
        self.L = nn.ModuleDict({layer: nn.ModuleDict() for layer in self.layers})

        # layer 1
        for dim in self.dims[1:]:
            self.L["l1"][dim] = nn.Linear(
                2 * (self.num_filters * self.colors), self.num_filters * self.colors,
            )

        # layer 2
        for dim in self.dims[1:]:
            self.L["l2"][dim] = nn.Linear(
                2 * (self.num_filters * self.colors), self.num_filters * self.colors,
            )

        # layer 3
        for dim in self.dims[1:]:
            self.L["l3"][dim] = nn.Linear(2 * self.colors, self.colors)

        self.save_hyperparameters()

    def forward(self, xs, laplacians):
        """
        parameters:
            components: the full, lower and upper Laplacians
                        only uses the full
            xs: inputs
        """
        Ls = laplacians

        # 1st layer
        outs_l1 = {f"d{i}": [] for i in range(self.num_dims)}
        for dim in range(self.num_dims):
            conv_outputs = [conv(Ls[dim], xs[dim]) for conv in self.C["l1"][f"d{dim}"]]
            outs_l1[f"d{dim}"] = self.aggregate(conv_outputs, 1, dim)

        # 2nd layer
        outs_l2 = {f"d{i}": [] for i in range(self.num_dims)}
        for dim in range(self.num_dims):
            conv_outputs = [
                conv(Ls[dim], nn.LeakyReLU()(outs_l1[f"d{dim}"]))
                for conv in self.C["l2"][f"d{dim}"]
            ]
            outs_l2[f"d{dim}"] = self.aggregate(conv_outputs, 2, dim)

        # 3rd layer
        outs_l3 = {f"d{i}": [] for i in range(self.num_dims)}
        for dim in range(self.num_dims):
            conv_outputs = [
                conv(Ls[dim], nn.LeakyReLU()(outs_l1[f"d{dim}"]))
                for conv in self.C["l3"][f"d{dim}"]
            ]
            outs_l3[f"d{dim}"] = self.aggregate(conv_outputs, 3, dim)

        return [outs_l3[dim] for dim in self.dims]

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=5e-3)

    def get_preds(self, batch):
        inputs = [batch[f"X{i}"] for i in range(self.num_dims)]

        components = self.get_components_from_batch(batch)
        laplacians = components["full"]

        preds = self(inputs, laplacians)

        return preds

    def aggregate(self, conv_outputs, layer, dim):
        """
        Aggregates the output of the convolution over the different components
        if the output is from a single component (e.g. from the irrotational components for nodes),
        then it just returns it otherwise, aggregates by using a MLP
        """

        if len(conv_outputs) == 1:
            return conv_outputs[0]

        (c_out, num_simplices) = conv_outputs[0].shape

        reshaped_outs = [out.reshape(num_simplices, c_out) for out in conv_outputs]

        out_concat = torch.cat(reshaped_outs, 1)

        out = self.L[f"l{layer}"][f"d{dim}"](out_concat)

        out = out.reshape(c_out, num_simplices)

        return out
