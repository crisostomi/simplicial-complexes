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

        self.num_layers = 3
        self.num_dims = 3

        self.C = nn.ModuleDict(
            {
                f"l{i}": nn.ModuleDict(
                    {f"d{j}": nn.ModuleDict() for j in range(0, self.num_dims)}
                )
                for i in range(1, self.num_layers + 1)
            }
        )

        # degree 0 convolutions
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

        # degree 1 convolutions
        self.C["l1"]["d1"] = nn.ModuleList(
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
        self.C["l2"]["d1"] = nn.ModuleList(
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
        self.C["l3"]["d1"] = nn.ModuleList(
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

        # degree 2 convolutions
        self.C["l1"]["d2"] = nn.ModuleList(
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
        self.C["l2"]["d2"] = nn.ModuleList(
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
        self.C["l3"]["d2"] = nn.ModuleList(
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

        self.L = nn.ModuleDict(
            {f"l{i}": nn.ModuleDict() for i in range(1, self.num_layers + 1)}
        )

        self.L["l1"]["d1"] = nn.Linear(
            2 * (self.num_filters * self.colors), self.num_filters * self.colors,
        )
        self.L["l1"]["d2"] = nn.Linear(
            2 * (self.num_filters * self.colors), self.num_filters * self.colors,
        )

        self.L["l2"]["d1"] = nn.Linear(
            2 * (self.num_filters * self.colors), self.num_filters * self.colors,
        )
        self.L["l2"]["d2"] = nn.Linear(
            2 * (self.num_filters * self.colors), self.num_filters * self.colors,
        )

        self.L["l3"]["d1"] = nn.Linear(2 * self.colors, self.colors)
        self.L["l3"]["d2"] = nn.Linear(2 * self.colors, self.colors)

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
        outs_l1 = {f"d{i}": [] for i in range(self.num_dims)}
        for dim in range(self.num_dims):
            conv_outputs = [conv(Ls[dim], xs[dim]) for conv in self.C["l1"][f"d{dim}"]]
            outs_l1[f"d{dim}"] = self.aggregate(conv_outputs, 1, dim)

        # 2nd pass of convolutions
        outs_l2 = {f"d{i}": [] for i in range(self.num_dims)}
        for dim in range(self.num_dims):
            conv_outputs = [
                conv(Ls[dim], nn.LeakyReLU()(outs_l1[f"d{dim}"]))
                for conv in self.C["l2"][f"d{dim}"]
            ]
            outs_l2[f"d{dim}"] = self.aggregate(conv_outputs, 2, dim)

        # 3rd pass of convolutions
        outs_l3 = {f"d{i}": [] for i in range(self.num_dims)}
        for dim in range(self.num_dims):
            conv_outputs = [
                conv(Ls[dim], nn.LeakyReLU()(outs_l1[f"d{dim}"]))
                for conv in self.C["l3"][f"d{dim}"]
            ]
            outs_l3[f"d{dim}"] = self.aggregate(conv_outputs, 3, dim)

        return [outs_l3["d0"], outs_l3["d1"], outs_l3["d2"]]

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=5e-3)

    def get_preds(self, batch):
        inputs = (
            batch["X0"],
            batch["X1"],
            batch["X2"],
        )
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
