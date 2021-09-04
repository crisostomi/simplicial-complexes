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

        # degree 0 convolutions

        self.C0_1 = SimplicialConvolution(
            filter_size=self.filter_size,
            C_in=self.colors,
            C_out=self.num_filters * self.colors,
            variance=self.variance,
        )

        self.C0_2 = SimplicialConvolution(
            filter_size=self.filter_size,
            C_in=self.num_filters * self.colors,
            C_out=self.num_filters * self.colors,
            variance=self.variance,
        )
        self.C0_3 = SimplicialConvolution(
            filter_size=self.filter_size,
            C_in=self.num_filters * self.colors,
            C_out=self.colors,
            variance=self.variance,
        )

        # degree 1 convolutions
        self.C1_1 = SimplicialConvolution(
            filter_size=self.filter_size,
            C_in=self.colors,
            C_out=self.num_filters * self.colors,
            variance=self.variance,
        )
        self.C1_2 = SimplicialConvolution(
            filter_size=self.filter_size,
            C_in=self.num_filters * self.colors,
            C_out=self.num_filters * self.colors,
            variance=self.variance,
        )
        self.C1_3 = SimplicialConvolution(
            filter_size=self.filter_size,
            C_in=self.num_filters * self.colors,
            C_out=self.colors,
            variance=self.variance,
        )

        # degree 2 convolutions
        self.C2_1 = SimplicialConvolution(
            filter_size=self.filter_size,
            C_in=self.colors,
            C_out=self.num_filters * self.colors,
            variance=self.variance,
        )
        self.C2_2 = SimplicialConvolution(
            filter_size=self.filter_size,
            C_in=self.num_filters * self.colors,
            C_out=self.num_filters * self.colors,
            variance=self.variance,
        )
        self.C2_3 = SimplicialConvolution(
            filter_size=self.filter_size,
            C_in=self.num_filters * self.colors,
            C_out=self.colors,
            variance=self.variance,
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
        out0_1 = self.C0_1(Ls[0], xs[0])
        out1_1 = self.C1_1(Ls[1], xs[1])
        out2_1 = self.C2_1(Ls[2], xs[2])

        # 2nd pass of convolutions
        out0_2 = self.C0_2(Ls[0], nn.LeakyReLU()(out0_1))
        out1_2 = self.C1_2(Ls[1], nn.LeakyReLU()(out1_1))
        out2_2 = self.C2_2(Ls[2], nn.LeakyReLU()(out2_1))

        # 3rd pass of convolutions
        out0_3 = self.C0_3(Ls[0], nn.LeakyReLU()(out0_2))
        out1_3 = self.C1_3(Ls[1], nn.LeakyReLU()(out1_2))
        out2_3 = self.C2_3(Ls[2], nn.LeakyReLU()(out2_2))

        return [out0_3, out1_3, out2_3]

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

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
