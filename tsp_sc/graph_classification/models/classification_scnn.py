import torch.nn as nn
from tsp_sc.common.misc import similar
from tsp_sc.common.simplicial_convolution import MySimplicialConvolution
import torch
import pytorch_lightning as pl


class ClassificationSCNN(pl.LightningModule):
    def __init__(self, params):
        """
    parameters:
        filter_size: size of the convolutional filter
        colors: number of channels
        aggregation: how to aggregate the convolution outputs, can 'sum' or 'MLP'
        component_to_use: whether to use both components or one of the two, values can be 'both', 'sol' or 'irr'
        keep_separated: whether to keep the intermediate layer outputs separated or to aggregate them
    """
        super().__init__()

        assert params["colors"] > 0
        # if only one component is used, then they must be kept separated
        assert params["component_to_use"] == "both" or params["keep_separated"]

        self.colors = params["colors"]
        self.aggregation = params["aggregation"]
        self.component_to_use = params["component_to_use"]
        self.keep_separated = params["keep_separated"]
        self.filter_size = params["filter_size"]

        num_filters = 30
        variance = 0.01
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
        self.C["l1"]["d0"]["irr"] = MySimplicialConvolution(
            self.filter_size,
            C_in=self.colors,
            C_out=num_filters * self.colors,
            variance=variance,
        )
        self.C["l2"]["d0"]["irr"] = MySimplicialConvolution(
            self.filter_size,
            C_in=num_filters * self.colors,
            C_out=num_filters * self.colors,
            variance=variance,
        )
        self.C["l3"]["d0"]["irr"] = MySimplicialConvolution(
            self.filter_size,
            C_in=num_filters * self.colors,
            C_out=self.colors,
            variance=variance,
        )

        # degree 1 convolutions
        self.C["l1"]["d1"]["sol"] = MySimplicialConvolution(
            self.filter_size,
            C_in=self.colors,
            C_out=(num_filters // 2) * self.colors,
            variance=variance,
        )
        self.C["l2"]["d1"]["sol"] = MySimplicialConvolution(
            self.filter_size,
            C_in=(num_filters // 2) * self.colors,
            C_out=(num_filters // 2) * self.colors,
            variance=variance,
        )
        self.C["l3"]["d1"]["sol"] = MySimplicialConvolution(
            self.filter_size,
            C_in=(num_filters // 2) * self.colors,
            C_out=self.colors,
            variance=variance,
        )

        self.C["l1"]["d1"]["irr"] = MySimplicialConvolution(
            self.filter_size,
            C_in=self.colors,
            C_out=(num_filters // 2) * self.colors,
            variance=variance,
        )
        self.C["l2"]["d1"]["irr"] = MySimplicialConvolution(
            self.filter_size,
            C_in=(num_filters // 2) * self.colors,
            C_out=(num_filters // 2) * self.colors,
            variance=variance,
        )
        self.C["l3"]["d1"]["irr"] = MySimplicialConvolution(
            self.filter_size,
            C_in=(num_filters // 2) * self.colors,
            C_out=self.colors,
            variance=variance,
        )

        # degree 2 convolutions
        self.C["l1"]["d2"]["sol"] = MySimplicialConvolution(
            self.filter_size,
            C_in=self.colors,
            C_out=(num_filters // 2) * self.colors,
            variance=variance,
        )
        self.C["l2"]["d2"]["sol"] = MySimplicialConvolution(
            self.filter_size,
            C_in=(num_filters // 2) * self.colors,
            C_out=(num_filters // 2) * self.colors,
            variance=variance,
        )
        self.C["l3"]["d2"]["sol"] = MySimplicialConvolution(
            self.filter_size,
            C_in=(num_filters // 2) * self.colors,
            C_out=self.colors,
            variance=variance,
        )

        self.C["l1"]["d2"]["irr"] = MySimplicialConvolution(
            self.filter_size,
            C_in=self.colors,
            C_out=(num_filters // 2) * self.colors,
            variance=variance,
        )
        self.C["l2"]["d2"]["irr"] = MySimplicialConvolution(
            self.filter_size,
            C_in=(num_filters // 2) * self.colors,
            C_out=(num_filters // 2) * self.colors,
            variance=variance,
        )
        self.C["l3"]["d2"]["irr"] = MySimplicialConvolution(
            self.filter_size,
            C_in=(num_filters // 2) * self.colors,
            C_out=self.colors,
            variance=variance,
        )

        if self.aggregation == "MLP":
            self.L = nn.ModuleDict(
                {f"l{i}": nn.ModuleDict() for i in range(1, self.num_layers + 1)}
            )

            self.L["l1"]["d1"] = nn.Linear(
                2 * ((num_filters // 2) * self.colors), (num_filters // 2) * self.colors
            )
            self.L["l1"]["d2"] = nn.Linear(
                2 * ((num_filters // 2) * self.colors), (num_filters // 2) * self.colors
            )

            self.L["l2"]["d1"] = nn.Linear(
                2 * ((num_filters // 2) * self.colors), (num_filters // 2) * self.colors
            )
            self.L["l2"]["d2"] = nn.Linear(
                2 * ((num_filters // 2) * self.colors), (num_filters // 2) * self.colors
            )

            self.L["l3"]["d1"] = nn.Linear(2 * self.colors, self.colors)
            self.L["l3"]["d2"] = nn.Linear(2 * self.colors, self.colors)

    def forward(self, inputs, components):
        """
        parameters:
            components: dict of lists, keys: 'full' for the Laplacian, 'sol' for the solenoidal and 'irr' for irrotational component
                        for each component, there is a list of length self.dims containing for each dimension the component for that dimension
            xs: inputs
        """

        layers = range(self.num_layers + 1)
        dims = range(self.num_dims)

        comps = {f"d{dim}": ["irr"] if dim == 0 else ["sol", "irr"] for dim in dims}
        activactions = {
            layer: nn.Identity() if layer == 1 else nn.LeakyReLU() for layer in layers
        }
        last_layer = f"l{self.num_layers}"

        outs = {f"l{layer}": {} for layer in layers}
        outs["l0"] = {f"d{dim}": inputs[dim] for dim in dims}

        for layer in layers[1:]:
            for dim in dims:
                prev_output = activactions[layer](outs[f"l{layer - 1}"][f"d{dim}"])
                comp_outputs = [
                    self.convolve(prev_output, components, layer, dim, comp)
                    for comp in comps[f"d{dim}"]
                ]
                outs[f"l{layer}"][f"d{dim}"] = self.aggregate(comp_outputs, layer, dim)

        final_out0, final_out1, final_out2 = (
            outs[last_layer]["d0"],
            outs[last_layer]["d1"],
            outs[last_layer]["d2"],
        )

        return [final_out0, final_out1, final_out2]

    def convolve(self, input, components, layer, dim, component):
        """
        Convolves input using the given component

        parameters:
            input: input signal to convolve
            components: dict of lists, keys: 'full' for the Laplacian, 'sol' for the solenoidal and 'irr' for irrotational component
                        for each component, there is a list of length self.dims containing for each dimension the component for that dimension
            layer: int, number of the layer
            dim: int, dimension of the simplices being considered
            component: string, 'full', 'sol' or 'irr'
    """
        convolution = self.C[f"l{layer}"][f"d{dim}"][component]
        # TODO: handle batching
        return convolution(components[component][dim], input)

    def merge_components(self, outs):
        """
        merges the outputs of the final layer
        if self.component_to_use is 'sol' or 'irr', then it only uses that component
        otherwise for each dimension the components are aggregated
    """

        if self.component_to_use != "both":
            final_out1, final_out2 = (
                outs["l3"]["d1"][self.component_to_use],
                outs["l3"]["d2"][self.component_to_use],
            )

        else:
            final_out1 = self.aggregate(
                outs["l3"]["d1"]["sol"], outs["l3"]["d1"]["irr"], "d1"
            )
            final_out2 = self.aggregate(
                outs["l3"]["d2"]["sol"], outs["l3"]["d2"]["irr"], "d2"
            )

        return final_out1, final_out2

    def aggregate(self, components_outputs, layer, dim):
        """
        Aggregates the output of the convolution over the different components
        if the output is from a single component (e.g. from the irrotational components for nodes), then it just returns it
        otherwise, depending on self.aggregation either aggregates by summing or by using a MLP
    """

        if len(components_outputs) == 1:
            return components_outputs[0]

        if self.aggregation == "sum":
            out = sum(components_outputs)

        # aggregation via MLP
        else:

            (c_out, num_simplices) = components_outputs[0].shape

            reshaped_outs = [
                out.reshape(num_simplices, c_out) for out in components_outputs
            ]

            out_concat = torch.cat(reshaped_outs, 1)

            out = self.L[f"l{layer}"][f"d{dim}"](out_concat)

            out = out.reshape(c_out, num_simplices)

        return out

    def training_step(self, batch, batch_idx):

        preds = self.get_preds(batch)

        criterion = nn.CrossEntropyLoss()
        loss = torch.FloatTensor([0.0]).type_as(X0)

        self.log("loss", loss)
        return loss

    def get_preds(self, batch):
        cochains = batch.cochains

        inputs = (
            cochains[0]["signal"],
            cochains[1]["signal"],
            cochains[2]["signal"],
        )

        components = self.get_components_from_batch(cochains)

        preds = self(inputs, components)
        return preds

    def get_components_from_batch(self, cochains):

        S0, S1, S2 = None, cochains[1]["solenoidal"], cochains[2]["solenoidal"]
        I0, I1, I2 = cochains[0]["irrotational"], cochains[1]["irrotational"], None
        L0, L1, L2 = (
            cochains[0]["laplacian"],
            cochains[1]["laplacian"],
            cochains[2]["laplacian"],
        )
        components = {"full": [L0, L1, L2], "irr": [I0, I1, I2], "sol": [S0, S1, S2]}
        return components

    def test_step(self, batch, batch_idx):
        preds = self.get_preds(batch)
        targets = (batch["Y0"], batch["Y1"], batch["Y2"])

        return {"preds": preds, "targets": targets}

    def test_epoch_end(self, test_batch_outputs):
        preds = [batch["preds"] for batch in test_batch_outputs][0]
        targets = [batch["targets"] for batch in test_batch_outputs][0]

        # self.log_dict()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
