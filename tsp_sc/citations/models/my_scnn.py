import torch.nn as nn
from tsp_sc.common.simplicial_convolution import MySimplicialConvolution
import torch
from tsp_sc.citations.models.citation_scnn import CitationSCNN


class MySCNN(CitationSCNN):
    def __init__(self, params):
        """
        parameters:
            filter_size: size of the convolutional filter
            colors: number of channels
            aggregation: how to aggregate the convolution outputs, can 'sum' or 'MLP'
            component_to_use: whether to use both components or one of the two, values can be 'both', 'sol' or 'irr'
            keep_separated: whether to keep the intermediate layer outputs separated or to aggregate them
        """
        super().__init__(params)

        assert params["colors"] > 0
        # if only one component is used, then they must be kept separated
        assert params["component_to_use"] == "both" or params["keep_separated"]

        self.considered_simplex_dim = params["considered_simplex_dim"]
        self.colors = params["colors"]
        self.aggregation = params["aggregation"]
        self.component_to_use = params["component_to_use"]
        self.keep_separated = params["keep_separated"]
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
        self.C["l1"]["d0"]["irr"] = MySimplicialConvolution(
            self.filter_size,
            C_in=self.colors,
            C_out=self.num_filters * self.colors,
            variance=self.variance,
        )
        self.C["l2"]["d0"]["irr"] = MySimplicialConvolution(
            self.filter_size,
            C_in=self.num_filters * self.colors,
            C_out=self.num_filters * self.colors,
            variance=self.variance,
        )
        self.C["l3"]["d0"]["irr"] = MySimplicialConvolution(
            self.filter_size,
            C_in=self.num_filters * self.colors,
            C_out=self.colors,
            variance=self.variance,
        )

        # degree 1 convolutions
        self.C["l1"]["d1"]["sol"] = MySimplicialConvolution(
            self.filter_size,
            C_in=self.colors,
            C_out=(self.num_filters // 2) * self.colors,
            variance=self.variance,
        )
        self.C["l2"]["d1"]["sol"] = MySimplicialConvolution(
            self.filter_size,
            C_in=(self.num_filters // 2) * self.colors,
            C_out=(self.num_filters // 2) * self.colors,
            variance=self.variance,
        )
        self.C["l3"]["d1"]["sol"] = MySimplicialConvolution(
            self.filter_size,
            C_in=(self.num_filters // 2) * self.colors,
            C_out=self.colors,
            variance=self.variance,
        )

        self.C["l1"]["d1"]["irr"] = MySimplicialConvolution(
            self.filter_size,
            C_in=self.colors,
            C_out=(self.num_filters // 2) * self.colors,
            variance=self.variance,
        )
        self.C["l2"]["d1"]["irr"] = MySimplicialConvolution(
            self.filter_size,
            C_in=(self.num_filters // 2) * self.colors,
            C_out=(self.num_filters // 2) * self.colors,
            variance=self.variance,
        )
        self.C["l3"]["d1"]["irr"] = MySimplicialConvolution(
            self.filter_size,
            C_in=(self.num_filters // 2) * self.colors,
            C_out=self.colors,
            variance=self.variance,
        )

        # degree 2 convolutions
        self.C["l1"]["d2"]["sol"] = MySimplicialConvolution(
            self.filter_size,
            C_in=self.colors,
            C_out=(self.num_filters // 2) * self.colors,
            variance=self.variance,
        )
        self.C["l2"]["d2"]["sol"] = MySimplicialConvolution(
            self.filter_size,
            C_in=(self.num_filters // 2) * self.colors,
            C_out=(self.num_filters // 2) * self.colors,
            variance=self.variance,
        )
        self.C["l3"]["d2"]["sol"] = MySimplicialConvolution(
            self.filter_size,
            C_in=(self.num_filters // 2) * self.colors,
            C_out=self.colors,
            variance=self.variance,
        )
        self.C["l1"]["d2"]["irr"] = MySimplicialConvolution(
            self.filter_size,
            C_in=self.colors,
            C_out=(self.num_filters // 2) * self.colors,
            variance=self.variance,
        )
        self.C["l2"]["d2"]["irr"] = MySimplicialConvolution(
            self.filter_size,
            C_in=(self.num_filters // 2) * self.colors,
            C_out=(self.num_filters // 2) * self.colors,
            variance=self.variance,
        )
        self.C["l3"]["d2"]["irr"] = MySimplicialConvolution(
            self.filter_size,
            C_in=(self.num_filters // 2) * self.colors,
            C_out=self.colors,
            variance=self.variance,
        )

        if self.aggregation == "MLP":
            self.L = nn.ModuleDict(
                {f"l{i}": nn.ModuleDict() for i in range(1, self.num_layers + 1)}
            )

            self.L["l1"]["d1"] = nn.Linear(
                2 * ((self.num_filters // 2) * self.colors),
                (self.num_filters // 2) * self.colors,
            )
            self.L["l1"]["d2"] = nn.Linear(
                2 * ((self.num_filters // 2) * self.colors),
                (self.num_filters // 2) * self.colors,
            )

            self.L["l2"]["d1"] = nn.Linear(
                2 * ((self.num_filters // 2) * self.colors),
                (self.num_filters // 2) * self.colors,
            )
            self.L["l2"]["d2"] = nn.Linear(
                2 * ((self.num_filters // 2) * self.colors),
                (self.num_filters // 2) * self.colors,
            )

            self.L["l3"]["d1"] = nn.Linear(2 * self.colors, self.colors)
            self.L["l3"]["d2"] = nn.Linear(2 * self.colors, self.colors)

        self.save_hyperparameters()

    def forward(self, inputs, components):
        """
        If self.keep_separated, the intermediate layers outputs are kept separated, and are aggregated only in the final layer
        Otherwise, these are summed in each layer

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

        if self.keep_separated:

            outs = {f"l{layer}": {f"d{dim}": {} for dim in dims} for layer in layers}
            outs["l0"] = {
                f"d{dim}": {comp: inputs[dim] for comp in comps[f"d{dim}"]}
                for dim in dims
            }

            for layer in layers[1:]:
                for dim in dims:
                    for comp in comps[f"d{dim}"]:
                        prev_output = activactions[layer](
                            outs[f"l{layer - 1}"][f"d{dim}"][comp]
                        )
                        outs[f"l{layer}"][f"d{dim}"][comp] = self.convolve(
                            prev_output, components, layer, dim, comp
                        )

            final_out0 = outs[last_layer]["d0"]["irr"]
            final_out1, final_out2 = self.merge_components(outs)

        else:

            outs = {f"l{layer}": {} for layer in layers}
            outs["l0"] = {f"d{dim}": inputs[dim] for dim in dims}

            for layer in layers[1:]:
                for dim in dims:
                    prev_output = activactions[layer](outs[f"l{layer - 1}"][f"d{dim}"])
                    comp_outputs = [
                        self.convolve(prev_output, components, layer, dim, comp)
                        for comp in comps[f"d{dim}"]
                    ]
                    outs[f"l{layer}"][f"d{dim}"] = self.aggregate(
                        comp_outputs, layer, dim
                    )

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
                [outs["l3"]["d1"]["sol"], outs["l3"]["d1"]["irr"]], layer=3, dim=2
            )
            final_out2 = self.aggregate(
                [outs["l3"]["d2"]["sol"], outs["l3"]["d2"]["irr"]], layer=3, dim=2
            )

        return final_out1, final_out2

    def aggregate(self, components_outputs, layer, dim):
        """
        Aggregates the output of the convolution over the different components
        if the output is from a single component (e.g. from the irrotational components for nodes),
        then it just returns it otherwise, depending on self.aggregation either
         aggregates by summing or by using a MLP
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

    def get_preds(self, batch):
        inputs = (
            batch["X0"],
            batch["X1"],
            batch["X2"],
        )
        components = self.get_components_from_batch(batch)

        preds = self(inputs, components)
        return preds

    def configure_optimizers(self):
        if not self.keep_separated and self.aggregation == "MLP":
            lr = 5e-3
        else:
            lr = 1e-3
        return torch.optim.Adam(self.parameters(), lr=lr)
