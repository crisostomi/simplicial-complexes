import torch.nn as nn
from tsp_sc.common.simplicial_convolution import MySimplicialConvolution
from tsp_sc.common.simp_complex import ComplexBatch
from torch_geometric.nn import JumpingKnowledge
import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from tsp_sc.common.utils import get_pooling_fn, get_nonlinearity
from torchmetrics import F1, Precision, Recall, Accuracy
from tsp_sc.graph_classification.models.mlp import MLP


class ClassificationSCNN(pl.LightningModule):
    def __init__(self, params):
        """
        parameters:
            filter_size: size of the convolutional filter
            colors: number of channels
            component_to_use: whether to use both components or one of the two, values can be 'both', 'sol' or 'irr'
    """
        super().__init__()

        self.F1 = F1(average="micro")
        self.prec = Precision(average="micro")
        self.recall = Recall(average="micro")
        self.accuracy = Accuracy()

        self.colors = params["colors"]
        self.aggregation = params["aggregation"]
        self.component_to_use = params["component_to_use"]
        self.keep_separated = params["keep_separated"]
        self.filter_size = params["filter_size"]
        self.readout = params["readout"]
        self.jump_mode = params["jump_mode"]
        self.global_nonlinearity = params["global_nonlinearity"]
        self.hidden_size = params["hidden_size"]
        self.num_classes = params["num_classes"]
        self.dropout_rate = params["dropout_rate"]
        self.feature_dim = params["num_features"]
        self.dim_aggregation = params["dim_aggregation"]
        self.learning_rate = params["learning_rate"]
        self.sched_step_size = params["sched_step_size"]
        self.sched_gamma = params["sched_gamma"]
        self.use_batch_norm = params["use_batch_norm"]
        self.comp_aggr_MLP_layers = params["comp_aggr_MLP_layers"]

        self.num_layers = params["num_layers"]
        self.num_dims = 3
        self.dims = range(0, self.num_dims)
        self.comps = self.get_comps(self.num_dims)
        self.layers = range(0, self.num_layers)
        self.hidden_sizes = [self.hidden_size for i in range(self.num_layers, 0, -1)]
        variance = 0.01

        self.C = nn.ModuleDict({f"d{j}": nn.ModuleDict() for j in self.dims})

        for dim in self.dims:

            self.C[f"d{dim}"] = nn.ModuleList()

            first_conv = nn.ModuleDict(
                {
                    comp: MySimplicialConvolution(
                        self.filter_size,
                        C_in=self.feature_dim[dim],
                        C_out=self.hidden_sizes[0],
                        variance=variance,
                    )
                    for comp in self.comps[f"d{dim}"]
                }
            )
            self.C[f"d{dim}"].append(first_conv)

            for layer in range(1, self.num_layers):
                convs = nn.ModuleDict(
                    {
                        comp: MySimplicialConvolution(
                            self.filter_size,
                            C_in=self.hidden_sizes[layer - 1],
                            C_out=self.hidden_sizes[layer],
                            variance=variance,
                        )
                        for comp in self.comps[f"d{dim}"]
                    }
                )
                self.C[f"d{dim}"].append(convs)

        self.L = nn.ModuleDict(
            {
                f"d{dim}": nn.ModuleList(
                    [
                        MLP(
                            num_layers=self.comp_aggr_MLP_layers,
                            input_dim=2 * self.hidden_sizes[layer],
                            hidden_dim=2 * self.hidden_sizes[layer],
                            output_dim=self.hidden_sizes[layer],
                        )
                        for layer in self.layers
                    ]
                )
                for dim in self.dims[1:]
            }
        )

        self.BN = nn.ModuleDict(
            {
                f"d{dim}": nn.ModuleList(
                    [nn.BatchNorm1d(self.hidden_sizes[layer]) for layer in self.layers]
                )
                for dim in self.dims
            }
        )

        self.activ = nn.ModuleList([nn.PReLU() for layer in self.layers])

        self.comp_aggr_activ = nn.ModuleList([nn.PReLU() for layer in self.layers])

        self.jump = (
            JumpingKnowledge(self.jump_mode) if self.jump_mode is not None else None
        )

        pooled_hidden_dim = (
            self.num_layers * self.hidden_size
            if self.jump_mode == "cat"
            else self.hidden_size
        )

        self.linear_dim_aggregation = (
            nn.Linear(
                self.num_dims * pooled_hidden_dim, self.hidden_size * self.num_layers,
            )
            if self.dim_aggregation == "linear"
            else None
        )

        self.dim_aggr_non_lin = nn.PReLU() if self.dim_aggregation == "linear" else None

        self.final_lin1 = nn.Linear(pooled_hidden_dim, self.hidden_size)

        self.final_lin2 = nn.Linear(self.hidden_size, self.num_classes)

        self.cross_entropy = nn.CrossEntropyLoss(reduction=params["reduction"])
        self.pooling_fn = get_pooling_fn(self.readout)

    def forward(self, inputs, components, batch):
        """
        parameters:
            components: dict of lists, keys: 'full' for the Laplacian, 'sol' for the solenoidal and 'irr' for irrotational component
                        for each component, there is a list of length self.dims containing for each dimension the component for that dimension
            xs: inputs
        """

        layers = range(self.num_layers)
        last_layer = self.num_layers - 1

        dims = range(batch.dimension + 1)
        comps = self.get_comps(batch.dimension)

        # outs[layer] is a dict that contains for each dim the corresponding output
        outs = [{} for layer in layers]
        jump_xs = None

        for layer in layers:
            for dim in dims:
                prev_output = (
                    inputs[dim].transpose(1, 0)
                    if layer == 0
                    else self.activ[layer](outs[layer - 1][f"d{dim}"])
                )
                comp_outputs = [
                    self.convolve(prev_output, components, layer, dim, comp)
                    for comp in comps[f"d{dim}"]
                ]

                # shape (N=num_simplices_dim, C=hidden_dim)
                aggregated = self.aggregate(comp_outputs, layer, dim)
                if self.use_batch_norm:
                    aggregated = self.batch_normalize(aggregated, dim, layer)
                outs[layer][f"d{dim}"] = aggregated

            if self.jump_mode is not None:
                if jump_xs is None:
                    jump_xs = {f"d{dim}": [] for dim in self.dims}
                for (dim, out) in outs[layer].items():
                    jump_xs[dim] += [out]

        cochain_outputs = [
            outs[last_layer][f"d{i}"] for i in range(batch.dimension + 1)
        ]

        if self.jump_mode is not None:
            cochain_outputs = self.jump_complex(jump_xs)

        # new_hidden_dim = hidden_dim * num_layers if jump mode == cat else hidden_dim
        # (num_dims, batch_size, new_hidden_dim)
        pooled_xs = self.pool_complex(cochain_outputs, batch)

        # shape (batch_size, new_hidden_dim)
        x = self.aggregate_dims(pooled_xs)

        model_nonlinearity = get_nonlinearity(
            self.global_nonlinearity, return_module=False
        )

        x = model_nonlinearity(self.final_lin1(x))

        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.final_lin2(x)

        return x

    def aggregate_dims(self, xs):
        """
        :param xs: (num_dims, batch_size, hidden_dim) if self.jump_mode != cat
                    else (num_dims, batch_size, hidden_dim*num_layers)
        :return: if sum: (batch_size, hidden_dim)
                 if cat: (batch_size, 3 x hidden_dim)
        """
        batch_size = xs.shape[1]
        if self.dim_aggregation == "sum":
            return xs.sum(dim=0)
        elif self.dim_aggregation == "linear":
            # (batch_size, num_dims, hidden_dim)
            xs = xs.transpose(1, 0)
            # (batch_size, num_dims * hidden_dim)
            xs = xs.reshape((batch_size, -1))
            # (batch_size, hidden_dim)
            out = self.linear_dim_aggregation(xs)
            out = self.dim_aggr_non_lin(out)
            return out
        else:
            raise NotImplementedError(
                f"Aggregation: {self.dim_aggregation} not supported."
            )

    def batch_normalize(self, input, dim, layer):
        """
        :param input: shape (N=num_simplices_dim, C=hidden_size)
        :param dim: dimension of the simplices
        :param layer: layer number in the network
        :return: output shape (hidden_size, num_simplices_dim)
        """
        assert input.shape[1] == self.hidden_size

        output = self.BN[f"d{dim}"][layer](input)

        return output

    @staticmethod
    def get_comps(max_dim):
        comps = {}
        dims = range(max_dim + 1)
        for dim in dims:
            if dim == 0:
                dim_comps = ["irr"]
            elif dim == max_dim:
                dim_comps = ["sol"]
            else:
                dim_comps = ["sol", "irr"]
            comps[f"d{dim}"] = dim_comps
        return comps

    def convolve(self, input, components, layer, dim, component):
        """
        Convolves input using the given component

        parameters:
            input: shape (N=num_simplices_dim, C=num_features) input signal to convolve
            components: dict of lists, keys: 'full' for the Laplacian, 'sol' for the solenoidal and 'irr' for irrotational component
                        for each component, there is a list of length self.dims containing for each dimension the component for that dimension
            layer: int, number of the layer
            dim: int, dimension of the simplices being considered
            component: string, 'full', 'sol' or 'irr'
    """
        C = self.feature_dim[dim] if layer == 0 else self.hidden_size
        assert input.shape[1] == C

        convolution = self.C[f"d{dim}"][layer][component]

        # convolution expects input of shape (C, N)
        input = input.transpose(1, 0)
        output = convolution(components[component][dim], input)

        output = output.transpose(1, 0)
        return output

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

        # aggregation via MLP
        else:
            linear_layer = self.L[f"d{dim}"][layer]
            out_concat = torch.cat(components_outputs, 1)
            out = linear_layer(out_concat)

        return out

    def jump_complex(self, jump_xs):
        """
        :param jump_xs: dictionary containing for each dim a list containing the output of each layer
                    { 'd0': [out_dim_0_layer_0, out_dim_0_layer_1, ... ], 'd1': [...] ... }
        :return: xs: list containing the concatenation
        """
        jump_xs = [jump_xs_dim for dim, jump_xs_dim in jump_xs.items()]
        xs = [self.jump(x) for x in jump_xs]
        return xs

    def training_step(self, batch: ComplexBatch, batch_idx):
        preds = self.get_preds(batch)

        labels = batch.get_labels()

        loss = self.cross_entropy(preds, labels)

        self.log("loss", loss, on_epoch=True, on_step=True, prog_bar=True)

        return loss

    def validation_step(self, batch: ComplexBatch, batch_idx):
        preds = self.get_preds(batch)

        labels = batch.get_labels()

        loss = self.cross_entropy(preds, labels)

        self.log("val/loss", loss, on_epoch=True, on_step=True, prog_bar=True)

        preds = torch.argmax(preds, -1)
        targets = batch.get_labels()

        acc = self.accuracy(preds, targets)

        self.log("val/acc", acc, on_epoch=True, on_step=True, logger=True)

        return {"val/loss": loss, "preds": preds, "targets": targets}

    def get_preds(self, batch):

        inputs = [
            batch.cochains[i]["signal"].transpose(1, 0)
            for i in range(batch.dimension + 1)
        ]

        components = self.get_components_from_batch(batch)

        preds = self(inputs, components, batch)
        return preds

    def get_components_from_batch(self, batch):
        dim = batch.dimension
        cochains = batch.cochains

        S = [None] + [cochains[i]["solenoidal"] for i in range(1, dim + 1)]
        I = [cochains[i]["irrotational"] for i in range(dim)] + [None]
        L = [cochains[i]["laplacian"] for i in range(dim + 1)]

        components = {"full": L, "irr": I, "sol": S}
        return components

    def test_step(self, batch, batch_idx):
        preds = self.get_preds(batch)
        preds = torch.argmax(preds, -1)
        targets = batch.get_labels()

        F1 = self.F1(preds, targets)
        prec = self.prec(preds, targets)
        recall = self.recall(preds, targets)
        acc = self.accuracy(preds, targets)

        self.log("test/F1", F1, on_epoch=True, logger=True)
        self.log("test/recall", prec, on_epoch=True, logger=True)
        self.log("test/prec", recall, on_epoch=True, logger=True)
        self.log("test/acc", acc, on_epoch=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.sched_step_size, gamma=self.sched_gamma
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def pool_complex(self, xs, batch):
        """

        :param xs: list (num_dims) where each item is a tensor (N=num_simplexes_dim, C=hidden_size)
                    if jump_mode is not cat, else (num_simplexes_dim, num_layers * hidden_size)
        :return: pooled complex (num_dims, batch_size, hidden_size)
        """
        feature_dim = (
            self.hidden_size
            if self.jump_mode != "cat"
            else self.num_layers * self.hidden_size
        )

        assert xs[0].shape[1] == feature_dim

        num_dims = len(xs)
        # each x has shape (num_simplexes_dim, hidden_size)

        batch_size = batch.cochains[0].batch.max() + 1

        # output is of shape [num_dims, batch_size, feature_dim]
        pooled_xs = torch.zeros(
            self.num_dims, batch_size, feature_dim, device=batch_size.device
        )

        for dim in range(num_dims):
            pooled = self.pooling_fn(
                xs[dim], batch.cochains[dim].batch, size=batch_size
            )
            pooled_xs[dim, :, :] = pooled

        return pooled_xs
