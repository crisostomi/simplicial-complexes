import argparse

from tsp_sc.common.io import load_config
from tsp_sc.common.misc import *
from tsp_sc.graph_classification.utils import get_paths
from tsp_sc.graph_classification.data.datamodule import GraphClassificationDataModule
from tsp_sc.graph_classification.models.classification_scnn import ClassificationSCNN
from pytorch_lightning import Trainer


parser = argparse.ArgumentParser()
parser.add_argument("config")
cli_args = parser.parse_args()

config = load_config(cli_args.config)

path_params, data_params, run_params, model_params = (
    config["paths"],
    config["data"],
    config["run_params"],
    config["models"],
)

assert data_params["considered_simplex_dim"] <= data_params["max_simplex_dim"]


device = "cuda" if torch.cuda.is_available else "cpu"
paths = get_paths(path_params, data_params)

data_module = GraphClassificationDataModule(paths, data_params)

model = ClassificationSCNN(model_params["citation_scnn"])

trainer = Trainer(max_epochs=run_params["max_epochs"], gpus=run_params["gpus"])
# trainer.fit(model, data_module)

# trainer.test(model)
