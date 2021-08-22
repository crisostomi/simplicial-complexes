import argparse

from tsp_sc.common.io import load_config
from tsp_sc.common.misc import *
from tsp_sc.citations.data.datamodule import CitationDataModule
from tsp_sc.citations.models.citation_scnn import CitationSCNN
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

data_module = CitationDataModule(paths, data_params, run_params)

model = CitationSCNN(model_params["citation_scnn"])

trainer = Trainer(max_epochs=run_params["max_epochs"], gpus=run_params["gpus"])
trainer.fit(model, data_module)

trainer.test(model)
