import argparse

from tsp_sc.common.io import *
from tsp_sc.common.misc import *
from tsp_sc.citations.utils.citations import get_paths
from tsp_sc.citations.utils.utils import get_model
from tsp_sc.citations.data.datamodule import CitationDataModule
from pytorch_lightning import Trainer
import wandb
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

pl.seed_everything(seed=42)


wandb.login()

parser = argparse.ArgumentParser()
parser.add_argument("config")
parser.add_argument("--starting_node")
cli_args = parser.parse_args()

config = load_config(cli_args.config)

add_cli_args(config, cli_args)

path_params, data_params, run_params, model_params = (
    config["paths"],
    config["data"],
    config["run_params"],
    config["models"],
)

assert data_params["considered_simplex_dim"] <= data_params["max_simplex_dim"]

device = "cuda" if torch.cuda.is_available else "cpu"
paths = get_paths(path_params, data_params)

data_module = CitationDataModule(paths, data_params)

# model_names = list(model_params.keys())
model_names = ["my_scnn_sol", "my_scnn_irr"]

for model_name in model_names:
    model_params[model_name]["considered_simplex_dim"] = data_params[
        "considered_simplex_dim"
    ]

    model = get_model(model_name, model_params[model_name])

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=run_params["min_delta"],
        patience=run_params["patience"],
        verbose=False,
        mode="min",
    )
    run_config = get_run_config(model_name, config)
    wandb_logger = WandbLogger(name=model_name, project="dummy", config=run_config)
    wandb_logger.watch(model)
    wandb_logger.log_hyperparams({"num_params": num_params(model)})

    trainer = Trainer(
        max_epochs=run_params["max_epochs"],
        gpus=run_params["gpus"],
        logger=wandb_logger,
        callbacks=[early_stopping_callback],
    )
    trainer.fit(model, data_module)

    trainer.test(model)

    wandb.finish()
