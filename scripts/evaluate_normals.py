import argparse

from inter_order.utils.misc import *
from inter_order.utils.io import load_config
from inter_order.utils.plotter import Plotter
from inter_order.models.mesh_scnn import MeshSCNN
from inter_order.data.datamodule import NormalsDataModule
from pytorch_lightning import Trainer

parser = argparse.ArgumentParser()
parser.add_argument("config")
cli_args = parser.parse_args()

config = load_config(cli_args.config)

paths, run_params, model_params = (
    config["paths"],
    config["run_params"],
    config["models"],
)

print_torch_version()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

to_load = {
    "laplacians": list,
    "boundaries": list,
    "positions": dict,
    "noisy_positions": dict,
    "original_positions": list,
    "normals": dict,
    "triangles": list,
}
load_folder = os.path.join(paths["data"], paths["mesh_name"])
loaded_data = load_data(to_load, load_folder)

data_module = NormalsDataModule(loaded_data)

plotter = Plotter(
    loaded_data["original_positions"],
    loaded_data["triangles"],
    data_module.triangle_normals[0],
)
model = MeshSCNN(model_params["mesh_scnn"], plotter)

trainer = Trainer(max_epochs=run_params["max_epochs"], gpus=run_params["gpus"])

trainer.fit(model, data_module)
trainer.test(model)
