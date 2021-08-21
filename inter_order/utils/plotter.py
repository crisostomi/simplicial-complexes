import plotly.figure_factory as ff
import numpy as np


class Plotter:
    def __init__(self, positions, triangles, true_normals):
        self.positions = positions
        self.triangles = triangles
        self.true_normals = true_normals
        self.pred_normals = []

    def plot_mesh(self, title, colors=None):
        x = self.positions[:, 0]
        y = self.positions[:, 1]
        z = self.positions[:, 2]

        if colors:
            fig = ff.create_trisurf(
                x=x,
                y=y,
                z=z,
                simplices=self.triangles,
                title=title,
                aspectratio=dict(x=1, y=1, z=0.3),
                color_func=colors,
            )
        else:
            fig = ff.create_trisurf(
                x=x,
                y=y,
                z=z,
                simplices=self.triangles,
                title=title,
                aspectratio=dict(x=1, y=1, z=0.3),
            )

        return fig

    def transform_normals_to_rgb(self, pred=False):
        normals = self.pred_normals if pred else self.true_normals

        normals = [norm.detach().cpu().numpy() for norm in normals]
        normals = 255 * (normals - np.min(normals)) / np.ptp(normals)
        normals = list(normals)

        normals_colors = [f"rgb({x}, {y}, {z})" for x, y, z in normals]
        return normals_colors
