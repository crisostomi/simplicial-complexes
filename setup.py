from setuptools import setup

setup(
    name="TSP-SC",
    version="1.0",
    scripts=[
        "scripts/prepare_mesh_normals.py",
        "scripts/evaluate_normals.py",
        "scripts/prepare_citation_simplices.py",
        "scripts/evaluate_citations.py",
        "scripts/evaluate_graph_classification.py",
        "scripts/prepare_graph_classification.py",
    ],
)
