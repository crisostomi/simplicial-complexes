from setuptools import setup

setup(
    name="TSP-SC",
    version="1.0",
    scripts=[
        "scripts/prepare_mesh_normals.py",
        "scripts/evaluate_normals.py",
        "scripts/evaluate_citations.py",
        "scripts/evaluate_graph_classification.py",
        "scripts/prepare_graph_classification.py",
        "scripts/citations/create_bipartite_graph.py",
        "scripts/citations/process_papers.py",
        "scripts/citations/project_graph.py",
        "scripts/citations/downsample.py",
        "scripts/citations/create_complex.py",
        "scripts/citations/prepare_signals.py",
        "scripts/citations/create_synth_signal.py",
    ],
)
