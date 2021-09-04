from tsp_sc.citations.utils.citations import *
from tsp_sc.common.io import load_config
import argparse


def process_papers(corpus, num_papers_in_millions):
    """
    First pass through all the papers.
    """

    papers = dict()
    edges = list()

    for file in tqdm(sorted(glob.glob(corpus))[:num_papers_in_millions]):
        with gzip.open(file, "rt") as file:

            for line in file:

                paper = json.loads(line)

                try:
                    year = paper["year"]
                except KeyError:
                    year = 0

                missing_authors = 0
                for author in paper["authors"]:
                    n_ids = len(author["ids"])
                    if n_ids == 0:
                        missing_authors += 1
                    elif n_ids == 1:
                        edges.append((paper["id"], author["ids"][0]))
                    else:
                        raise ValueError("No author should have multiple IDs.")

                # papers[paper_id_i] = [ [citation_1, .. , citation_n], num_out_citations, year_publication, num_missing_authors ]
                papers[paper["id"]] = (
                    paper["inCitations"],
                    len(paper["outCitations"]),
                    year,
                    missing_authors,
                )

    print(f"processed {len(papers):,} papers")
    print(f"collected {len(edges):,} paper-author links")

    return papers, edges


def count_citations(papers, years):
    """
    Second pass to check the publication year of the referencing papers.
    """

    years = np.array(years)

    # for each paper
    for pid, attributes in tqdm(papers.items()):

        missing_citations = 0
        counts = np.zeros_like(years)

        # for each in_citation
        for citation in attributes[0]:

            try:
                year = papers[citation][2]
            except KeyError:
                missing_citations += 1  # unknown paper
                continue

            if year != 0:
                counts += year < years
            else:
                missing_citations += 1  # unknown year

        papers[pid] = tuple(counts) + attributes[1:] + (missing_citations,)


def fastdump(obj, file):
    p = pickle.Pickler(file)
    p.fast = True
    p.dump(obj)


def save(papers, edges, pickle_file):
    data = dict(edges=edges, papers=papers)
    with open(pickle_file, "wb+") as f:
        fastdump(data, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    cli_args = parser.parse_args()

    config = load_config(cli_args.config)
    paths = config["paths"]

    print(f"Processing papers")
    num_papers_in_millions = 5
    papers, edges = process_papers(paths["corpus"], num_papers_in_millions)

    print(f"Counting citations")
    count_citations(papers, range(1994, 2024, 5))

    save(papers, edges, paths["paper_author_graph_full"])
