import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.feature_extraction.text import CountVectorizer


def generate_matrices(n_papers=100, n_keywords=50, n_authors=50, n_conferences=3, max_keywords_per_paper=5,
                      max_authors_per_paper=3):
    # Seed for reproducibility
    np.random.seed(42)

    # Generate random keywords and papers
    keywords = [f"Keyword_{i}" for i in range(n_keywords)]
    papers = pd.DataFrame({
        'PaperID': range(n_papers),
        'Keywords': ['; '.join(np.random.choice(keywords, np.random.randint(1, max_keywords_per_paper), replace=False))
                     for _ in range(n_papers)],
        'Field': np.random.choice(['Computer Science', 'Biology', 'Physics'], n_papers),
        'Authors': ['; '.join(
            np.random.choice(range(n_authors), np.random.randint(1, max_authors_per_paper), replace=False).astype(str))
                    for _ in range(n_papers)],
        'Conference': np.random.choice(range(n_conferences), n_papers)
    })

    # Create bag-of-words representation for Keywords
    vectorizer = CountVectorizer(tokenizer=lambda x: x.split('; '))
    keywords_bow = vectorizer.fit_transform(papers['Keywords'])

    # Convert to DataFrame to visualize
    keywords_bow_df = pd.DataFrame(keywords_bow.toarray(), columns=vectorizer.get_feature_names_out())

    feature_matrix = keywords_bow.todense()

    # Initialize adjacency matrices for each relationship type
    adj_matrix_co_authorship = np.zeros((n_papers, n_papers), dtype=int)
    adj_matrix_co_conference = np.zeros((n_papers, n_papers), dtype=int)
    adj_matrix_co_keyword = np.zeros((n_papers, n_papers), dtype=int)

    # Edge covariate matrices for counts
    covariate_matrix_authors = np.zeros((n_papers, n_papers), dtype=int)
    covariate_matrix_keywords = np.zeros((n_papers, n_papers), dtype=int)

    # Populate adjacency and covariate matrices
    for (p1, paper1), (p2, paper2) in combinations(papers.iterrows(), 2):
        shared_authors = len(set(paper1['Authors'].split('; ')).intersection(paper2['Authors'].split('; ')))
        shared_keywords = len(set(paper1['Keywords'].split('; ')).intersection(paper2['Keywords'].split('; ')))

        if shared_authors > 0:
            adj_matrix_co_authorship[p1, p2] = adj_matrix_co_authorship[p2, p1] = 1
            covariate_matrix_authors[p1, p2] = covariate_matrix_authors[p2, p1] = shared_authors

        if paper1['Conference'] == paper2['Conference']:
            adj_matrix_co_conference[p1, p2] = adj_matrix_co_conference[p2, p1] = 1

        if shared_keywords > 0:
            adj_matrix_co_keyword[p1, p2] = adj_matrix_co_keyword[p2, p1] = 1
            covariate_matrix_keywords[p1, p2] = covariate_matrix_keywords[p2, p1] = shared_keywords

    # Lists to return
    adjacency_matrices = [adj_matrix_co_authorship, adj_matrix_co_conference, adj_matrix_co_keyword]
    covariate_matrices = [covariate_matrix_authors, None, covariate_matrix_keywords]

    # Let's check the sum of all connections in each matrix to verify
    sum_co_authorship = np.sum(adj_matrix_co_authorship)
    sum_co_conference = np.sum(adj_matrix_co_conference)
    sum_co_keyword = np.sum(adj_matrix_co_keyword)
    print(sum_co_authorship, sum_co_conference, sum_co_keyword)

    return adjacency_matrices, covariate_matrices, feature_matrix


# # Example of using the function
# adj_matrices, cov_matrices, feat_matrix = generate_matrices()
# print('Generated adjacency and covariate matrices.')