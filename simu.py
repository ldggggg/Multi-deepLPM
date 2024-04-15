import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.feature_extraction.text import CountVectorizer

# Parameters
n_papers = 100  # Total number of papers
n_keywords = 50  # Distinct keywords to choose from
n_authors = 50  # Distinct authors
n_conferences = 3  # Distinct conferences
academic_fields = ['Computer Science', 'Biology', 'Physics']  # Academic fields
max_keywords_per_paper = 5  # Max number of keywords per paper
max_authors_per_paper = 3  # Max number of authors per paper

# Simulating data
np.random.seed(42)  # For reproducibility

# Generate random keywords
keywords = [f"Keyword_{i}" for i in range(n_keywords)]

# Generate papers
papers = pd.DataFrame({
    'PaperID': range(n_papers),
    'Keywords': ['; '.join(np.random.choice(keywords, np.random.randint(1, max_keywords_per_paper), replace=False)) for _ in range(n_papers)],
    'Field': np.random.choice(academic_fields, n_papers),
    'Authors': ['; '.join(np.random.choice(range(n_authors), np.random.randint(1, max_authors_per_paper), replace=False).astype(str)) for _ in range(n_papers)],
    'Conference': np.random.choice(range(n_conferences), n_papers)
})

# Create bag-of-words representation for Keywords
vectorizer = CountVectorizer(tokenizer=lambda x: x.split('; '))
keywords_bow = vectorizer.fit_transform(papers['Keywords'])

# Convert to DataFrame to visualize
keywords_bow_df = pd.DataFrame(keywords_bow.toarray(), columns=vectorizer.get_feature_names_out())

# Simulate relationships
# Co-authorship
co_authorship = [(p1, p2) for p1, p2 in combinations(papers['PaperID'], 2)
                 if set(papers.loc[p1, 'Authors'].split('; ')).intersection(papers.loc[p2, 'Authors'].split('; '))]

# Co-conference
co_conference = [(p1, p2) for p1, p2 in combinations(papers['PaperID'], 2)
                 if papers.loc[p1, 'Conference'] == papers.loc[p2, 'Conference']]

# Co-keyword usage (simplified for demonstration, considering direct match only)
co_keyword = [(p1, p2) for p1, p2 in combinations(papers['PaperID'], 2)
              if set(papers.loc[p1, 'Keywords'].split('; ')).intersection(papers.loc[p2, 'Keywords'].split('; '))]

# Summary
len(co_authorship), len(co_conference), len(co_keyword), papers.head(), keywords_bow_df.head()


# Initialize adjacency matrices with zeros
n = len(papers)  # Number of papers
adj_matrix_co_authorship = np.zeros((n, n), dtype=int)
adj_matrix_co_conference = np.zeros((n, n), dtype=int)
adj_matrix_co_keyword = np.zeros((n, n), dtype=int)

# Fill in the adjacency matrices based on the relationships

# Co-authorship
for p1, p2 in co_authorship:
    adj_matrix_co_authorship[p1, p2] = 1
    adj_matrix_co_authorship[p2, p1] = 1  # Undirected graph

# Co-conference
for p1, p2 in co_conference:
    adj_matrix_co_conference[p1, p2] = 1
    adj_matrix_co_conference[p2, p1] = 1  # Undirected graph

# Co-keyword
for p1, p2 in co_keyword:
    adj_matrix_co_keyword[p1, p2] = 1
    adj_matrix_co_keyword[p2, p1] = 1  # Undirected graph

# Let's check the sum of all connections in each matrix to verify
sum_co_authorship = np.sum(adj_matrix_co_authorship)
sum_co_conference = np.sum(adj_matrix_co_conference)
sum_co_keyword = np.sum(adj_matrix_co_keyword)

print(sum_co_authorship, sum_co_conference, sum_co_keyword)

feature_matrix = keywords_bow.todense()