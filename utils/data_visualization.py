import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
import json

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def main():
    warnings.filterwarnings('ignore')

    # Load the dataset - prioritize chunked data for GitHub compatibility
    import os
    import json

    df = None

    # First try to load from chunks (GitHub-compatible)
    if os.path.exists('data/metadata.json'):
        print("Loading data from chunks for processing...")
        try:
            data_list = []
            with open('data/metadata.json', 'r') as f:
                metadata = json.load(f)

            for chunk_file in metadata['chunk_files']:
                chunk_path = os.path.join('data', chunk_file)
                if os.path.exists(chunk_path):
                    with open(chunk_path, 'r') as f:
                        chunk_data = json.load(f)
                        if isinstance(chunk_data, list):
                            data_list.extend(chunk_data)
                        else:
                            data_list.append(chunk_data)

            if data_list:
                df = pd.DataFrame(data_list)
                print(f"Successfully loaded {len(df)} records from {len(metadata['chunk_files'])} chunks")
        except Exception as e:
            print(f"Error loading from chunks: {e}")

    # Fallback to merged file
    if df is None and os.path.exists('hseg_final_dataset.json'):
        print("Loading from merged JSON file...")
        df = pd.read_json('hseg_final_dataset.json')
    elif df is None and os.path.exists('hseg_final_dataset.csv'):
        print("Loading from CSV file...")
        df = pd.read_csv('hseg_final_dataset.csv')

    if df is None or df.empty:
        print("Warning: No data files found or data is empty")
        return {}

    # Define survey sections for analysis
    sections = {
        'Power Abuse & Suppression': ['q1', 'q2', 'q3', 'q4'],
        'Discrimination & Exclusion': ['q5', 'q6', 'q7'],
        'Manipulative Work Culture': ['q8', 'q9', 'q10'],
        'Failure of Accountability': ['q11', 'q12', 'q13', 'q14'],
        'Mental Health Harm': ['q15', 'q16', 'q17', 'q18'],
        'Erosion of Voice & Autonomy': ['q19', 'q20', 'q21', 'q22']
    }

    # Calculate section scores
    for section_name, questions in sections.items():
        df[f'{section_name}_score'] = df[questions].mean(axis=1)

    # Overall culture score
    all_questions = [q for section_questions in sections.values() for q in section_questions]
    df['overall_culture_score'] = df[all_questions].mean(axis=1)

    # Create section score columns list for easy reference
    section_score_cols = [f'{section}_score' for section in sections.keys()]

    # Calculate mean scores by domain
    domain_means = df.groupby('domain')[section_score_cols].mean()

    # Calculate section correlations
    section_corr = df[section_score_cols].corr()

    # Create correlation matrix for all quantitative questions
    corr_matrix = df[all_questions].corr()

    # Create correlation dataframe
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_pairs.append({
                'Question 1': corr_matrix.columns[i],
                'Question 2': corr_matrix.columns[j],
                'Correlation': corr_matrix.iloc[i, j]
            })
    corr_df = pd.DataFrame(corr_pairs).sort_values('Correlation', ascending=False)

    # Calculate organization means for section scores
    min_responses = 50
    org_response_counts = df['organization_name'].value_counts()
    large_orgs = org_response_counts[org_response_counts >= min_responses].index
    org_means = df[df['organization_name'].isin(large_orgs)].groupby('organization_name')[section_score_cols].mean()

    # At the end of the main function, create the dictionary to be returned
    processed_data = {
        "overview": {
            "total_responses": len(df),
            "num_organizations": df['organization_name'].nunique(),
            "num_domains": df['domain'].nunique(),
            "num_departments": df['department'].nunique(),
            "overall_culture_score": df['overall_culture_score'].mean()
        },
        "domains": domain_means.to_dict('index'),
        "sections": section_corr.to_dict('index'),
        "correlations": corr_df.to_dict('records'),
        "organizations": org_means.to_dict('index'),
        "demographics": [],
        "metadata": {
            "timestamp": pd.Timestamp.now().isoformat()
        }
    }

    with open('processed_hseg_data.json', 'w') as f:
        json.dump(processed_data, f, indent=2, cls=NpEncoder)

    return processed_data

if __name__ == '__main__':
    main()