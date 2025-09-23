#!/usr/bin/env python
# coding: utf-8

# # HSEG Survey Data: Comprehensive Analysis & Visualization
# 
# ## Executive Summary
# 
# This notebook provides a comprehensive analysis of the HSEG (Healthcare, University, Business) workplace culture intelligence survey dataset containing 49,550 responses across 33 questions. The analysis is structured into four key areas:
# 
# 1. **Intra-Section Analysis**: Deep dive into the 6 cultural dimensions
# 2. **Intra-Business Analysis**: Organization-specific insights
# 3. **Domain Analysis**: Cross-domain comparisons (Healthcare vs University vs Business)
# 4. **Cross-Industry Visualization**: Industry-wide patterns and benchmarking
# 
# ### Survey Structure Overview
# - **Power Abuse & Suppression**: Q1-Q4 (4 questions)
# - **Discrimination & Exclusion**: Q5-Q7 (3 questions)
# - **Manipulative Work Culture**: Q8-Q10 (3 questions)
# - **Failure of Accountability**: Q11-Q14 (4 questions)
# - **Mental Health Harm**: Q15-Q18 (4 questions)
# - **Erosion of Voice & Autonomy**: Q19-Q22 (4 questions)
# - **Strategic Open-Ended**: Q23-Q25 (3 text questions)
# - **Demographics**: Q26-Q33 (8 questions)

# ## 1. Setup and Data Loading

# In[2]:


# Import required libraries
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
warnings.filterwarnings('ignore')

# Set up plotting parameters
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

# Plotly configuration
import plotly.io as pio
pio.templates.default = "plotly_white"

print("Libraries imported successfully!")


# In[3]:


# SOLUTION 1: Modified imports that work around the issue
print("Attempting to import libraries with fallback options...")

# Import required libraries (basic ones first)
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
import warnings
warnings.filterwarnings('ignore')

print("✅ Basic libraries imported successfully!")

# Try importing scikit-learn components with error handling
sklearn_available = True
try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    print("✅ scikit-learn imported successfully!")
except ImportError as e:
    sklearn_available = False
    print(f"❌ scikit-learn import failed: {e}")
    print("⚠️  Will provide alternative solutions for ML functionality")

# Set up plotting parameters
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

# Plotly configuration
import plotly.io as pio
pio.templates.default = "plotly_white"

print(f"\n📊 Setup complete! scikit-learn available: {sklearn_available}")

# Alternative implementations if sklearn is not available
if not sklearn_available:
    print("\n🔧 Setting up alternative implementations...")
    
    # Simple PCA alternative using numpy
    def simple_pca(data, n_components=2):
        """Simple PCA implementation using numpy"""
        # Center the data
        data_centered = data - np.mean(data, axis=0)
        
        # Compute covariance matrix
        cov_matrix = np.cov(data_centered.T)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalues (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Project data
        transformed = data_centered @ eigenvectors[:, :n_components]
        
        return transformed, eigenvectors[:, :n_components], eigenvalues[:n_components]
    
    # Simple standardization
    def simple_standardize(data):
        """Simple standardization using numpy"""
        return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    
    # Simple K-means clustering
    def simple_kmeans(data, k=3, max_iters=100):
        """Simple K-means implementation"""
        n_samples, n_features = data.shape
        
        # Initialize centroids randomly
        centroids = data[np.random.choice(n_samples, k, replace=False)]
        
        for _ in range(max_iters):
            # Assign points to nearest centroid
            distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)
            
            # Update centroids
            new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
            
            # Check for convergence
            if np.allclose(centroids, new_centroids):
                break
                
            centroids = new_centroids
        
        return labels, centroids
    
    print("✅ Alternative implementations ready!")

print("\n" + "="*60)
if sklearn_available:
    print("🎉 ALL LIBRARIES IMPORTED SUCCESSFULLY!")
    print("You can use all advanced ML features including:")
    print("  • PCA and t-SNE for dimensionality reduction")
    print("  • StandardScaler for data normalization") 
    print("  • KMeans for clustering")
else:
    print("⚠️  SKLEARN UNAVAILABLE - USING ALTERNATIVES")
    print("Available alternative functions:")
    print("  • simple_pca() for basic PCA analysis")
    print("  • simple_standardize() for data scaling")
    print("  • simple_kmeans() for basic clustering")
    print("\nTo fix sklearn permanently, run in terminal:")
    print("  conda update --all")
    print("  conda install scikit-learn=1.3.0")
    print("  pip install --upgrade pyarrow")

print("="*60)


# In[2]:


# Load the dataset
df = pd.read_csv('data/hseg_final_dataset.csv')

# Display basic information about the dataset
print(f"Dataset shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst few rows:")
df.head()


# In[3]:


# Data Quality Assessment
print("=== DATA QUALITY ASSESSMENT ===")
print(f"Total Records: {len(df):,}")
print(f"\nMissing Values by Column:")
missing_data = df.isnull().sum()
missing_percent = (missing_data / len(df)) * 100
missing_summary = pd.DataFrame({
    'Missing Count': missing_data,
    'Missing Percentage': missing_percent
})
print(missing_summary[missing_summary['Missing Count'] > 0])

# Check data types
print(f"\nData Types:")
print(df.dtypes.value_counts())

# Domain distribution
print(f"\nDomain Distribution:")
print(df['domain'].value_counts())

# Basic statistics for quantitative questions
quantitative_cols = [f'q{i}' for i in range(1, 23)]  # Q1-Q22 are quantitative
print(f"\nQuantitative Questions Summary:")
df[quantitative_cols].describe()


# In[4]:


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
    df[f'{section_name}_sum'] = df[questions].sum(axis=1)

# Overall culture score
all_questions = [q for section_questions in sections.values() for q in section_questions]
df['overall_culture_score'] = df[all_questions].mean(axis=1)

# Create section score columns list for easy reference
section_score_cols = [f'{section}_score' for section in sections.keys()]

print("Section scores calculated successfully!")
print(f"\nSection Score Summary:")
df[section_score_cols + ['overall_culture_score']].describe()


# ## 2. INTRA-SECTION ANALYSIS
# 
# Deep dive into the 6 cultural dimensions, examining correlations, distributions, and patterns within each section.

# In[5]:


# 2.1 Section Score Distributions
fig = make_subplots(
    rows=2, cols=3,
    subplot_titles=list(sections.keys()),
    specs=[[{"secondary_y": False}]*3]*2
)

colors = px.colors.qualitative.Set3

for i, (section_name, _) in enumerate(sections.items()):
    row = i // 3 + 1
    col = i % 3 + 1
    
    score_col = f'{section_name}_score'
    
    fig.add_trace(
        go.Histogram(
            x=df[score_col],
            name=section_name,
            marker_color=colors[i],
            opacity=0.7,
            nbinsx=30
        ),
        row=row, col=col
    )

fig.update_layout(
    title_text="Section Score Distributions",
    showlegend=False,
    height=800
)

fig.show()


# In[6]:


# 2.2 Violin Plots for Section Score Distributions by Domain
fig = make_subplots(
    rows=2, cols=3,
    subplot_titles=list(sections.keys()),
    specs=[[{"type": "xy"}]*3]*2
)

domains = df['domain'].unique()
domain_colors = {'Healthcare': '#FF6B6B', 'University': '#4ECDC4', 'Business': '#45B7D1'}

for i, (section_name, _) in enumerate(sections.items()):
    row = i // 3 + 1
    col = i % 3 + 1
    
    score_col = f'{section_name}_score'
    
    for domain in domains:
        domain_data = df[df['domain'] == domain][score_col]
        
        fig.add_trace(
            go.Violin(
                y=domain_data,
                name=domain,
                side='positive',
                line_color=domain_colors[domain],
                fillcolor=domain_colors[domain],
                opacity=0.6,
                showlegend=(i == 0)  # Only show legend for first subplot
            ),
            row=row, col=col
        )

fig.update_layout(
    title_text="Section Score Distributions by Domain (Violin Plots)",
    height=800
)

fig.show()


# In[7]:


# 2.3 Correlation Heatmap - Overall Questions
plt.figure(figsize=(16, 12))

# Create correlation matrix for all quantitative questions
corr_matrix = df[all_questions].corr()

# Create a mask for the upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Generate heatmap
sns.heatmap(corr_matrix, 
            mask=mask,
            annot=True, 
            cmap='RdYlBu_r', 
            center=0,
            square=True,
            fmt='.2f',
            cbar_kws={"shrink": .8})

plt.title('Inter-Question Correlation Matrix', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Print highest correlations
print("Highest Correlations (excluding self-correlations):")
corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        corr_pairs.append({
            'Question 1': corr_matrix.columns[i],
            'Question 2': corr_matrix.columns[j],
            'Correlation': corr_matrix.iloc[i, j]
        })

corr_df = pd.DataFrame(corr_pairs).sort_values('Correlation', ascending=False)
print(corr_df.head(10))


# In[8]:


# 2.4 Section Correlation Heatmap
fig, ax = plt.subplots(figsize=(10, 8))

section_corr = df[section_score_cols].corr()

sns.heatmap(section_corr, 
            annot=True, 
            cmap='RdYlBu_r', 
            center=0,
            square=True,
            fmt='.3f',
            cbar_kws={"shrink": .8},
            ax=ax)

plt.title('Section Score Correlation Matrix', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Statistical significance testing for section correlations
print("Section Correlation Significance Testing:")
for i, section1 in enumerate(section_score_cols):
    for j, section2 in enumerate(section_score_cols[i+1:], i+1):
        corr, p_value = stats.pearsonr(df[section1], df[section2])
        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        print(f"{section1} vs {section2}: r={corr:.3f}, p={p_value:.2e} {significance}")


# In[9]:


# 2.5 Radar Charts for Section Profiles by Domain
# Calculate mean scores by domain
domain_means = df.groupby('domain')[section_score_cols].mean()

# Create radar chart
fig = go.Figure()

# Shorten section names for better display
short_names = [
    'Power Abuse',
    'Discrimination', 
    'Manipulative Culture',
    'Failed Accountability',
    'Mental Health Harm',
    'Voice Erosion'
]

for domain in domains:
    values = domain_means.loc[domain].values.tolist()
    values += [values[0]]  # Close the radar chart
    
    categories = short_names + [short_names[0]]  # Close the categories
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=domain,
        line_color=domain_colors[domain]
    ))

fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[1, 4]  # Assuming 1-4 scale
        )),
    showlegend=True,
    title="Section Profile Comparison by Domain",
    font_size=12
)

fig.show()

# Print numerical values
print("Mean Section Scores by Domain:")
print(domain_means.round(3))


# In[10]:


# 2.6 Hierarchical Clustering of Questions within Sections
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.flatten()

for i, (section_name, questions) in enumerate(sections.items()):
    if len(questions) > 2:  # Only cluster if more than 2 questions
        section_data = df[questions].corr()
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(1 - section_data, method='ward')
        
        # Create dendrogram
        dendrogram(linkage_matrix, 
                  labels=questions,
                  ax=axes[i],
                  orientation='top')
        
        axes[i].set_title(f'{section_name}\nQuestion Clustering', fontweight='bold')
        axes[i].tick_params(axis='x', rotation=45)
    else:
        axes[i].text(0.5, 0.5, f'{section_name}\n(Too few questions for clustering)', 
                    ha='center', va='center', transform=axes[i].transAxes,
                    fontsize=12, fontweight='bold')
        axes[i].set_xticks([])
        axes[i].set_yticks([])

plt.tight_layout()
plt.show()


# ## 3. INTRA-BUSINESS ANALYSIS
# 
# Organization-specific dashboards and analysis within individual companies.

# In[11]:


# 3.1 Top Organizations by Response Count
org_counts = df['organization_name'].value_counts().head(20)
print("Top 20 Organizations by Response Count:")
print(org_counts)

# Visualize top organizations
fig = px.bar(x=org_counts.values, 
             y=org_counts.index,
             orientation='h',
             title="Top 20 Organizations by Response Count",
             labels={'x': 'Number of Responses', 'y': 'Organization'},
             color=org_counts.values,
             color_continuous_scale='viridis')

fig.update_layout(height=600, showlegend=False)
fig.show()


# In[12]:


# 3.2 Organizational Benchmarking Matrix
# Focus on organizations with at least 50 responses for statistical reliability
min_responses = 50
org_response_counts = df['organization_name'].value_counts()
large_orgs = org_response_counts[org_response_counts >= min_responses].index

print(f"Organizations with {min_responses}+ responses: {len(large_orgs)}")

# Calculate organization means for section scores
org_means = df[df['organization_name'].isin(large_orgs)].groupby('organization_name')[section_score_cols].mean()

# Create heatmap
plt.figure(figsize=(14, 10))

# Transpose for better visualization
org_means_display = org_means.T
org_means_display.index = short_names

sns.heatmap(org_means_display, 
            annot=True, 
            cmap='RdYlBu_r', 
            center=2.5,  # Assuming 1-4 scale
            fmt='.2f',
            cbar_kws={"shrink": .8})

plt.title(f'Organizational Benchmarking Matrix\n(Organizations with {min_responses}+ responses)', 
          fontsize=16, fontweight='bold')
plt.xlabel('Organization', fontweight='bold')
plt.ylabel('Cultural Dimension', fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Print summary statistics
print("\nOrganizational Benchmarking Summary:")
print(org_means.describe().round(3))


# In[13]:


# 3.3 Department/Position Level Analysis within Organizations
# Focus on a specific large organization for detailed analysis
target_org = org_response_counts.index[0]  # Largest organization
org_data = df[df['organization_name'] == target_org].copy()

print(f"Detailed Analysis: {target_org}")
print(f"Total responses: {len(org_data)}")

# Department breakdown
dept_counts = org_data['department'].value_counts()
print(f"\nDepartment distribution:")
print(dept_counts)

# Position level breakdown
position_counts = org_data['position_level'].value_counts()
print(f"\nPosition level distribution:")
print(position_counts)

# Create subplots for department and position analysis
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        'Department Distribution',
        'Position Level Distribution', 
        'Section Scores by Department',
        'Section Scores by Position Level'
    ],
    specs=[[{"type": "pie"}, {"type": "pie"}],
           [{"type": "xy"}, {"type": "xy"}]]
)

# Department pie chart
fig.add_trace(
    go.Pie(labels=dept_counts.index, values=dept_counts.values, name="Departments"),
    row=1, col=1
)

# Position level pie chart
fig.add_trace(
    go.Pie(labels=position_counts.index, values=position_counts.values, name="Positions"),
    row=1, col=2
)

# Section scores by department (box plot)
if len(dept_counts) > 1:
    dept_means = org_data.groupby('department')['overall_culture_score'].mean().sort_values(ascending=False)
    
    for dept in dept_means.index:
        dept_scores = org_data[org_data['department'] == dept]['overall_culture_score']
        fig.add_trace(
            go.Box(y=dept_scores, name=dept, showlegend=False),
            row=2, col=1
        )

# Section scores by position level (box plot)
if len(position_counts) > 1:
    pos_means = org_data.groupby('position_level')['overall_culture_score'].mean().sort_values(ascending=False)
    
    for pos in pos_means.index:
        pos_scores = org_data[org_data['position_level'] == pos]['overall_culture_score']
        fig.add_trace(
            go.Box(y=pos_scores, name=pos, showlegend=False),
            row=2, col=2
        )

fig.update_layout(height=800, title_text=f"Organizational Deep Dive: {target_org}")
fig.show()


# In[14]:


# 3.4 Tenure vs Culture Score Relationships
# Analyze how tenure affects culture scores across organizations

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Overall tenure vs culture score
tenure_order = ['<1_year', '1-3_years', '3-7_years', '7-15_years', '15+_years']
tenure_available = [t for t in tenure_order if t in df['tenure_range'].unique()]

# Box plot: Tenure vs Overall Culture Score
sns.boxplot(data=df, x='tenure_range', y='overall_culture_score', 
           order=tenure_available, ax=axes[0,0])
axes[0,0].set_title('Overall Culture Score by Tenure', fontweight='bold')
axes[0,0].tick_params(axis='x', rotation=45)

# Violin plot: Tenure vs Overall Culture Score by Domain
sns.violinplot(data=df, x='tenure_range', y='overall_culture_score', 
              hue='domain', order=tenure_available, ax=axes[0,1])
axes[0,1].set_title('Culture Score by Tenure and Domain', fontweight='bold')
axes[0,1].tick_params(axis='x', rotation=45)

# Heatmap: Mean scores by tenure and domain
tenure_domain_means = df.groupby(['domain', 'tenure_range'])['overall_culture_score'].mean().unstack()
tenure_domain_means = tenure_domain_means.reindex(columns=tenure_available)

sns.heatmap(tenure_domain_means, annot=True, fmt='.2f', 
           cmap='RdYlBu_r', center=2.5, ax=axes[1,0])
axes[1,0].set_title('Mean Culture Score: Domain vs Tenure', fontweight='bold')

# Line plot: Tenure progression
tenure_means = df.groupby('tenure_range')['overall_culture_score'].mean().reindex(tenure_available)
axes[1,1].plot(range(len(tenure_means)), tenure_means.values, 'o-', linewidth=2, markersize=8)
axes[1,1].set_xticks(range(len(tenure_means)))
axes[1,1].set_xticklabels(tenure_means.index, rotation=45)
axes[1,1].set_title('Culture Score Trend by Tenure', fontweight='bold')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Statistical analysis
print("Tenure vs Culture Score Analysis:")
from scipy.stats import kruskal
tenure_groups = [df[df['tenure_range'] == t]['overall_culture_score'].dropna() 
                for t in tenure_available]
h_stat, p_value = kruskal(*tenure_groups)
print(f"Kruskal-Wallis test: H={h_stat:.3f}, p={p_value:.3e}")

print("\nMean scores by tenure:")
print(tenure_means.round(3))


# ## 4. DOMAIN ANALYSIS
# 
# Cross-domain comparisons between Healthcare, University, and Business sectors.

# In[15]:


# 4.1 Cross-Domain Comparison Matrices
print("=== DOMAIN ANALYSIS ===")
print(f"Domain distribution:")
domain_dist = df['domain'].value_counts()
print(domain_dist)
print(f"\nDomain percentages:")
print((domain_dist / len(df) * 100).round(2))

# Statistical comparison between domains
from scipy.stats import f_oneway

print("\n=== STATISTICAL COMPARISON BETWEEN DOMAINS ===")
for section_col in section_score_cols:
    healthcare_scores = df[df['domain'] == 'Healthcare'][section_col].dropna()
    university_scores = df[df['domain'] == 'University'][section_col].dropna() 
    business_scores = df[df['domain'] == 'Business'][section_col].dropna()
    
    f_stat, p_value = f_oneway(healthcare_scores, university_scores, business_scores)
    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
    
    print(f"{section_col}: F={f_stat:.3f}, p={p_value:.3e} {significance}")
    
    # Post-hoc pairwise comparisons
    from scipy.stats import ttest_ind
    
    # Healthcare vs University
    t_stat, p_val = ttest_ind(healthcare_scores, university_scores)
    print(f"  Healthcare vs University: t={t_stat:.3f}, p={p_val:.3e}")
    
    # Healthcare vs Business  
    t_stat, p_val = ttest_ind(healthcare_scores, business_scores)
    print(f"  Healthcare vs Business: t={t_stat:.3f}, p={p_val:.3e}")
    
    # University vs Business
    t_stat, p_val = ttest_ind(university_scores, business_scores)
    print(f"  University vs Business: t={t_stat:.3f}, p={p_val:.3e}")
    print()


# In[16]:


# 4.2 Domain-Specific Cultural Patterns Visualization
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        'Section Scores by Domain (Box Plots)',
        'Domain Score Distributions',
        'Section Score Correlations by Domain', 
        'Overall Culture Score by Domain'
    ],
    specs=[[{"type": "xy"}, {"type": "xy"}],
           [{"type": "xy"}, {"type": "xy"}]]
)

# Box plots for each section by domain
for i, (section_name, _) in enumerate(sections.items()):
    score_col = f'{section_name}_score'
    
    for domain in domains:
        domain_data = df[df['domain'] == domain][score_col]
        
        fig.add_trace(
            go.Box(
                y=domain_data,
                name=f"{domain}",
                marker_color=domain_colors[domain],
                showlegend=(i == 0),
                offsetgroup=domain,
                x=[short_names[i]] * len(domain_data)
            ),
            row=1, col=1
        )

# Overall culture score by domain
for domain in domains:
    domain_data = df[df['domain'] == domain]['overall_culture_score']
    
    fig.add_trace(
        go.Histogram(
            x=domain_data,
            name=domain,
            marker_color=domain_colors[domain],
            opacity=0.7,
            showlegend=False
        ),
        row=2, col=2
    )

fig.update_layout(height=1000, title_text="Domain Analysis Dashboard")
fig.show()


# In[17]:


# 4.3 Sankey Diagram: Domain → Organization → Department Flow
# Create flow data for Sankey diagram
sankey_data = df.groupby(['domain', 'organization_name', 'department']).size().reset_index(name='count')

# Filter for better visualization (top organizations and departments)
top_orgs = df['organization_name'].value_counts().head(10).index
sankey_filtered = sankey_data[sankey_data['organization_name'].isin(top_orgs)]

# Prepare data for Sankey
all_nodes = list(sankey_filtered['domain'].unique()) + \
           list(sankey_filtered['organization_name'].unique()) + \
           list(sankey_filtered['department'].unique())

node_dict = {node: i for i, node in enumerate(all_nodes)}

# Create source, target, and value arrays
source = []
target = []
value = []
labels = all_nodes

# Domain to Organization flows
domain_org_flows = sankey_filtered.groupby(['domain', 'organization_name'])['count'].sum().reset_index()
for _, row in domain_org_flows.iterrows():
    source.append(node_dict[row['domain']])
    target.append(node_dict[row['organization_name']])
    value.append(row['count'])

# Organization to Department flows
org_dept_flows = sankey_filtered.groupby(['organization_name', 'department'])['count'].sum().reset_index()
for _, row in org_dept_flows.iterrows():
    source.append(node_dict[row['organization_name']])
    target.append(node_dict[row['department']])
    value.append(row['count'])

# Create Sankey diagram
fig = go.Figure(data=[go.Sankey(
    node = dict(
      pad = 15,
      thickness = 20,
      line = dict(color = "black", width = 0.5),
      label = labels,
      color = "blue"
    ),
    link = dict(
      source = source,
      target = target,
      value = value
  ))])

fig.update_layout(
    title_text="Flow Analysis: Domain → Organization → Department", 
    font_size=10,
    height=600
)
fig.show()

print(f"Sankey diagram shows flow of {sankey_filtered['count'].sum():,} responses")
print(f"Across {len(sankey_filtered['domain'].unique())} domains, {len(sankey_filtered['organization_name'].unique())} organizations, and {len(sankey_filtered['department'].unique())} departments")


# In[18]:


# 4.4 Parallel Coordinates for Multi-dimensional Domain Comparison
# Prepare data for parallel coordinates
parallel_data = df[['domain'] + section_score_cols + ['overall_culture_score']].copy()

# Sample data for better performance (if needed)
if len(parallel_data) > 5000:
    parallel_sample = parallel_data.groupby('domain').apply(
        lambda x: x.sample(min(len(x), 1500), random_state=42)
    ).reset_index(drop=True)
else:
    parallel_sample = parallel_data.copy()

# Create parallel coordinates plot
fig = px.parallel_coordinates(
    parallel_sample,
    color='domain',
    dimensions=section_score_cols,
    color_discrete_map=domain_colors,
    title="Parallel Coordinates: Cultural Dimensions by Domain"
)

fig.update_layout(height=600)
fig.show()

print(f"Parallel coordinates plot shows {len(parallel_sample):,} sampled responses")
print(f"Original dataset: {len(df):,} responses")


# In[21]:


# 4.4 Parallel Coordinates for Multi-dimensional Domain Comparison

# First, let's debug the data structure
print("Data structure debugging:")
print(f"DataFrame shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Section score columns: {section_score_cols}")
print(f"Domain unique values: {df['domain'].unique()}")
print(f"Data types:\n{df[['domain'] + section_score_cols].dtypes}")

# Check for any missing values or issues
print(f"\nMissing values:")
print(df[['domain'] + section_score_cols].isnull().sum())

# Prepare data for parallel coordinates
parallel_data = df[['domain'] + section_score_cols + ['overall_culture_score']].copy()

# Clean the data
parallel_data = parallel_data.dropna()  # Remove any rows with missing values

# Ensure numeric columns are properly typed
for col in section_score_cols:
    parallel_data[col] = pd.to_numeric(parallel_data[col], errors='coerce')

# Remove any rows that became NaN after conversion
parallel_data = parallel_data.dropna()

print(f"\nCleaned data shape: {parallel_data.shape}")

# Sample data for better performance (if needed)
if len(parallel_data) > 5000:
    parallel_sample = parallel_data.groupby('domain').apply(
        lambda x: x.sample(min(len(x), 1500), random_state=42)
    ).reset_index(drop=True)
else:
    parallel_sample = parallel_data.copy()

print(f"Sample data shape: {parallel_sample.shape}")

# Method 1: Basic parallel coordinates (most reliable)
try:
    fig = px.parallel_coordinates(
        parallel_sample,
        color='domain',
        dimensions=section_score_cols,
        title="Parallel Coordinates: Cultural Dimensions by Domain"
    )
    fig.update_layout(height=600)
    fig.show()
    print("✓ Basic parallel coordinates plot created successfully")
except Exception as e:
    print(f"✗ Basic method failed: {e}")

# Method 2: With color mapping (if domain_colors is defined)
try:
    if 'domain_colors' in globals():
        fig = px.parallel_coordinates(
            parallel_sample,
            color='domain',
            dimensions=section_score_cols,
            color_discrete_map=domain_colors,
            title="Parallel Coordinates: Cultural Dimensions by Domain (with colors)"
        )
        fig.update_layout(height=600)
        fig.show()
        print("✓ Color-mapped parallel coordinates plot created successfully")
    else:
        print("✗ domain_colors not defined, skipping color-mapped version")
except Exception as e:
    print(f"✗ Color-mapped method failed: {e}")

# Method 3: Alternative approach using go.Parcoords
try:
    import plotly.graph_objects as go
    
    # Create dimension list for go.Parcoords
    dimensions = []
    for i, col in enumerate(section_score_cols):
        dimensions.append(dict(
            range=[parallel_sample[col].min(), parallel_sample[col].max()],
            label=col.replace('_score', '').replace('_', ' ').title(),
            values=parallel_sample[col]
        ))
    
    # Create color mapping for domains
    domain_list = parallel_sample['domain'].unique()
    color_map = {domain: i for i, domain in enumerate(domain_list)}
    colors = parallel_sample['domain'].map(color_map)
    
    fig = go.Figure(data=
        go.Parcoords(
            line=dict(color=colors,
                     colorscale='viridis',
                     showscale=True,
                     cmax=len(domain_list)-1,
                     cmin=0),
            dimensions=dimensions
        )
    )
    
    fig.update_layout(
        title="Parallel Coordinates: Cultural Dimensions by Domain (Alternative)",
        height=600
    )
    fig.show()
    print("✓ Alternative go.Parcoords plot created successfully")
except Exception as e:
    print(f"✗ Alternative method failed: {e}")

print(f"\nParallel coordinates analysis complete")
print(f"Plotted {len(parallel_sample):,} sampled responses")
print(f"Original dataset: {len(df):,} responses")


# 

# ## 5. CROSS-INDUSTRY VISUALIZATION
# 
# Industry-wide benchmarking, clustering analysis, and network visualizations.

# In[22]:


# 5.1 Principal Component Analysis (PCA)
print("=== PRINCIPAL COMPONENT ANALYSIS ===")

# Prepare data for PCA
pca_data = df[section_score_cols].dropna()
scaler = StandardScaler()
pca_scaled = scaler.fit_transform(pca_data)

# Perform PCA
pca = PCA()
pca_result = pca.fit_transform(pca_scaled)

# Calculate explained variance
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

print(f"Explained variance by component:")
for i, var in enumerate(explained_variance_ratio):
    print(f"PC{i+1}: {var:.3f} ({var*100:.1f}%)")

print(f"\nCumulative explained variance:")
for i, cum_var in enumerate(cumulative_variance):
    print(f"PC1-PC{i+1}: {cum_var:.3f} ({cum_var*100:.1f}%)")

# Visualize PCA results
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        'Explained Variance by Component',
        'Cumulative Explained Variance',
        'PCA Scatter Plot (PC1 vs PC2)',
        'Component Loadings'
    ]
)

# Scree plot
fig.add_trace(
    go.Bar(
        x=[f'PC{i+1}' for i in range(len(explained_variance_ratio))],
        y=explained_variance_ratio,
        name='Explained Variance'
    ),
    row=1, col=1
)

# Cumulative variance plot
fig.add_trace(
    go.Scatter(
        x=[f'PC{i+1}' for i in range(len(cumulative_variance))],
        y=cumulative_variance,
        mode='lines+markers',
        name='Cumulative Variance'
    ),
    row=1, col=2
)

# PCA scatter plot by domain
domain_indices = df[section_score_cols].dropna().index
domain_labels = df.loc[domain_indices, 'domain']

for domain in domains:
    domain_mask = domain_labels == domain
    fig.add_trace(
        go.Scatter(
            x=pca_result[domain_mask, 0],
            y=pca_result[domain_mask, 1],
            mode='markers',
            name=domain,
            marker=dict(color=domain_colors[domain], size=4, opacity=0.6)
        ),
        row=2, col=1
    )

# Component loadings
loadings = pca.components_[:2].T  # First two components
feature_names = [name.replace(' & ', '\n&\n') for name in short_names]

fig.add_trace(
    go.Scatter(
        x=loadings[:, 0],
        y=loadings[:, 1],
        mode='markers+text',
        text=feature_names,
        textposition='middle right',
        marker=dict(size=10, color='red'),
        name='Loadings'
    ),
    row=2, col=2
)

# Add arrows for loadings
for i, (x, y) in enumerate(loadings):
    fig.add_shape(
        type="line",
        x0=0, y0=0, x1=x, y1=y,
        line=dict(color="red", width=2),
        row=2, col=2
    )

fig.update_layout(height=1000, title_text="Principal Component Analysis")
fig.show()


# In[23]:


# 5.2 t-SNE Visualization for Non-linear Dimensionality Reduction
print("=== t-SNE ANALYSIS ===")

# Sample data for t-SNE (computationally intensive)
tsne_sample_size = min(5000, len(pca_data))
sample_indices = np.random.choice(len(pca_data), tsne_sample_size, replace=False)
tsne_data = pca_scaled[sample_indices]
tsne_domains = domain_labels.iloc[sample_indices]

# Perform t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_result = tsne.fit_transform(tsne_data)

print(f"t-SNE performed on {tsne_sample_size:,} samples")

# Create t-SNE visualization
fig = px.scatter(
    x=tsne_result[:, 0],
    y=tsne_result[:, 1],
    color=tsne_domains,
    color_discrete_map=domain_colors,
    title="t-SNE Visualization of Cultural Dimensions",
    labels={'x': 't-SNE 1', 'y': 't-SNE 2', 'color': 'Domain'}
)

fig.update_traces(marker=dict(size=4, opacity=0.6))
fig.update_layout(height=600)
fig.show()


# In[24]:


# 5.3 K-means Clustering Analysis
print("=== K-MEANS CLUSTERING ANALYSIS ===")

# Determine optimal number of clusters using elbow method
max_clusters = 10
inertias = []
silhouette_scores = []

from sklearn.metrics import silhouette_score

for k in range(2, max_clusters + 1):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(pca_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(pca_scaled, cluster_labels))

# Find optimal k
optimal_k_silhouette = np.argmax(silhouette_scores) + 2
print(f"Optimal number of clusters (silhouette): {optimal_k_silhouette}")

# Perform final clustering
final_kmeans = KMeans(n_clusters=optimal_k_silhouette, random_state=42, n_init=10)
cluster_labels = final_kmeans.fit_predict(pca_scaled)

# Visualize clustering results
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        'Elbow Method',
        'Silhouette Scores',
        'Clusters in PCA Space',
        'Cluster Characteristics'
    ]
)

# Elbow plot
fig.add_trace(
    go.Scatter(
        x=list(range(2, max_clusters + 1)),
        y=inertias,
        mode='lines+markers',
        name='Inertia'
    ),
    row=1, col=1
)

# Silhouette scores
fig.add_trace(
    go.Scatter(
        x=list(range(2, max_clusters + 1)),
        y=silhouette_scores,
        mode='lines+markers',
        name='Silhouette Score'
    ),
    row=1, col=2
)

# Clusters in PCA space
for cluster_id in range(optimal_k_silhouette):
    cluster_mask = cluster_labels == cluster_id
    fig.add_trace(
        go.Scatter(
            x=pca_result[cluster_mask, 0],
            y=pca_result[cluster_mask, 1],
            mode='markers',
            name=f'Cluster {cluster_id}',
            marker=dict(size=4, opacity=0.6)
        ),
        row=2, col=1
    )

fig.update_layout(height=1000, title_text="K-means Clustering Analysis")
fig.show()

# Analyze cluster characteristics
cluster_df = df[section_score_cols].dropna().copy()
cluster_df['cluster'] = cluster_labels
cluster_df['domain'] = domain_labels.values

print("\nCluster Characteristics:")
cluster_summary = cluster_df.groupby('cluster')[section_score_cols].mean()
print(cluster_summary.round(3))

print("\nCluster Domain Distribution:")
cluster_domain_dist = pd.crosstab(cluster_df['cluster'], cluster_df['domain'], normalize='index')
print(cluster_domain_dist.round(3))


# In[25]:


# 5.4 Industry Benchmarking Dashboard
print("=== INDUSTRY BENCHMARKING DASHBOARD ===")

# Calculate industry statistics
industry_stats = df.groupby('domain').agg({
    'overall_culture_score': ['mean', 'std', 'count'],
    'organization_name': 'nunique',
    'employee_count': ['mean', 'median']
}).round(3)

# Flatten column names
industry_stats.columns = ['_'.join(col).strip() for col in industry_stats.columns]

print("Industry Statistics Summary:")
print(industry_stats)

# Create comprehensive dashboard
fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=[
        'Mean Culture Scores by Domain',
        'Response Distribution by Domain',
        'Organization Size Distribution', 
        'Section Score Comparison',
        'Score Variability by Domain',
        'Culture Score Percentiles'
    ],
    specs=[[{"type": "bar"}, {"type": "pie"}],
           [{"type": "box"}, {"type": "xy"}],
           [{"type": "bar"}, {"type": "xy"}]]
)

# Mean culture scores by domain
domain_means = df.groupby('domain')['overall_culture_score'].mean()
fig.add_trace(
    go.Bar(
        x=domain_means.index,
        y=domain_means.values,
        marker_color=[domain_colors[d] for d in domain_means.index],
        name='Mean Score'
    ),
    row=1, col=1
)

# Response distribution pie chart
fig.add_trace(
    go.Pie(
        labels=domain_dist.index,
        values=domain_dist.values,
        marker_colors=[domain_colors[d] for d in domain_dist.index],
        name='Responses'
    ),
    row=1, col=2
)

# Organization size distribution
for domain in domains:
    domain_sizes = df[df['domain'] == domain]['employee_count']
    fig.add_trace(
        go.Box(
            y=domain_sizes,
            name=domain,
            marker_color=domain_colors[domain],
            showlegend=False
        ),
        row=2, col=1
    )

# Section score comparison radar-style
for i, domain in enumerate(domains):
    domain_section_means = df[df['domain'] == domain][section_score_cols].mean()
    fig.add_trace(
        go.Scatter(
            x=short_names,
            y=domain_section_means.values,
            mode='lines+markers',
            name=domain,
            line=dict(color=domain_colors[domain]),
            showlegend=False
        ),
        row=2, col=2
    )

# Score variability (standard deviation)
domain_stds = df.groupby('domain')['overall_culture_score'].std()
fig.add_trace(
    go.Bar(
        x=domain_stds.index,
        y=domain_stds.values,
        marker_color=[domain_colors[d] for d in domain_stds.index],
        name='Std Dev'
    ),
    row=3, col=1
)

# Percentiles
percentiles = [10, 25, 50, 75, 90]
for domain in domains:
    domain_scores = df[df['domain'] == domain]['overall_culture_score']
    domain_percentiles = [np.percentile(domain_scores, p) for p in percentiles]
    
    fig.add_trace(
        go.Scatter(
            x=percentiles,
            y=domain_percentiles,
            mode='lines+markers',
            name=domain,
            line=dict(color=domain_colors[domain]),
            showlegend=False
        ),
        row=3, col=2
    )

fig.update_layout(height=1200, title_text="Industry Benchmarking Dashboard")
fig.show()


# In[26]:


# 5.5 Network Analysis - Organization Relationships
print("=== NETWORK ANALYSIS ===")

# Create organization similarity network based on cultural profiles
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import fcluster

# Filter organizations with sufficient responses
min_responses_network = 30
org_counts_network = df['organization_name'].value_counts()
network_orgs = org_counts_network[org_counts_network >= min_responses_network].index

print(f"Network analysis includes {len(network_orgs)} organizations with {min_responses_network}+ responses")

# Calculate organization profiles
org_profiles = df[df['organization_name'].isin(network_orgs)].groupby('organization_name')[section_score_cols].mean()

# Calculate pairwise distances
distances = pdist(org_profiles.values, metric='euclidean')
distance_matrix = squareform(distances)

# Convert to similarity (inverse of distance)
similarity_matrix = 1 / (1 + distance_matrix)

# Create network visualization using plotly
import networkx as nx

# Create graph
G = nx.Graph()

# Add nodes
org_domains = df[df['organization_name'].isin(network_orgs)].groupby('organization_name')['domain'].first()
for org in network_orgs:
    G.add_node(org, domain=org_domains[org])

# Add edges (only for similarities above threshold)
similarity_threshold = 0.8  # Adjust as needed
for i, org1 in enumerate(network_orgs):
    for j, org2 in enumerate(network_orgs[i+1:], i+1):
        similarity = similarity_matrix[i, j]
        if similarity > similarity_threshold:
            G.add_edge(org1, org2, weight=similarity)

print(f"Network has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
print(f"Similarity threshold: {similarity_threshold}")

# Calculate layout
pos = nx.spring_layout(G, k=1, iterations=50)

# Prepare node traces
node_trace = go.Scatter(
    x=[],
    y=[],
    text=[],
    mode='markers+text',
    textposition='middle center',
    hoverinfo='text',
    marker=dict(size=[], color=[], colorbar=dict(title="Domain"), line=dict(width=2))
)

# Prepare edge traces
edge_x = []
edge_y = []

for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=1, color='#888'),
    hoverinfo='none',
    mode='lines'
)

# Add node information
for node in G.nodes():
    x, y = pos[node]
    node_trace['x'] += tuple([x])
    node_trace['y'] += tuple([y])
    
    # Node info
    node_info = f'{node}<br>Domain: {org_domains[node]}<br>Connections: {len(list(G.neighbors(node)))}'
    node_trace['text'] += tuple([node_info])
    
    # Color by domain
    domain = org_domains[node]
    node_trace['marker']['color'] += tuple([list(domains).index(domain)])
    
    # Size by number of responses
    size = org_counts_network[node] / 10  # Scale for visualization
    node_trace['marker']['size'] += tuple([max(10, min(30, size))])

# Create figure
fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title='Organization Network - Cultural Similarity',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[ dict(
                        text="Node size = response count, Connections = cultural similarity",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002,
                        xanchor="left", yanchor="bottom",
                        font=dict(size=12)
                    )],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                ))

fig.show()

# Network statistics
print(f"\nNetwork Statistics:")
print(f"Density: {nx.density(G):.3f}")
print(f"Average clustering coefficient: {nx.average_clustering(G):.3f}")
if nx.is_connected(G):
    print(f"Average shortest path length: {nx.average_shortest_path_length(G):.3f}")
else:
    print(f"Graph is not connected. Number of components: {nx.number_connected_components(G)}")


# ## 6. ADVANCED STATISTICAL ANALYSIS

# In[27]:


# 6.1 Ridge Plots for Distribution Comparisons
print("=== RIDGE PLOTS FOR DISTRIBUTION COMPARISONS ===")

# Create ridge plot for section scores by domain
fig, axes = plt.subplots(len(domains), 1, figsize=(12, 4*len(domains)), sharex=True)

for i, domain in enumerate(domains):
    domain_data = df[df['domain'] == domain]
    
    for j, (section_name, _) in enumerate(sections.items()):
        score_col = f'{section_name}_score'
        scores = domain_data[score_col].dropna()
        
        # Create density plot
        axes[i].fill_between(
            np.linspace(scores.min(), scores.max(), 100),
            j,
            j + stats.gaussian_kde(scores)(np.linspace(scores.min(), scores.max(), 100)),
            alpha=0.7,
            label=short_names[j]
        )
    
    axes[i].set_title(f'{domain} Domain - Section Score Distributions')
    axes[i].set_yticks(range(len(sections)))
    axes[i].set_yticklabels(short_names)
    axes[i].set_xlabel('Score')
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# In[28]:


# 6.2 Treemap Visualization for Hierarchical Data
print("=== TREEMAP VISUALIZATION ===")

# Create hierarchical data: Domain > Organization > Department
treemap_data = df.groupby(['domain', 'organization_name', 'department']).agg({
    'response_id': 'count',
    'overall_culture_score': 'mean'
}).rename(columns={'response_id': 'count', 'overall_culture_score': 'avg_score'}).reset_index()

# Filter for better visualization
treemap_filtered = treemap_data[treemap_data['count'] >= 5]  # At least 5 responses

# Create treemap
fig = px.treemap(
    treemap_filtered,
    path=['domain', 'organization_name', 'department'],
    values='count',
    color='avg_score',
    color_continuous_scale='RdYlBu_r',
    title='Organizational Structure Treemap\n(Size = Response Count, Color = Average Culture Score)'
)

fig.update_layout(height=800)
fig.show()

print(f"Treemap includes {len(treemap_filtered)} domain-organization-department combinations")
print(f"Total responses represented: {treemap_filtered['count'].sum():,}")


# In[29]:


# 6.3 Sunburst Chart for Nested Categories
print("=== SUNBURST CHART ===")

# Prepare data for sunburst
sunburst_data = df.groupby(['domain', 'position_level', 'department']).agg({
    'response_id': 'count',
    'overall_culture_score': 'mean'
}).rename(columns={'response_id': 'count', 'overall_culture_score': 'avg_score'}).reset_index()

# Filter for visualization
sunburst_filtered = sunburst_data[sunburst_data['count'] >= 3]

# Create sunburst chart
fig = px.sunburst(
    sunburst_filtered,
    path=['domain', 'position_level', 'department'],
    values='count',
    color='avg_score',
    color_continuous_scale='RdYlBu_r',
    title='Organizational Hierarchy Sunburst\n(Size = Response Count, Color = Average Culture Score)'
)

fig.update_layout(height=700)
fig.show()

print(f"Sunburst includes {len(sunburst_filtered)} domain-position-department combinations")


# ## 7. EXECUTIVE SUMMARY & INSIGHTS
# 
# Key findings and actionable insights from the comprehensive analysis.

# In[30]:


# 7.1 Generate Executive Summary Statistics
print("=== EXECUTIVE SUMMARY ===")
print(f"Dataset Overview:")
print(f"- Total Responses: {len(df):,}")
print(f"- Organizations: {df['organization_name'].nunique():,}")
print(f"- Domains: {df['domain'].nunique()} ({', '.join(df['domain'].unique())})")
print(f"- Departments: {df['department'].nunique():,}")
print(f"- Response Rate by Domain:")
for domain in domains:
    count = len(df[df['domain'] == domain])
    pct = count / len(df) * 100
    print(f"  - {domain}: {count:,} ({pct:.1f}%)")

print(f"\nKey Cultural Findings:")
overall_mean = df['overall_culture_score'].mean()
overall_std = df['overall_culture_score'].std()
print(f"- Overall Culture Score: {overall_mean:.2f} ± {overall_std:.2f}")

print(f"\nSection Rankings (Worst to Best):")
section_means = df[section_score_cols].mean().sort_values(ascending=False)
for i, (section, score) in enumerate(section_means.items(), 1):
    section_name = section.replace('_score', '')
    print(f"{i}. {section_name}: {score:.2f}")

print(f"\nDomain Comparison:")
domain_means = df.groupby('domain')['overall_culture_score'].mean().sort_values(ascending=False)
for i, (domain, score) in enumerate(domain_means.items(), 1):
    print(f"{i}. {domain}: {score:.2f}")

print(f"\nStatistical Significance:")
f_stat, p_value = f_oneway(
    df[df['domain'] == 'Healthcare']['overall_culture_score'].dropna(),
    df[df['domain'] == 'University']['overall_culture_score'].dropna(),
    df[df['domain'] == 'Business']['overall_culture_score'].dropna()
)
print(f"- Domain differences: F={f_stat:.3f}, p={p_value:.3e}")
significance = "Highly Significant" if p_value < 0.001 else "Significant" if p_value < 0.05 else "Not Significant"
print(f"- Interpretation: {significance}")

print(f"\nCorrelation Insights:")
strongest_corr = section_corr.values[np.triu_indices_from(section_corr.values, k=1)].max()
corr_location = np.where(section_corr.values == strongest_corr)
if len(corr_location[0]) > 0:
    row_idx, col_idx = corr_location[0][0], corr_location[1][0]
    section1 = section_corr.index[row_idx].replace('_score', '')
    section2 = section_corr.columns[col_idx].replace('_score', '')
    print(f"- Strongest correlation: {section1} & {section2} (r={strongest_corr:.3f})")

print(f"\nTenure Insights:")
tenure_corr, tenure_p = stats.pearsonr(
    pd.Categorical(df['tenure_range'], categories=tenure_available, ordered=True).codes,
    df['overall_culture_score']
)
print(f"- Tenure vs Culture Score correlation: r={tenure_corr:.3f}, p={tenure_p:.3e}")
tenure_trend = "improves" if tenure_corr > 0 else "declines" if tenure_corr < 0 else "remains stable"
print(f"- Culture score {tenure_trend} with tenure")


# In[31]:


# 7.2 Key Recommendations Dashboard
print("\n=== KEY RECOMMENDATIONS ===")

# Identify organizations needing attention
org_scores = df.groupby('organization_name').agg({
    'overall_culture_score': ['mean', 'count'],
    'domain': 'first'
}).round(3)

org_scores.columns = ['avg_score', 'response_count', 'domain']
org_scores = org_scores[org_scores['response_count'] >= 20]  # Focus on orgs with sufficient data

# Worst performing organizations
worst_orgs = org_scores.nsmallest(5, 'avg_score')
print("\nTop 5 Organizations Needing Immediate Attention:")
for i, (org, data) in enumerate(worst_orgs.iterrows(), 1):
    print(f"{i}. {org} ({data['domain']}): {data['avg_score']:.2f} (n={data['response_count']})")

# Best performing organizations
best_orgs = org_scores.nlargest(5, 'avg_score')
print("\nTop 5 Best Performing Organizations (Benchmarks):")
for i, (org, data) in enumerate(best_orgs.iterrows(), 1):
    print(f"{i}. {org} ({data['domain']}): {data['avg_score']:.2f} (n={data['response_count']})")

# Section-specific recommendations
print("\nSection-Specific Priorities (Areas Needing Most Attention):")
section_priorities = section_means.sort_values(ascending=False)  # Highest scores = worst problems
for i, (section, score) in enumerate(section_priorities.items(), 1):
    section_name = section.replace('_score', '')
    priority_level = "Critical" if score > 3.0 else "High" if score > 2.5 else "Medium" if score > 2.0 else "Low"
    print(f"{i}. {section_name}: {score:.2f} ({priority_level} Priority)")

# Domain-specific recommendations
print("\nDomain-Specific Recommendations:")
for domain in domain_means.index:
    score = domain_means[domain]
    domain_data = df[df['domain'] == domain]
    worst_section = domain_data[section_score_cols].mean().idxmax().replace('_score', '')
    worst_score = domain_data[section_score_cols].mean().max()
    
    print(f"\n{domain} Domain (Score: {score:.2f}):")
    print(f"  - Primary concern: {worst_section} ({worst_score:.2f})")
    print(f"  - Organizations: {domain_data['organization_name'].nunique()}")
    print(f"  - Responses: {len(domain_data):,}")
    
    if domain == 'Healthcare':
        print(f"  - Recommendation: Focus on mental health support and burnout prevention")
    elif domain == 'University':
        print(f"  - Recommendation: Address power dynamics and accountability structures")
    elif domain == 'Business':
        print(f"  - Recommendation: Implement robust HR policies and leadership training")


# In[32]:


# 7.3 Final Visualization - Executive Dashboard
fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=[
        'Overall Culture Scores by Domain',
        'Section Priority Matrix',
        'Organizational Performance Distribution',
        'Culture Score vs Response Volume',
        'Tenure Impact Analysis',
        'Department Performance Comparison'
    ],
    specs=[[{"type": "bar"}, {"type": "scatter"}],
           [{"type": "histogram"}, {"type": "scatter"}],
           [{"type": "box"}, {"type": "bar"}]]
)

# Overall scores by domain
fig.add_trace(
    go.Bar(
        x=domain_means.index,
        y=domain_means.values,
        marker_color=[domain_colors[d] for d in domain_means.index],
        name='Domain Scores',
        showlegend=False
    ),
    row=1, col=1
)

# Section priority matrix
section_std = df[section_score_cols].std()
fig.add_trace(
    go.Scatter(
        x=section_means.values,
        y=section_std.values,
        text=[name.replace(' & ', '\n&\n') for name in section_means.index.str.replace('_score', '')],
        mode='markers+text',
        marker=dict(size=15, color='red'),
        textposition='middle right',
        name='Section Priority',
        showlegend=False
    ),
    row=1, col=2
)

# Organizational performance distribution
org_scores_all = df.groupby('organization_name')['overall_culture_score'].mean()
fig.add_trace(
    go.Histogram(
        x=org_scores_all.values,
        nbinsx=20,
        marker_color='lightblue',
        name='Org Distribution',
        showlegend=False
    ),
    row=2, col=1
)

# Culture score vs response volume
org_data_scatter = df.groupby('organization_name').agg({
    'overall_culture_score': 'mean',
    'response_id': 'count',
    'domain': 'first'
})

for domain in domains:
    domain_orgs = org_data_scatter[org_data_scatter['domain'] == domain]
    fig.add_trace(
        go.Scatter(
            x=domain_orgs['response_id'],
            y=domain_orgs['overall_culture_score'],
            mode='markers',
            name=domain,
            marker=dict(color=domain_colors[domain], size=8),
            showlegend=False
        ),
        row=2, col=2
    )

# Tenure impact
for tenure in tenure_available:
    tenure_scores = df[df['tenure_range'] == tenure]['overall_culture_score']
    fig.add_trace(
        go.Box(
            y=tenure_scores,
            name=tenure,
            showlegend=False
        ),
        row=3, col=1
    )

# Department performance
dept_means = df.groupby('department')['overall_culture_score'].mean().sort_values(ascending=False).head(10)
fig.add_trace(
    go.Bar(
        x=dept_means.values,
        y=dept_means.index,
        orientation='h',
        marker_color='green',
        name='Dept Performance',
        showlegend=False
    ),
    row=3, col=2
)

fig.update_layout(
    height=1200,
    title_text="HSEG Survey Analysis - Executive Dashboard",
    title_font_size=20
)

fig.show()

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"This comprehensive analysis examined {len(df):,} survey responses")
print(f"across {df['organization_name'].nunique()} organizations")
print(f"in {len(domains)} domains: {', '.join(domains)}")
print("\nKey deliverables:")
print("✓ Intra-section analysis with correlation matrices and clustering")
print("✓ Intra-business analysis with organizational benchmarking")
print("✓ Domain analysis with statistical comparisons")
print("✓ Cross-industry visualization with PCA, t-SNE, and network analysis")
print("✓ Advanced visualizations including treemaps, sunburst, and ridge plots")
print("✓ Executive summary with actionable recommendations")
print("\nReady for stakeholder presentation and strategic decision-making.")


# ## 5. DEMOGRAPHIC-BASED CULTURE ANALYSIS
# 
# Comprehensive analysis of workplace culture across demographic segments including age, gender, race/ethnicity, education, tenure, position level, role, and supervision status.

# In[33]:


# 5.1 Demographic Data Overview and Preprocessing
print("=== DEMOGRAPHIC DATA OVERVIEW ===")

# Define demographic columns and their labels
demographic_cols = {
    'q26': 'Age Range',
    'q27': 'Gender Identity', 
    'q28': 'Race/Ethnicity',
    'q29': 'Education Level',
    'q30': 'Tenure',
    'q31': 'Position Level',
    'q32': 'Domain Role',
    'q33': 'Supervises Others'
}

# Display distribution for each demographic
for col, label in demographic_cols.items():
    if col in df.columns:
        print(f"\n{label} ({col}) Distribution:")
        dist = df[col].value_counts().sort_index()
        print(dist)
        print(f"Missing values: {df[col].isnull().sum()} ({df[col].isnull().sum()/len(df)*100:.1f}%)")
    else:
        print(f"\n{label} ({col}): Column not found in dataset")

# Check data quality for demographic analysis
print(f"\n=== DEMOGRAPHIC DATA QUALITY ===")
print(f"Total responses: {len(df):,}")

# Calculate complete demographic profiles
demographic_complete = df[list(demographic_cols.keys())].notna().all(axis=1)
print(f"Complete demographic profiles: {demographic_complete.sum():,} ({demographic_complete.sum()/len(df)*100:.1f}%)")

# Create clean demographic dataset
df_demo = df.copy()

# Standardize demographic categories for analysis
print(f"\nStandardizing demographic categories...")

# Age range standardization (if needed)
if 'q26' in df_demo.columns:
    age_mapping = {
        'Under 25': '<25',
        '25-34': '25-34', 
        '35-44': '35-44',
        '45-54': '45-54',
        '55-64': '55-64',
        '65+': '65+'
    }
    df_demo['age_range_clean'] = df_demo['q26'].map(age_mapping).fillna(df_demo['q26'])

print("Demographic preprocessing completed!")


# In[34]:


# 5.2 Age Range Analysis (Q26)
print("=== AGE RANGE ANALYSIS ===")

# Check if age data exists
if 'q26' in df.columns:
    age_col = 'q26'
elif 'age_range' in df.columns:
    age_col = 'age_range'
else:
    print("Age range data not found in expected columns")
    age_col = None

if age_col:
    # Statistical analysis by age group
    age_stats = df.groupby(age_col).agg({
        'overall_culture_score': ['count', 'mean', 'std', 'median'],
        **{col: 'mean' for col in section_score_cols}
    }).round(3)
    
    print("Culture Scores by Age Range:")
    print(age_stats['overall_culture_score'])
    
    # ANOVA test for age differences
    age_groups = []
    age_labels = []
    for age_group in df[age_col].dropna().unique():
        age_data = df[df[age_col] == age_group]['overall_culture_score'].dropna()
        if len(age_data) >= 10:  # Minimum group size
            age_groups.append(age_data)
            age_labels.append(age_group)
    
    if len(age_groups) > 1:
        f_stat, p_value = f_oneway(*age_groups)
        print(f"\nAge Group ANOVA: F={f_stat:.3f}, p={p_value:.3e}")
        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        print(f"Statistical significance: {significance}")
        
        # Effect size (eta-squared)
        total_sum_sq = sum([np.sum((group - df['overall_culture_score'].mean())**2) for group in age_groups])
        between_sum_sq = sum([len(group) * (group.mean() - df['overall_culture_score'].mean())**2 for group in age_groups])
        eta_squared = between_sum_sq / total_sum_sq
        print(f"Effect size (η²): {eta_squared:.3f}")
    
    # Visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Culture Scores by Age Range',
            'Age Distribution',
            'Section Scores by Age Range',
            'Age vs Domain Interaction'
        ],
        specs=[[{"type": "box"}, {"type": "bar"}],
               [{"type": "xy"}, {"type": "xy"}]]
    )
    
    # Box plot by age
    for age_group in sorted(df[age_col].dropna().unique()):
        age_data = df[df[age_col] == age_group]['overall_culture_score']
        fig.add_trace(
            go.Box(y=age_data, name=age_group, showlegend=False),
            row=1, col=1
        )
    
    # Age distribution
    age_counts = df[age_col].value_counts().sort_index()
    fig.add_trace(
        go.Bar(x=age_counts.index, y=age_counts.values, showlegend=False, marker_color='lightblue'),
        row=1, col=2
    )
    
    # Section scores by age (heatmap data preparation)
    age_section_means = df.groupby(age_col)[section_score_cols].mean()
    
    # Line plot for section scores by age
    for i, section in enumerate(section_score_cols):
        section_name = section.replace('_score', '').replace(' & ', ' &\n')
        fig.add_trace(
            go.Scatter(
                x=age_section_means.index,
                y=age_section_means[section],
                mode='lines+markers',
                name=section_name,
                showlegend=(i < 3)  # Show legend for first 3 only
            ),
            row=2, col=1
        )
    
    # Age vs Domain interaction
    age_domain_means = df.groupby([age_col, 'domain'])['overall_culture_score'].mean().unstack()
    for domain in age_domain_means.columns:
        fig.add_trace(
            go.Scatter(
                x=age_domain_means.index,
                y=age_domain_means[domain],
                mode='lines+markers',
                name=domain,
                line=dict(color=domain_colors[domain]),
                showlegend=False
            ),
            row=2, col=2
        )
    
    fig.update_layout(height=800, title_text="Age Range Analysis Dashboard")
    fig.show()
    
    # Statistical summary
    print("\nAge Range Insights:")
    age_means = df.groupby(age_col)['overall_culture_score'].mean().sort_values()
    print(f"Lowest culture scores: {age_means.index[0]} ({age_means.iloc[0]:.2f})")
    print(f"Highest culture scores: {age_means.index[-1]} ({age_means.iloc[-1]:.2f})")
    print(f"Age range spread: {age_means.iloc[-1] - age_means.iloc[0]:.2f} points")

else:
    print("Age range analysis skipped - data not available")


# In[35]:


# 5.3 Gender Identity Analysis (Q27)
print("=== GENDER IDENTITY ANALYSIS ===")

# Check for gender data
gender_cols = ['q27', 'gender', 'gender_identity']
gender_col = None
for col in gender_cols:
    if col in df.columns:
        gender_col = col
        break

if gender_col:
    print(f"Using gender column: {gender_col}")
    
    # Clean gender data
    gender_counts = df[gender_col].value_counts()
    print(f"Gender distribution:")
    print(gender_counts)
    
    # Filter out very small groups for statistical analysis
    min_group_size = 30
    valid_genders = gender_counts[gender_counts >= min_group_size].index
    df_gender = df[df[gender_col].isin(valid_genders)].copy()
    
    print(f"\nAnalyzing {len(valid_genders)} gender groups with {min_group_size}+ responses each")
    print(f"Total responses in analysis: {len(df_gender):,}")
    
    # Statistical analysis
    gender_stats = df_gender.groupby(gender_col).agg({
        'overall_culture_score': ['count', 'mean', 'std', 'median'],
        **{col: 'mean' for col in section_score_cols}
    }).round(3)
    
    print("\nCulture Scores by Gender:")
    print(gender_stats['overall_culture_score'])
    
    # Statistical testing
    if len(valid_genders) >= 2:
        gender_groups = [df_gender[df_gender[gender_col] == gender]['overall_culture_score'].dropna() 
                        for gender in valid_genders]
        
        if len(valid_genders) == 2:
            # T-test for two groups
            t_stat, p_value = stats.ttest_ind(gender_groups[0], gender_groups[1])
            print(f"\nGender T-test: t={t_stat:.3f}, p={p_value:.3e}")
            
            # Cohen's d effect size
            pooled_std = np.sqrt(((len(gender_groups[0])-1)*gender_groups[0].std()**2 + 
                                 (len(gender_groups[1])-1)*gender_groups[1].std()**2) / 
                                (len(gender_groups[0])+len(gender_groups[1])-2))
            cohens_d = (gender_groups[0].mean() - gender_groups[1].mean()) / pooled_std
            print(f"Cohen's d effect size: {cohens_d:.3f}")
            
        else:
            # ANOVA for multiple groups
            f_stat, p_value = f_oneway(*gender_groups)
            print(f"\nGender ANOVA: F={f_stat:.3f}, p={p_value:.3e}")
        
        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        print(f"Statistical significance: {significance}")
    
    # Visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Culture Scores by Gender',
            'Gender Representation',
            'Gender-Section Heatmap',
            'Gender-Domain Interaction'
        ],
        specs=[[{"type": "violin"}, {"type": "pie"}],
               [{"type": "xy"}, {"type": "bar"}]]
    )
    
    # Violin plots by gender
    colors_gender = px.colors.qualitative.Set2
    for i, gender in enumerate(valid_genders):
        gender_data = df_gender[df_gender[gender_col] == gender]['overall_culture_score']
        fig.add_trace(
            go.Violin(
                y=gender_data,
                name=gender,
                box_visible=True,
                meanline_visible=True,
                fillcolor=colors_gender[i % len(colors_gender)],
                opacity=0.6,
                showlegend=False
            ),
            row=1, col=1
        )
    
    # Gender representation pie chart
    fig.add_trace(
        go.Pie(
            labels=gender_counts.index,
            values=gender_counts.values,
            name="Gender Distribution"
        ),
        row=1, col=2
    )
    
    # Gender-section heatmap (prepare data)
    gender_section_means = df_gender.groupby(gender_col)[section_score_cols].mean()
    
    # Plot as grouped bar chart
    for i, section in enumerate(section_score_cols):
        section_name = section.replace('_score', '')
        for j, gender in enumerate(valid_genders):
            fig.add_trace(
                go.Bar(
                    x=[section_name],
                    y=[gender_section_means.loc[gender, section]],
                    name=gender,
                    marker_color=colors_gender[j % len(colors_gender)],
                    showlegend=(i == 0),
                    offsetgroup=gender
                ),
                row=2, col=1
            )
    
    # Gender-domain interaction
    gender_domain_means = df_gender.groupby([gender_col, 'domain'])['overall_culture_score'].mean().unstack()
    for gender in valid_genders:
        if gender in gender_domain_means.index:
            fig.add_trace(
                go.Bar(
                    x=gender_domain_means.columns,
                    y=gender_domain_means.loc[gender],
                    name=f"{gender}",
                    showlegend=False
                ),
                row=2, col=2
            )
    
    fig.update_layout(height=1000, title_text="Gender Identity Analysis Dashboard")
    fig.show()
    
    # Gender insights
    print("\nGender Identity Insights:")
    gender_means = df_gender.groupby(gender_col)['overall_culture_score'].mean().sort_values()
    print(f"Lowest culture scores: {gender_means.index[0]} ({gender_means.iloc[0]:.2f})")
    print(f"Highest culture scores: {gender_means.index[-1]} ({gender_means.iloc[-1]:.2f})")
    print(f"Gender score gap: {gender_means.iloc[-1] - gender_means.iloc[0]:.2f} points")
    
    # Check for specific gender disparities by section
    print("\nSection-specific gender gaps:")
    for section in section_score_cols:
        section_gender_means = df_gender.groupby(gender_col)[section].mean()
        gap = section_gender_means.max() - section_gender_means.min()
        worst_gender = section_gender_means.idxmin()
        best_gender = section_gender_means.idxmax()
        section_name = section.replace('_score', '')
        print(f"{section_name}: {gap:.2f} point gap ({worst_gender}: {section_gender_means[worst_gender]:.2f}, {best_gender}: {section_gender_means[best_gender]:.2f})")

else:
    print("Gender identity analysis skipped - data not available")


# In[36]:


# 5.4 Race/Ethnicity Analysis (Q28)
print("=== RACE/ETHNICITY ANALYSIS ===")

# Check for race/ethnicity data
race_cols = ['q28', 'race', 'ethnicity', 'race_ethnicity']
race_col = None
for col in race_cols:
    if col in df.columns:
        race_col = col
        break

if race_col:
    print(f"Using race/ethnicity column: {race_col}")
    
    # Clean race/ethnicity data
    race_counts = df[race_col].value_counts()
    print(f"Race/Ethnicity distribution:")
    for race, count in race_counts.items():
        pct = count / len(df) * 100
        print(f"  {race}: {count:,} ({pct:.1f}%)")
    
    # Filter groups for statistical analysis
    min_group_size = 50  # Higher threshold for race/ethnicity analysis
    valid_races = race_counts[race_counts >= min_group_size].index
    df_race = df[df[race_col].isin(valid_races)].copy()
    
    print(f"\nAnalyzing {len(valid_races)} racial/ethnic groups with {min_group_size}+ responses each")
    print(f"Total responses in analysis: {len(df_race):,}")
    
    # Statistical analysis
    race_stats = df_race.groupby(race_col).agg({
        'overall_culture_score': ['count', 'mean', 'std', 'median'],
        **{col: 'mean' for col in section_score_cols}
    }).round(3)
    
    print("\nCulture Scores by Race/Ethnicity:")
    print(race_stats['overall_culture_score'])
    
    # Statistical testing
    if len(valid_races) >= 2:
        race_groups = [df_race[df_race[race_col] == race]['overall_culture_score'].dropna() 
                      for race in valid_races]
        
        # ANOVA for racial differences
        f_stat, p_value = f_oneway(*race_groups)
        print(f"\nRace/Ethnicity ANOVA: F={f_stat:.3f}, p={p_value:.3e}")
        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        print(f"Statistical significance: {significance}")
        
        # Post-hoc pairwise comparisons (if significant)
        if p_value < 0.05:
            print("\nPairwise comparisons:")
            from itertools import combinations
            for race1, race2 in combinations(valid_races, 2):
                group1 = df_race[df_race[race_col] == race1]['overall_culture_score']
                group2 = df_race[df_race[race_col] == race2]['overall_culture_score']
                t_stat, p_val = stats.ttest_ind(group1, group2)
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                print(f"  {race1} vs {race2}: t={t_stat:.3f}, p={p_val:.3e} {sig}")
    
    # Create comprehensive visualization
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            'Culture Scores by Race/Ethnicity',
            'Racial/Ethnic Representation',
            'Section Scores Heatmap',
            'Race-Domain Interaction',
            'Score Distributions by Race',
            'Confidence Intervals'
        ],
        specs=[[{"type": "box"}, {"type": "bar"}],
               [{"type": "xy"}, {"type": "bar"}],
               [{"type": "violin"}, {"type": "xy"}]]
    )
    
    # Box plots by race/ethnicity
    colors_race = px.colors.qualitative.Set3
    for i, race in enumerate(valid_races):
        race_data = df_race[df_race[race_col] == race]['overall_culture_score']
        fig.add_trace(
            go.Box(
                y=race_data,
                name=race,
                marker_color=colors_race[i % len(colors_race)],
                showlegend=False
            ),
            row=1, col=1
        )
    
    # Representation bar chart
    fig.add_trace(
        go.Bar(
            x=race_counts.index,
            y=race_counts.values,
            marker_color='lightcoral',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Section scores heatmap data
    race_section_means = df_race.groupby(race_col)[section_score_cols].mean()
    
    # Create heatmap using plotly
    z_values = race_section_means.values
    x_labels = [col.replace('_score', '') for col in section_score_cols]
    y_labels = race_section_means.index
    
    fig.add_trace(
        go.Heatmap(
            z=z_values,
            x=x_labels,
            y=y_labels,
            colorscale='RdYlBu_r',
            showscale=False
        ),
        row=2, col=1
    )
    
    # Race-domain interaction
    race_domain_means = df_race.groupby([race_col, 'domain'])['overall_culture_score'].mean().unstack()
    for i, race in enumerate(valid_races):
        if race in race_domain_means.index:
            fig.add_trace(
                go.Bar(
                    x=race_domain_means.columns,
                    y=race_domain_means.loc[race],
                    name=race,
                    marker_color=colors_race[i % len(colors_race)],
                    showlegend=False
                ),
                row=2, col=2
            )
    
    # Violin plots for distribution comparison
    for i, race in enumerate(valid_races):
        race_data = df_race[df_race[race_col] == race]['overall_culture_score']
        fig.add_trace(
            go.Violin(
                y=race_data,
                name=race,
                side='positive',
                fillcolor=colors_race[i % len(colors_race)],
                opacity=0.6,
                showlegend=False
            ),
            row=3, col=1
        )
    
    # Confidence intervals
    race_means = []
    race_cis = []
    race_names = []
    for race in valid_races:
        race_data = df_race[df_race[race_col] == race]['overall_culture_score'].dropna()
        mean_score = race_data.mean()
        sem = stats.sem(race_data)
        ci = stats.t.interval(0.95, len(race_data)-1, loc=mean_score, scale=sem)
        
        race_means.append(mean_score)
        race_cis.append([mean_score - ci[0], ci[1] - mean_score])
        race_names.append(race)
    
    fig.add_trace(
        go.Scatter(
            x=race_means,
            y=race_names,
            error_x=dict(
                type='data',
                symmetric=False,
                array=[ci[1] for ci in race_cis],
                arrayminus=[ci[0] for ci in race_cis]
            ),
            mode='markers',
            marker=dict(size=10, color='red'),
            showlegend=False
        ),
        row=3, col=2
    )
    
    fig.update_layout(height=1200, title_text="Race/Ethnicity Analysis Dashboard")
    fig.show()
    
    # Race/ethnicity insights
    print("\nRace/Ethnicity Insights:")
    race_means = df_race.groupby(race_col)['overall_culture_score'].mean().sort_values()
    print(f"Lowest culture scores: {race_means.index[0]} ({race_means.iloc[0]:.2f})")
    print(f"Highest culture scores: {race_means.index[-1]} ({race_means.iloc[-1]:.2f})")
    print(f"Racial/ethnic score gap: {race_means.iloc[-1] - race_means.iloc[0]:.2f} points")
    
    # Identify sections with largest racial disparities
    print("\nSections with largest racial disparities:")
    for section in section_score_cols:
        section_race_means = df_race.groupby(race_col)[section].mean()
        gap = section_race_means.max() - section_race_means.min()
        worst_race = section_race_means.idxmin()
        best_race = section_race_means.idxmax()
        section_name = section.replace('_score', '')
        print(f"{section_name}: {gap:.2f} point gap ({worst_race}: {section_race_means[worst_race]:.2f}, {best_race}: {section_race_means[best_race]:.2f})")

else:
    print("Race/ethnicity analysis skipped - data not available")


# In[37]:


# 5.5 Education Level Analysis (Q29)
print("=== EDUCATION LEVEL ANALYSIS ===")

# Check for education data
education_cols = ['q29', 'education', 'education_level']
education_col = None
for col in education_cols:
    if col in df.columns:
        education_col = col
        break

if education_col:
    print(f"Using education column: {education_col}")
    
    # Clean education data and create ordered categories
    education_counts = df[education_col].value_counts()
    print(f"Education Level distribution:")
    for edu, count in education_counts.items():
        pct = count / len(df) * 100
        print(f"  {edu}: {count:,} ({pct:.1f}%)")
    
    # Define education hierarchy for ordered analysis
    education_order = [
        'High School',
        'Some College',
        "Associate's Degree",
        "Bachelor's Degree", 
        "Master's Degree",
        'Doctoral Degree',
        'Professional Degree'
    ]
    
    # Map actual values to standard order
    available_education = [edu for edu in education_order if edu in df[education_col].unique()]
    if not available_education:
        # Use actual unique values if standard mapping doesn't work
        available_education = sorted(df[education_col].dropna().unique())
    
    print(f"\nEducation levels for analysis: {available_education}")
    
    # Filter for analysis
    df_education = df[df[education_col].isin(available_education)].copy()
    
    # Statistical analysis
    education_stats = df_education.groupby(education_col).agg({
        'overall_culture_score': ['count', 'mean', 'std', 'median'],
        **{col: 'mean' for col in section_score_cols}
    }).round(3)
    
    print("\nCulture Scores by Education Level:")
    education_ordered_stats = education_stats.reindex(available_education)
    print(education_ordered_stats['overall_culture_score'])
    
    # Statistical testing for education gradient
    if len(available_education) >= 3:
        # Test for linear trend
        education_means = []
        for edu in available_education:
            edu_scores = df_education[df_education[education_col] == edu]['overall_culture_score'].dropna()
            if len(edu_scores) > 0:
                education_means.append(edu_scores.mean())
        
        # Correlation with education level (ordinal)
        education_numeric = list(range(len(available_education)))
        corr, p_value = stats.pearsonr(education_numeric, education_means)
        print(f"\nEducation level correlation: r={corr:.3f}, p={p_value:.3e}")
        
        # ANOVA test
        education_groups = [df_education[df_education[education_col] == edu]['overall_culture_score'].dropna() 
                           for edu in available_education]
        education_groups = [group for group in education_groups if len(group) >= 10]
        
        if len(education_groups) >= 2:
            f_stat, anova_p = f_oneway(*education_groups)
            print(f"Education ANOVA: F={f_stat:.3f}, p={anova_p:.3e}")
            significance = "***" if anova_p < 0.001 else "**" if anova_p < 0.01 else "*" if anova_p < 0.05 else "ns"
            print(f"Statistical significance: {significance}")
    
    # Visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Culture Scores by Education Level',
            'Education Distribution',
            'Education-Section Analysis',
            'Education Progression Trend'
        ],
        specs=[[{"type": "box"}, {"type": "bar"}],
               [{"type": "xy"}, {"type": "scatter"}]]
    )
    
    # Box plots by education level
    colors_education = px.colors.qualitative.Pastel
    for i, edu in enumerate(available_education):
        edu_data = df_education[df_education[education_col] == edu]['overall_culture_score']
        fig.add_trace(
            go.Box(
                y=edu_data,
                name=edu,
                marker_color=colors_education[i % len(colors_education)],
                showlegend=False
            ),
            row=1, col=1
        )
    
    # Education distribution
    edu_counts_ordered = education_counts.reindex(available_education)
    fig.add_trace(
        go.Bar(
            x=edu_counts_ordered.index,
            y=edu_counts_ordered.values,
            marker_color='lightgreen',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Education-section heatmap
    education_section_means = df_education.groupby(education_col)[section_score_cols].mean()
    education_section_ordered = education_section_means.reindex(available_education)
    
    # Plot as grouped lines
    for i, section in enumerate(section_score_cols):
        section_name = section.replace('_score', '').replace(' & ', ' &\n')
        fig.add_trace(
            go.Scatter(
                x=available_education,
                y=education_section_ordered[section],
                mode='lines+markers',
                name=section_name,
                showlegend=(i < 3)
            ),
            row=2, col=1
        )
    
    # Education progression trend
    education_overall_means = df_education.groupby(education_col)['overall_culture_score'].mean().reindex(available_education)
    education_overall_counts = df_education.groupby(education_col)['overall_culture_score'].count().reindex(available_education)
    
    fig.add_trace(
        go.Scatter(
            x=list(range(len(available_education))),
            y=education_overall_means.values,
            mode='lines+markers+text',
            text=available_education,
            textposition='top center',
            marker=dict(size=education_overall_counts.values/50, color='red'),
            line=dict(width=3),
            showlegend=False
        ),
        row=2, col=2
    )
    
    fig.update_layout(height=1000, title_text="Education Level Analysis Dashboard")
    fig.show()
    
    # Education insights
    print("\nEducation Level Insights:")
    education_means_sorted = df_education.groupby(education_col)['overall_culture_score'].mean().sort_values()
    print(f"Lowest culture scores: {education_means_sorted.index[0]} ({education_means_sorted.iloc[0]:.2f})")
    print(f"Highest culture scores: {education_means_sorted.index[-1]} ({education_means_sorted.iloc[-1]:.2f})")
    print(f"Education score range: {education_means_sorted.iloc[-1] - education_means_sorted.iloc[0]:.2f} points")
    
    # Correlation between education and each section
    print("\nEducation correlation by section:")
    for section in section_score_cols:
        section_means_by_edu = []
        for edu in available_education:
            section_score = df_education[df_education[education_col] == edu][section].mean()
            section_means_by_edu.append(section_score)
        
        if len(section_means_by_edu) > 2:
            section_corr, section_p = stats.pearsonr(education_numeric[:len(section_means_by_edu)], section_means_by_edu)
            section_name = section.replace('_score', '')
            print(f"  {section_name}: r={section_corr:.3f}, p={section_p:.3e}")

else:
    print("Education level analysis skipped - data not available")


# ## 6. PROFESSIONAL ROLE ANALYSIS
# 
# Comprehensive analysis of workplace culture across professional dimensions including position levels, departments, roles, and supervision responsibilities.

# In[38]:


# 6.1 Position Level Deep Dive Analysis (Q31)
print("=== POSITION LEVEL ANALYSIS ===")

# Check for position level data
position_cols = ['q31', 'position_level', 'job_level', 'level']
position_col = None
for col in position_cols:
    if col in df.columns:
        position_col = col
        break

if position_col:
    print(f"Using position level column: {position_col}")
    
    # Define position hierarchy
    position_hierarchy = [
        'Entry Level',
        'Mid Level', 
        'Senior Level',
        'Executive Level',
        'C-Suite'
    ]
    
    # Clean position data
    position_counts = df[position_col].value_counts()
    print(f"Position Level distribution:")
    for pos, count in position_counts.items():
        pct = count / len(df) * 100
        print(f"  {pos}: {count:,} ({pct:.1f}%)")
    
    # Map to hierarchy or use actual values
    available_positions = [pos for pos in position_hierarchy if pos in df[position_col].unique()]
    if not available_positions:
        available_positions = sorted(df[position_col].dropna().unique())
    
    print(f"\nPosition levels for analysis: {available_positions}")
    
    # Filter for analysis
    df_position = df[df[position_col].isin(available_positions)].copy()
    
    # Statistical analysis by position level
    position_stats = df_position.groupby(position_col).agg({
        'overall_culture_score': ['count', 'mean', 'std', 'median'],
        **{col: 'mean' for col in section_score_cols}
    }).round(3)
    
    print("\nCulture Scores by Position Level:")
    if available_positions == [pos for pos in position_hierarchy if pos in available_positions]:
        position_ordered_stats = position_stats.reindex(available_positions)
    else:
        position_ordered_stats = position_stats
    print(position_ordered_stats['overall_culture_score'])
    
    # Statistical testing for position hierarchy effect
    if len(available_positions) >= 3:
        # ANOVA test
        position_groups = [df_position[df_position[position_col] == pos]['overall_culture_score'].dropna() 
                          for pos in available_positions]
        position_groups = [group for group in position_groups if len(group) >= 10]
        
        if len(position_groups) >= 2:
            f_stat, p_value = f_oneway(*position_groups)
            print(f"\nPosition Level ANOVA: F={f_stat:.3f}, p={p_value:.3e}")
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            print(f"Statistical significance: {significance}")
            
            # Test for linear trend in hierarchy
            if len(available_positions) >= 4:
                position_means = [df_position[df_position[position_col] == pos]['overall_culture_score'].mean() 
                                for pos in available_positions]
                position_numeric = list(range(len(available_positions)))
                corr, corr_p = stats.pearsonr(position_numeric, position_means)
                print(f"Hierarchical trend correlation: r={corr:.3f}, p={corr_p:.3e}")
    
    # Comprehensive visualization
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            'Culture Scores by Position Level',
            'Position Level Distribution',
            'Position-Section Score Matrix',
            'Position-Domain Interaction',
            'Leadership vs Individual Contributors',
            'Position Level Progression'
        ],
        specs=[[{"type": "box"}, {"type": "bar"}],
               [{"type": "xy"}, {"type": "bar"}],
               [{"type": "violin"}, {"type": "scatter"}]]
    )
    
    # Box plots by position level
    colors_position = px.colors.qualitative.Dark2
    for i, pos in enumerate(available_positions):
        pos_data = df_position[df_position[position_col] == pos]['overall_culture_score']
        fig.add_trace(
            go.Box(
                y=pos_data,
                name=pos,
                marker_color=colors_position[i % len(colors_position)],
                showlegend=False
            ),
            row=1, col=1
        )
    
    # Position distribution
    pos_counts_ordered = position_counts.reindex(available_positions) if len(available_positions) <= 5 else position_counts
    fig.add_trace(
        go.Bar(
            x=pos_counts_ordered.index,
            y=pos_counts_ordered.values,
            marker_color='lightblue',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Position-section heatmap
    position_section_means = df_position.groupby(position_col)[section_score_cols].mean()
    
    # Create heatmap
    z_values = position_section_means.values
    x_labels = [col.replace('_score', '') for col in section_score_cols]
    y_labels = position_section_means.index
    
    fig.add_trace(
        go.Heatmap(
            z=z_values,
            x=x_labels,
            y=y_labels,
            colorscale='RdYlBu_r',
            showscale=True
        ),
        row=2, col=1
    )
    
    # Position-domain interaction
    position_domain_means = df_position.groupby([position_col, 'domain'])['overall_culture_score'].mean().unstack()
    for i, pos in enumerate(available_positions):
        if pos in position_domain_means.index:
            fig.add_trace(
                go.Bar(
                    x=position_domain_means.columns,
                    y=position_domain_means.loc[pos],
                    name=pos,
                    marker_color=colors_position[i % len(colors_position)],
                    showlegend=False
                ),
                row=2, col=2
            )
    
    # Leadership categories
    leadership_positions = ['Senior Level', 'Executive Level', 'C-Suite', 'Manager', 'Director', 'VP']
    df_position['is_leadership'] = df_position[position_col].apply(
        lambda x: 'Leadership' if any(lead in str(x) for lead in leadership_positions) else 'Individual Contributor'
    )
    
    for leadership_type in ['Leadership', 'Individual Contributor']:
        leadership_data = df_position[df_position['is_leadership'] == leadership_type]['overall_culture_score']
        fig.add_trace(
            go.Violin(
                y=leadership_data,
                name=leadership_type,
                side='positive',
                showlegend=False
            ),
            row=3, col=1
        )
    
    # Position progression trend
    position_means = df_position.groupby(position_col)['overall_culture_score'].mean()
    position_counts_for_trend = df_position.groupby(position_col).size()
    
    fig.add_trace(
        go.Scatter(
            x=list(range(len(position_means))),
            y=position_means.values,
            mode='lines+markers+text',
            text=position_means.index,
            textposition='top center',
            marker=dict(size=position_counts_for_trend.values/50, color='red'),
            line=dict(width=3),
            showlegend=False
        ),
        row=3, col=2
    )
    
    fig.update_layout(height=1200, title_text="Position Level Analysis Dashboard")
    fig.show()
    
    # Position level insights
    print("\nPosition Level Insights:")
    position_means_sorted = df_position.groupby(position_col)['overall_culture_score'].mean().sort_values()
    print(f"Lowest culture scores: {position_means_sorted.index[0]} ({position_means_sorted.iloc[0]:.2f})")
    print(f"Highest culture scores: {position_means_sorted.index[-1]} ({position_means_sorted.iloc[-1]:.2f})")
    print(f"Position score range: {position_means_sorted.iloc[-1] - position_means_sorted.iloc[0]:.2f} points")
    
    # Leadership vs IC comparison
    leadership_stats = df_position.groupby('is_leadership')['overall_culture_score'].agg(['count', 'mean', 'std'])
    print(f"\nLeadership vs Individual Contributors:")
    print(leadership_stats)
    
    # Statistical test
    if len(df_position['is_leadership'].unique()) == 2:
        leadership_scores = df_position[df_position['is_leadership'] == 'Leadership']['overall_culture_score']
        ic_scores = df_position[df_position['is_leadership'] == 'Individual Contributor']['overall_culture_score']
        t_stat, t_p = stats.ttest_ind(leadership_scores, ic_scores)
        print(f"Leadership vs IC t-test: t={t_stat:.3f}, p={t_p:.3e}")

else:
    print("Position level analysis skipped - data not available")


# In[39]:


# 6.2 Department Analysis
print("=== DEPARTMENT ANALYSIS ===")

# Check department data availability
if 'department' in df.columns:
    print("Department column found")
    
    # Clean department data
    dept_counts = df['department'].value_counts()
    print(f"Department distribution (top 15):")
    for dept, count in dept_counts.head(15).items():
        pct = count / len(df) * 100
        print(f"  {dept}: {count:,} ({pct:.1f}%)")
    
    # Filter departments with sufficient data for analysis
    min_dept_responses = 30
    valid_departments = dept_counts[dept_counts >= min_dept_responses].index
    df_dept = df[df['department'].isin(valid_departments)].copy()
    
    print(f"\nAnalyzing {len(valid_departments)} departments with {min_dept_responses}+ responses each")
    print(f"Total responses in analysis: {len(df_dept):,}")
    
    # Department statistics
    dept_stats = df_dept.groupby('department').agg({
        'overall_culture_score': ['count', 'mean', 'std', 'median'],
        **{col: 'mean' for col in section_score_cols}
    }).round(3)
    
    print("\nTop 10 Departments by Culture Score:")
    dept_means = dept_stats['overall_culture_score']['mean'].sort_values(ascending=False)
    print(dept_means.head(10))
    
    print("\nBottom 10 Departments by Culture Score:")
    print(dept_means.tail(10))
    
    # Statistical analysis
    if len(valid_departments) >= 3:
        dept_groups = [df_dept[df_dept['department'] == dept]['overall_culture_score'].dropna() 
                      for dept in valid_departments]
        dept_groups = [group for group in dept_groups if len(group) >= 10]
        
        if len(dept_groups) >= 3:
            f_stat, p_value = f_oneway(*dept_groups)
            print(f"\nDepartment ANOVA: F={f_stat:.3f}, p={p_value:.3e}")
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            print(f"Statistical significance: {significance}")
    
    # Create comprehensive department visualization
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            'Top 15 Departments by Culture Score',
            'Department Response Counts',
            'Department-Section Heatmap (Top 10)',
            'Department-Domain Distribution',
            'Department Score Distribution',
            'Department Variability Analysis'
        ],
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "xy"}, {"type": "bar"}],
               [{"type": "box"}, {"type": "scatter"}]]
    )
    
    # Top departments by culture score
    top_15_depts = dept_means.head(15)
    fig.add_trace(
        go.Bar(
            x=top_15_depts.values,
            y=top_15_depts.index,
            orientation='h',
            marker_color='green',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Department response counts
    top_15_counts = dept_counts.head(15)
    fig.add_trace(
        go.Bar(
            x=top_15_counts.index,
            y=top_15_counts.values,
            marker_color='lightblue',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Department-section heatmap for top 10 departments
    top_10_depts = dept_means.head(10).index
    dept_section_top10 = df_dept[df_dept['department'].isin(top_10_depts)].groupby('department')[section_score_cols].mean()
    
    z_values = dept_section_top10.values
    x_labels = [col.replace('_score', '') for col in section_score_cols]
    y_labels = dept_section_top10.index
    
    fig.add_trace(
        go.Heatmap(
            z=z_values,
            x=x_labels,
            y=y_labels,
            colorscale='RdYlBu_r',
            showscale=True
        ),
        row=2, col=1
    )
    
    # Department-domain distribution
    dept_domain_counts = pd.crosstab(df_dept['department'], df_dept['domain'])
    top_depts_for_domain = dept_means.head(8).index  # Top 8 for visibility
    
    for domain in domains:
        if domain in dept_domain_counts.columns:
            domain_data = dept_domain_counts.loc[top_depts_for_domain, domain]
            fig.add_trace(
                go.Bar(
                    x=domain_data.index,
                    y=domain_data.values,
                    name=domain,
                    marker_color=domain_colors[domain],
                    showlegend=(domain == domains[0])
                ),
                row=2, col=2
            )
    
    # Department score distributions (top 8)
    top_8_depts = dept_means.head(8).index
    colors_dept = px.colors.qualitative.Set1
    for i, dept in enumerate(top_8_depts):
        dept_data = df_dept[df_dept['department'] == dept]['overall_culture_score']
        fig.add_trace(
            go.Box(
                y=dept_data,
                name=dept[:20],  # Truncate long names
                marker_color=colors_dept[i % len(colors_dept)],
                showlegend=False
            ),
            row=3, col=1
        )
    
    # Department variability analysis
    dept_means_all = df_dept.groupby('department')['overall_culture_score'].mean()
    dept_stds = df_dept.groupby('department')['overall_culture_score'].std()
    dept_counts_scatter = df_dept.groupby('department').size()
    
    # Create scatter plot of mean vs std
    fig.add_trace(
        go.Scatter(
            x=dept_means_all.values,
            y=dept_stds.values,
            mode='markers',
            text=dept_means_all.index,
            marker=dict(
                size=dept_counts_scatter.values/10,
                color=dept_means_all.values,
                colorscale='RdYlBu_r',
                showscale=False
            ),
            showlegend=False
        ),
        row=3, col=2
    )
    
    fig.update_layout(height=1400, title_text="Department Analysis Dashboard")
    fig.show()
    
    # Department insights
    print("\nDepartment Analysis Insights:")
    print(f"Best performing department: {dept_means.index[0]} ({dept_means.iloc[0]:.2f})")
    print(f"Worst performing department: {dept_means.index[-1]} ({dept_means.iloc[-1]:.2f})")
    print(f"Department score range: {dept_means.iloc[0] - dept_means.iloc[-1]:.2f} points")
    
    # Identify departments needing attention
    print(f"\nDepartments needing immediate attention (bottom 10%):")
    bottom_10_pct = int(len(dept_means) * 0.1)
    worst_departments = dept_means.tail(max(3, bottom_10_pct))
    for dept, score in worst_departments.items():
        count = dept_counts[dept]
        print(f"  {dept}: {score:.2f} (n={count})")
    
    # Cross-domain department comparison
    print("\nCross-domain department analysis:")
    common_departments = []
    for dept in valid_departments:
        dept_domains = df_dept[df_dept['department'] == dept]['domain'].nunique()
        if dept_domains >= 2:  # Department exists in multiple domains
            common_departments.append(dept)
    
    if common_departments:
        print(f"Departments present in multiple domains: {len(common_departments)}")
        for dept in common_departments[:5]:  # Show top 5
            dept_domain_scores = df_dept[df_dept['department'] == dept].groupby('domain')['overall_culture_score'].mean()
            print(f"  {dept}: {dict(dept_domain_scores.round(2))}")

else:
    print("Department analysis skipped - data not available")


# In[40]:


# 6.3 Supervision Analysis (Q33)
print("=== SUPERVISION ANALYSIS ===")

# Check for supervision data
supervision_cols = ['q33', 'supervises_others', 'management', 'supervisor']
supervision_col = None
for col in supervision_cols:
    if col in df.columns:
        supervision_col = col
        break

if supervision_col:
    print(f"Using supervision column: {supervision_col}")
    
    # Clean supervision data
    supervision_counts = df[supervision_col].value_counts()
    print(f"Supervision distribution:")
    for sup, count in supervision_counts.items():
        pct = count / len(df) * 100
        print(f"  {sup}: {count:,} ({pct:.1f}%)")
    
    # Standardize supervision categories
    def standardize_supervision(value):
        if pd.isna(value):
            return None
        value_str = str(value).lower()
        if any(term in value_str for term in ['yes', 'true', '1', 'supervisor', 'manager']):
            return 'Supervisor'
        elif any(term in value_str for term in ['no', 'false', '0', 'individual']):
            return 'Non-Supervisor'
        else:
            return value
    
    df['supervision_clean'] = df[supervision_col].apply(standardize_supervision)
    supervision_clean_counts = df['supervision_clean'].value_counts()
    print(f"\nStandardized supervision distribution:")
    print(supervision_clean_counts)
    
    # Filter for analysis
    valid_supervision = supervision_clean_counts.index.dropna()
    df_supervision = df[df['supervision_clean'].isin(valid_supervision)].copy()
    
    print(f"\nAnalyzing {len(valid_supervision)} supervision categories")
    print(f"Total responses in analysis: {len(df_supervision):,}")
    
    # Statistical analysis
    supervision_stats = df_supervision.groupby('supervision_clean').agg({
        'overall_culture_score': ['count', 'mean', 'std', 'median'],
        **{col: 'mean' for col in section_score_cols}
    }).round(3)
    
    print("\nCulture Scores by Supervision Status:")
    print(supervision_stats['overall_culture_score'])
    
    # Statistical testing
    if len(valid_supervision) >= 2:
        supervision_groups = [df_supervision[df_supervision['supervision_clean'] == sup]['overall_culture_score'].dropna() 
                             for sup in valid_supervision]
        
        if len(valid_supervision) == 2:
            # T-test for two groups
            t_stat, p_value = stats.ttest_ind(supervision_groups[0], supervision_groups[1])
            print(f"\nSupervision T-test: t={t_stat:.3f}, p={p_value:.3e}")
            
            # Cohen's d effect size
            pooled_std = np.sqrt(((len(supervision_groups[0])-1)*supervision_groups[0].std()**2 + 
                                 (len(supervision_groups[1])-1)*supervision_groups[1].std()**2) / 
                                (len(supervision_groups[0])+len(supervision_groups[1])-2))
            cohens_d = (supervision_groups[0].mean() - supervision_groups[1].mean()) / pooled_std
            print(f"Cohen's d effect size: {cohens_d:.3f}")
        else:
            # ANOVA for multiple groups
            f_stat, p_value = f_oneway(*supervision_groups)
            print(f"\nSupervision ANOVA: F={f_stat:.3f}, p={p_value:.3e}")
        
        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        print(f"Statistical significance: {significance}")
    
    # Comprehensive supervision visualization
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            'Culture Scores by Supervision Status',
            'Supervision Distribution',
            'Supervision-Section Analysis',
            'Supervision-Domain Interaction',
            'Supervision-Position Level',
            'Management Burden Analysis'
        ],
        specs=[[{"type": "violin"}, {"type": "pie"}],
               [{"type": "xy"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "box"}]]
    )
    
    # Violin plots by supervision status
    colors_supervision = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    for i, sup in enumerate(valid_supervision):
        sup_data = df_supervision[df_supervision['supervision_clean'] == sup]['overall_culture_score']
        fig.add_trace(
            go.Violin(
                y=sup_data,
                name=sup,
                box_visible=True,
                meanline_visible=True,
                fillcolor=colors_supervision[i % len(colors_supervision)],
                opacity=0.6,
                showlegend=False
            ),
            row=1, col=1
        )
    
    # Supervision distribution pie chart
    fig.add_trace(
        go.Pie(
            labels=supervision_clean_counts.index,
            values=supervision_clean_counts.values,
            name="Supervision Distribution"
        ),
        row=1, col=2
    )
    
    # Supervision-section analysis
    supervision_section_means = df_supervision.groupby('supervision_clean')[section_score_cols].mean()
    
    # Create grouped bar chart
    for i, section in enumerate(section_score_cols):
        section_name = section.replace('_score', '')
        for j, sup in enumerate(valid_supervision):
            fig.add_trace(
                go.Bar(
                    x=[section_name],
                    y=[supervision_section_means.loc[sup, section]],
                    name=sup,
                    marker_color=colors_supervision[j % len(colors_supervision)],
                    showlegend=(i == 0),
                    offsetgroup=sup
                ),
                row=2, col=1
            )
    
    # Supervision-domain interaction
    supervision_domain_means = df_supervision.groupby(['supervision_clean', 'domain'])['overall_culture_score'].mean().unstack()
    for i, sup in enumerate(valid_supervision):
        if sup in supervision_domain_means.index:
            fig.add_trace(
                go.Bar(
                    x=supervision_domain_means.columns,
                    y=supervision_domain_means.loc[sup],
                    name=f"{sup}",
                    marker_color=colors_supervision[i % len(colors_supervision)],
                    showlegend=False
                ),
                row=2, col=2
            )
    
    # Supervision-position level analysis (if position data available)
    if position_col and position_col in df_supervision.columns:
        supervision_position_means = df_supervision.groupby(['supervision_clean', position_col])['overall_culture_score'].mean().unstack()
        
        x_positions = []
        y_values = []
        colors_bars = []
        text_labels = []
        
        for i, sup in enumerate(valid_supervision):
            if sup in supervision_position_means.index:
                for j, pos in enumerate(supervision_position_means.columns):
                    if not pd.isna(supervision_position_means.loc[sup, pos]):
                        x_positions.append(f"{sup}\n{pos}")
                        y_values.append(supervision_position_means.loc[sup, pos])
                        colors_bars.append(colors_supervision[i % len(colors_supervision)])
                        text_labels.append(f"{supervision_position_means.loc[sup, pos]:.2f}")
        
        fig.add_trace(
            go.Bar(
                x=x_positions,
                y=y_values,
                text=text_labels,
                textposition='auto',
                marker_color=colors_bars,
                showlegend=False
            ),
            row=3, col=1
        )
    
    # Management burden analysis (if supervisors exist)
    if 'Supervisor' in valid_supervision:
        # Analyze section scores for supervisors vs non-supervisors
        supervisor_data = df_supervision[df_supervision['supervision_clean'] == 'Supervisor']
        non_supervisor_data = df_supervision[df_supervision['supervision_clean'] == 'Non-Supervisor']
        
        section_differences = []
        section_names_clean = []
        
        for section in section_score_cols:
            if len(supervisor_data) > 0 and len(non_supervisor_data) > 0:
                sup_mean = supervisor_data[section].mean()
                non_sup_mean = non_supervisor_data[section].mean()
                difference = sup_mean - non_sup_mean
                section_differences.append(difference)
                section_names_clean.append(section.replace('_score', ''))
        
        fig.add_trace(
            go.Bar(
                x=section_names_clean,
                y=section_differences,
                marker_color=['red' if x > 0 else 'blue' for x in section_differences],
                text=[f"{x:+.2f}" for x in section_differences],
                textposition='auto',
                showlegend=False
            ),
            row=3, col=2
        )
    
    fig.update_layout(height=1200, title_text="Supervision Analysis Dashboard")
    fig.show()
    
    # Supervision insights
    print("\nSupervision Analysis Insights:")
    supervision_means = df_supervision.groupby('supervision_clean')['overall_culture_score'].mean().sort_values()
    print(f"Culture scores by supervision status:")
    for sup, score in supervision_means.items():
        count = supervision_clean_counts[sup]
        print(f"  {sup}: {score:.2f} (n={count:,})")
    
    # Section-specific supervision impacts
    if 'Supervisor' in valid_supervision and 'Non-Supervisor' in valid_supervision:
        print("\nSection-specific supervision impacts:")
        supervisor_section_means = df_supervision[df_supervision['supervision_clean'] == 'Supervisor'][section_score_cols].mean()
        non_supervisor_section_means = df_supervision[df_supervision['supervision_clean'] == 'Non-Supervisor'][section_score_cols].mean()
        
        for section in section_score_cols:
            difference = supervisor_section_means[section] - non_supervisor_section_means[section]
            section_name = section.replace('_score', '')
            impact = "Higher" if difference > 0 else "Lower"
            print(f"  {section_name}: Supervisors score {difference:+.2f} points {impact.lower()}")

else:
    print("Supervision analysis skipped - data not available")


# ## 7. INTERSECTIONAL ANALYSIS
# 
# Multi-dimensional demographic analysis examining how combinations of demographic factors interact to influence workplace culture experiences.

# In[41]:


# 7.1 Age × Gender Intersectional Analysis
print("=== AGE × GENDER INTERSECTIONAL ANALYSIS ===")

# Check for both age and gender data
age_available = age_col is not None and age_col in df.columns
gender_available = gender_col is not None and gender_col in df.columns

if age_available and gender_available:
    print(f"Analyzing intersection of {age_col} and {gender_col}")
    
    # Create intersectional dataset
    df_age_gender = df[[age_col, gender_col, 'overall_culture_score'] + section_score_cols + ['domain']].dropna()
    
    # Filter for sufficient group sizes
    age_gender_counts = df_age_gender.groupby([age_col, gender_col]).size()
    valid_combinations = age_gender_counts[age_gender_counts >= 15].index  # Minimum 15 per group
    
    print(f"Valid age-gender combinations (n≥15): {len(valid_combinations)}")
    
    if len(valid_combinations) > 0:
        # Create filtered dataset
        df_ag_filtered = df_age_gender[df_age_gender.set_index([age_col, gender_col]).index.isin(valid_combinations)].copy()
        
        # Statistical analysis
        age_gender_stats = df_ag_filtered.groupby([age_col, gender_col]).agg({
            'overall_culture_score': ['count', 'mean', 'std'],
            **{col: 'mean' for col in section_score_cols}
        }).round(3)
        
        print("\nAge-Gender Intersection Culture Scores:")
        print(age_gender_stats['overall_culture_score']['mean'].unstack())
        
        # Two-way ANOVA
        from scipy.stats import f_oneway
        
        # Prepare data for two-way ANOVA (simplified approach)
        age_gender_means = df_ag_filtered.groupby([age_col, gender_col])['overall_culture_score'].mean()
        
        print(f"\nAge-Gender intersection effects:")
        for age in df_ag_filtered[age_col].unique():
            for gender in df_ag_filtered[gender_col].unique():
                if (age, gender) in age_gender_means.index:
                    score = age_gender_means[(age, gender)]
                    count = df_ag_filtered[(df_ag_filtered[age_col] == age) & (df_ag_filtered[gender_col] == gender)].shape[0]
                    print(f"  {age} × {gender}: {score:.2f} (n={count})")
        
        # Visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Age-Gender Heatmap',
                'Age-Gender-Domain Interaction',
                'Section Scores by Age-Gender',
                'Age-Gender Distribution'
            ],
            specs=[[{"type": "xy"}, {"type": "bar"}],
                   [{"type": "xy"}, {"type": "bar"}]]
        )
        
        # Age-Gender heatmap
        age_gender_pivot = df_ag_filtered.pivot_table(
            values='overall_culture_score', 
            index=gender_col, 
            columns=age_col, 
            aggfunc='mean'
        )
        
        fig.add_trace(
            go.Heatmap(
                z=age_gender_pivot.values,
                x=age_gender_pivot.columns,
                y=age_gender_pivot.index,
                colorscale='RdYlBu_r',
                text=np.round(age_gender_pivot.values, 2),
                texttemplate="%{text}",
                textfont={"size": 10},
                showscale=True
            ),
            row=1, col=1
        )
        
        # Age-Gender-Domain interaction
        age_gender_domain = df_ag_filtered.groupby([age_col, gender_col, 'domain'])['overall_culture_score'].mean().reset_index()
        
        for domain in domains:
            domain_data = age_gender_domain[age_gender_domain['domain'] == domain]
            if len(domain_data) > 0:
                fig.add_trace(
                    go.Bar(
                        x=[f"{row[age_col]}-{row[gender_col]}" for _, row in domain_data.iterrows()],
                        y=domain_data['overall_culture_score'],
                        name=domain,
                        marker_color=domain_colors[domain],
                        showlegend=(domain == domains[0])
                    ),
                    row=1, col=2
                )
        
        # Section scores by age-gender (top combinations)
        top_combinations = age_gender_counts.nlargest(6).index
        colors_combinations = px.colors.qualitative.Set2
        
        for i, section in enumerate(section_score_cols[:3]):  # Show first 3 sections
            section_name = section.replace('_score', '')
            
            for j, (age, gender) in enumerate(top_combinations):
                if j < 4:  # Limit to 4 combinations for readability
                    combination_data = df_ag_filtered[
                        (df_ag_filtered[age_col] == age) & 
                        (df_ag_filtered[gender_col] == gender)
                    ]
                    if len(combination_data) > 0:
                        score = combination_data[section].mean()
                        fig.add_trace(
                            go.Bar(
                                x=[section_name],
                                y=[score],
                                name=f"{age}-{gender}",
                                marker_color=colors_combinations[j % len(colors_combinations)],
                                showlegend=(i == 0),
                                offsetgroup=f"{age}-{gender}"
                            ),
                            row=2, col=1
                        )
        
        # Age-Gender distribution
        age_gender_dist = df_ag_filtered.groupby([age_col, gender_col]).size().reset_index(name='count')
        
        fig.add_trace(
            go.Bar(
                x=[f"{row[age_col]}-{row[gender_col]}" for _, row in age_gender_dist.iterrows()],
                y=age_gender_dist['count'],
                marker_color='lightgreen',
                showlegend=False
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=1000, title_text="Age × Gender Intersectional Analysis")
        fig.show()
        
        # Key insights
        print("\nAge × Gender Key Insights:")
        overall_mean = df_ag_filtered['overall_culture_score'].mean()
        
        # Find most/least advantaged intersections
        age_gender_means_sorted = age_gender_means.sort_values()
        print(f"Most challenging intersection: {age_gender_means_sorted.index[0]} ({age_gender_means_sorted.iloc[0]:.2f})")
        print(f"Best performing intersection: {age_gender_means_sorted.index[-1]} ({age_gender_means_sorted.iloc[-1]:.2f})")
        print(f"Intersection gap: {age_gender_means_sorted.iloc[-1] - age_gender_means_sorted.iloc[0]:.2f} points")
        
        # Check for specific patterns
        if len(age_gender_means_sorted) >= 4:
            below_average = age_gender_means_sorted[age_gender_means_sorted < overall_mean]
            above_average = age_gender_means_sorted[age_gender_means_sorted >= overall_mean]
            
            print(f"\nBelow average intersections: {len(below_average)}")
            for intersection, score in below_average.items():
                print(f"  {intersection}: {score:.2f}")
            
            print(f"\nAbove average intersections: {len(above_average)}")
            for intersection, score in above_average.items():
                print(f"  {intersection}: {score:.2f}")

else:
    print("Age × Gender intersectional analysis skipped - insufficient data")


# In[42]:


# 7.2 Position × Department Intersectional Analysis
print("=== POSITION × DEPARTMENT INTERSECTIONAL ANALYSIS ===")

# Check for both position and department data
position_available = position_col is not None and position_col in df.columns
department_available = 'department' in df.columns

if position_available and department_available:
    print(f"Analyzing intersection of {position_col} and department")
    
    # Create intersectional dataset
    df_pos_dept = df[[position_col, 'department', 'overall_culture_score'] + section_score_cols + ['domain']].dropna()
    
    # Filter for sufficient group sizes
    pos_dept_counts = df_pos_dept.groupby([position_col, 'department']).size()
    valid_combinations = pos_dept_counts[pos_dept_counts >= 20].index  # Minimum 20 per group
    
    print(f"Valid position-department combinations (n≥20): {len(valid_combinations)}")
    
    if len(valid_combinations) > 0:
        # Create filtered dataset
        df_pd_filtered = df_pos_dept[df_pos_dept.set_index([position_col, 'department']).index.isin(valid_combinations)].copy()
        
        # Statistical analysis
        pos_dept_stats = df_pd_filtered.groupby([position_col, 'department']).agg({
            'overall_culture_score': ['count', 'mean', 'std'],
            **{col: 'mean' for col in section_score_cols}
        }).round(3)
        
        print("\nPosition-Department Intersection Culture Scores (Top 10):")
        pos_dept_means = pos_dept_stats['overall_culture_score']['mean'].sort_values(ascending=False)
        print(pos_dept_means.head(10))
        
        print("\nWorst Position-Department Combinations:")
        print(pos_dept_means.tail(5))
        
        # Visualization
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Position-Department Heatmap (Top Departments)',
                'Top 15 Position-Department Combinations',
                'Position-Department-Domain Analysis',
                'Cross-Department Position Comparison',
                'Department Leadership Analysis',
                'Position-Department Response Distribution'
            ],
            specs=[[{"type": "xy"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "box"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Position-Department heatmap (top departments by response count)
        top_departments = df_pd_filtered['department'].value_counts().head(8).index
        pos_dept_pivot = df_pd_filtered[df_pd_filtered['department'].isin(top_departments)].pivot_table(
            values='overall_culture_score',
            index='department',
            columns=position_col,
            aggfunc='mean'
        )
        
        fig.add_trace(
            go.Heatmap(
                z=pos_dept_pivot.values,
                x=pos_dept_pivot.columns,
                y=pos_dept_pivot.index,
                colorscale='RdYlBu_r',
                text=np.round(pos_dept_pivot.values, 2),
                texttemplate="%{text}",
                textfont={"size": 8},
                showscale=True
            ),
            row=1, col=1
        )
        
        # Top 15 position-department combinations
        top_15_combinations = pos_dept_means.head(15)
        combination_labels = [f"{pos}\n{dept}" for pos, dept in top_15_combinations.index]
        
        fig.add_trace(
            go.Bar(
                x=top_15_combinations.values,
                y=combination_labels,
                orientation='h',
                marker_color='green',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Position-Department-Domain analysis
        pos_dept_domain = df_pd_filtered.groupby([position_col, 'department', 'domain'])['overall_culture_score'].mean().reset_index()
        
        # Show top combinations by domain
        top_combos_by_domain = {}
        for domain in domains:
            domain_data = pos_dept_domain[pos_dept_domain['domain'] == domain]
            if len(domain_data) > 0:
                top_combo = domain_data.loc[domain_data['overall_culture_score'].idxmax()]
                top_combos_by_domain[domain] = (top_combo[position_col], top_combo['department'], top_combo['overall_culture_score'])
        
        domain_labels = []
        domain_scores = []
        domain_colors_list = []
        
        for domain, (pos, dept, score) in top_combos_by_domain.items():
            domain_labels.append(f"{domain}\n{pos}-{dept}")
            domain_scores.append(score)
            domain_colors_list.append(domain_colors[domain])
        
        fig.add_trace(
            go.Bar(
                x=domain_labels,
                y=domain_scores,
                marker_color=domain_colors_list,
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Cross-department position comparison (same position across departments)
        common_positions = df_pd_filtered[position_col].value_counts().head(4).index
        colors_positions = px.colors.qualitative.Set1
        
        for i, pos in enumerate(common_positions):
            pos_data = df_pd_filtered[df_pd_filtered[position_col] == pos]
            dept_scores = pos_data.groupby('department')['overall_culture_score'].mean().sort_values(ascending=False).head(5)
            
            for dept in dept_scores.index:
                dept_data = pos_data[pos_data['department'] == dept]['overall_culture_score']
                fig.add_trace(
                    go.Box(
                        y=dept_data,
                        name=f"{pos[:15]}",  # Truncate position name
                        marker_color=colors_positions[i % len(colors_positions)],
                        showlegend=(dept == dept_scores.index[0])  # Show legend only once per position
                    ),
                    row=2, col=2
                )
        
        # Department leadership analysis
        leadership_keywords = ['Senior', 'Executive', 'Manager', 'Director', 'VP', 'Lead']
        df_pd_filtered['is_leadership'] = df_pd_filtered[position_col].apply(
            lambda x: any(keyword in str(x) for keyword in leadership_keywords)
        )
        
        dept_leadership = df_pd_filtered.groupby(['department', 'is_leadership'])['overall_culture_score'].mean().unstack()
        top_depts_leadership = df_pd_filtered['department'].value_counts().head(8).index
        
        if True in dept_leadership.columns and False in dept_leadership.columns:
            leadership_diff = dept_leadership[True] - dept_leadership[False]
            leadership_diff = leadership_diff.reindex(top_depts_leadership).dropna()
            
            fig.add_trace(
                go.Bar(
                    x=leadership_diff.index,
                    y=leadership_diff.values,
                    marker_color=['red' if x > 0 else 'blue' for x in leadership_diff.values],
                    text=[f"{x:+.2f}" for x in leadership_diff.values],
                    textposition='auto',
                    showlegend=False
                ),
                row=3, col=1
            )
        
        # Position-Department response distribution
        pos_dept_dist = df_pd_filtered.groupby([position_col, 'department']).size().reset_index(name='count')
        top_combos_dist = pos_dept_dist.nlargest(12, 'count')
        
        combo_labels_dist = [f"{row[position_col][:10]}\n{row['department'][:15]}" for _, row in top_combos_dist.iterrows()]
        
        fig.add_trace(
            go.Bar(
                x=combo_labels_dist,
                y=top_combos_dist['count'],
                marker_color='lightcoral',
                showlegend=False
            ),
            row=3, col=2
        )
        
        fig.update_layout(height=1400, title_text="Position × Department Intersectional Analysis")
        fig.show()
        
        # Key insights
        print("\nPosition × Department Key Insights:")
        
        # Best and worst combinations
        print(f"Best performing combination: {pos_dept_means.index[0]} ({pos_dept_means.iloc[0]:.2f})")
        print(f"Worst performing combination: {pos_dept_means.index[-1]} ({pos_dept_means.iloc[-1]:.2f})")
        print(f"Position-Department gap: {pos_dept_means.iloc[0] - pos_dept_means.iloc[-1]:.2f} points")
        
        # Department consistency analysis
        print("\nDepartment consistency (standard deviation of positions within departments):")
        dept_consistency = df_pd_filtered.groupby('department')[position_col].apply(
            lambda x: df_pd_filtered[df_pd_filtered['department'] == x.name]['overall_culture_score'].std()
        ).sort_values()
        
        print("Most consistent departments (low variation across positions):")
        for dept, std in dept_consistency.head(5).items():
            print(f"  {dept}: {std:.2f}")
        
        print("\nLeast consistent departments (high variation across positions):")
        for dept, std in dept_consistency.tail(5).items():
            print(f"  {dept}: {std:.2f}")
        
        # Position mobility analysis
        print("\nPosition performance across departments:")
        for pos in common_positions[:3]:
            pos_dept_scores = df_pd_filtered[df_pd_filtered[position_col] == pos].groupby('department')['overall_culture_score'].mean()
            if len(pos_dept_scores) >= 3:
                best_dept = pos_dept_scores.idxmax()
                worst_dept = pos_dept_scores.idxmin()
                range_score = pos_dept_scores.max() - pos_dept_scores.min()
                print(f"  {pos}: Best in {best_dept} ({pos_dept_scores.max():.2f}), Worst in {worst_dept} ({pos_dept_scores.min():.2f}), Range: {range_score:.2f}")

else:
    print("Position × Department intersectional analysis skipped - insufficient data")


# In[43]:


# 7.3 Multi-dimensional Demographic Clustering Analysis
print("=== MULTI-DIMENSIONAL DEMOGRAPHIC CLUSTERING ===")

# Prepare demographic features for clustering
demographic_features = []
demographic_feature_names = []

# Add available demographic variables
if age_col and age_col in df.columns:
    df['age_encoded'] = pd.Categorical(df[age_col]).codes
    demographic_features.append('age_encoded')
    demographic_feature_names.append('Age')

if gender_col and gender_col in df.columns:
    df['gender_encoded'] = pd.Categorical(df[gender_col]).codes
    demographic_features.append('gender_encoded')
    demographic_feature_names.append('Gender')

if race_col and race_col in df.columns:
    df['race_encoded'] = pd.Categorical(df[race_col]).codes
    demographic_features.append('race_encoded')
    demographic_feature_names.append('Race/Ethnicity')

if education_col and education_col in df.columns:
    df['education_encoded'] = pd.Categorical(df[education_col]).codes
    demographic_features.append('education_encoded')
    demographic_feature_names.append('Education')

if position_col and position_col in df.columns:
    df['position_encoded'] = pd.Categorical(df[position_col]).codes
    demographic_features.append('position_encoded')
    demographic_feature_names.append('Position Level')

if 'department' in df.columns:
    df['department_encoded'] = pd.Categorical(df['department']).codes
    demographic_features.append('department_encoded')
    demographic_feature_names.append('Department')

if supervision_col and supervision_col in df.columns:
    df['supervision_encoded'] = pd.Categorical(df[supervision_col]).codes
    demographic_features.append('supervision_encoded')
    demographic_feature_names.append('Supervision')

df['domain_encoded'] = pd.Categorical(df['domain']).codes
demographic_features.append('domain_encoded')
demographic_feature_names.append('Domain')

print(f"Available demographic features for clustering: {len(demographic_features)}")
print(f"Features: {demographic_feature_names}")

if len(demographic_features) >= 3:
    # Create clustering dataset
    cluster_data = df[demographic_features + ['overall_culture_score'] + section_score_cols].dropna()
    
    print(f"Clustering dataset: {len(cluster_data):,} complete records")
    
    if len(cluster_data) >= 100:
        # Prepare features for clustering
        X_demographic = cluster_data[demographic_features].values
        y_culture = cluster_data['overall_culture_score'].values
        
        # Standardize features
        scaler_demo = StandardScaler()
        X_demographic_scaled = scaler_demo.fit_transform(X_demographic)
        
        # Determine optimal number of clusters
        max_clusters = min(8, len(cluster_data) // 50)
        silhouette_scores_demo = []
        inertias_demo = []
        
        for k in range(2, max_clusters + 1):
            kmeans_demo = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels_demo = kmeans_demo.fit_predict(X_demographic_scaled)
            silhouette_scores_demo.append(silhouette_score(X_demographic_scaled, cluster_labels_demo))
            inertias_demo.append(kmeans_demo.inertia_)
        
        # Select optimal k
        optimal_k_demo = np.argmax(silhouette_scores_demo) + 2
        print(f"Optimal number of demographic clusters: {optimal_k_demo}")
        
        # Perform final clustering
        final_kmeans_demo = KMeans(n_clusters=optimal_k_demo, random_state=42, n_init=10)
        cluster_labels_final = final_kmeans_demo.fit_predict(X_demographic_scaled)
        
        # Add cluster labels to data
        cluster_data['demographic_cluster'] = cluster_labels_final
        
        # Analyze clusters
        cluster_analysis = cluster_data.groupby('demographic_cluster').agg({
            'overall_culture_score': ['count', 'mean', 'std'],
            **{col: 'mean' for col in section_score_cols}
        }).round(3)
        
        print("\nDemographic Cluster Analysis:")
        print(cluster_analysis['overall_culture_score'])
        
        # Visualization
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Cluster Quality Metrics',
                'Cluster Culture Score Distributions',
                'Cluster Demographic Composition',
                'Cluster Section Score Profiles',
                'Cluster PCA Visualization',
                'Cluster Domain Distribution'
            ],
            specs=[[{"type": "xy"}, {"type": "box"}],
                   [{"type": "xy"}, {"type": "xy"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Cluster quality metrics
        fig.add_trace(
            go.Scatter(
                x=list(range(2, max_clusters + 1)),
                y=silhouette_scores_demo,
                mode='lines+markers',
                name='Silhouette Score',
                yaxis='y1'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=list(range(2, max_clusters + 1)),
                y=inertias_demo,
                mode='lines+markers',
                name='Inertia',
                yaxis='y2',
                line=dict(dash='dash')
            ),
            row=1, col=1
        )
        
        # Cluster culture score distributions
        colors_clusters = px.colors.qualitative.Plotly
        for cluster_id in range(optimal_k_demo):
            cluster_scores = cluster_data[cluster_data['demographic_cluster'] == cluster_id]['overall_culture_score']
            fig.add_trace(
                go.Box(
                    y=cluster_scores,
                    name=f'Cluster {cluster_id}',
                    marker_color=colors_clusters[cluster_id % len(colors_clusters)],
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # Cluster demographic composition (heatmap)
        cluster_demographics = pd.DataFrame()
        for i, feature in enumerate(demographic_features):
            if feature in cluster_data.columns:
                cluster_feature_means = cluster_data.groupby('demographic_cluster')[feature].mean()
                cluster_demographics[demographic_feature_names[i]] = cluster_feature_means
        
        if not cluster_demographics.empty:
            fig.add_trace(
                go.Heatmap(
                    z=cluster_demographics.values,
                    x=cluster_demographics.columns,
                    y=[f'Cluster {i}' for i in cluster_demographics.index],
                    colorscale='Viridis',
                    showscale=True
                ),
                row=2, col=1
            )
        
        # Cluster section score profiles
        cluster_sections = cluster_data.groupby('demographic_cluster')[section_score_cols].mean()
        
        for cluster_id in range(optimal_k_demo):
            fig.add_trace(
                go.Scatter(
                    x=[col.replace('_score', '') for col in section_score_cols],
                    y=cluster_sections.loc[cluster_id],
                    mode='lines+markers',
                    name=f'Cluster {cluster_id}',
                    marker_color=colors_clusters[cluster_id % len(colors_clusters)],
                    showlegend=False
                ),
                row=2, col=2
            )
        
        # PCA visualization of clusters
        pca_demo = PCA(n_components=2)
        X_pca_demo = pca_demo.fit_transform(X_demographic_scaled)
        
        for cluster_id in range(optimal_k_demo):
            cluster_mask = cluster_labels_final == cluster_id
            fig.add_trace(
                go.Scatter(
                    x=X_pca_demo[cluster_mask, 0],
                    y=X_pca_demo[cluster_mask, 1],
                    mode='markers',
                    name=f'Cluster {cluster_id}',
                    marker=dict(
                        color=colors_clusters[cluster_id % len(colors_clusters)],
                        size=4,
                        opacity=0.6
                    ),
                    showlegend=False
                ),
                row=3, col=1
            )
        
        # Cluster domain distribution
        cluster_domain_dist = pd.crosstab(cluster_data['demographic_cluster'], cluster_data['domain'], normalize='index')
        
        for domain in domains:
            if domain in cluster_domain_dist.columns:
                fig.add_trace(
                    go.Bar(
                        x=[f'Cluster {i}' for i in range(optimal_k_demo)],
                        y=cluster_domain_dist[domain],
                        name=domain,
                        marker_color=domain_colors[domain],
                        showlegend=False
                    ),
                    row=3, col=2
                )
        
        fig.update_layout(height=1400, title_text="Multi-dimensional Demographic Clustering Analysis")
        fig.show()
        
        # Key insights
        print("\nMulti-dimensional Clustering Insights:")
        
        # Cluster culture score ranking
        cluster_means = cluster_data.groupby('demographic_cluster')['overall_culture_score'].mean().sort_values()
        print(f"Cluster culture score ranking:")
        for cluster_id, score in cluster_means.items():
            count = len(cluster_data[cluster_data['demographic_cluster'] == cluster_id])
            print(f"  Cluster {cluster_id}: {score:.2f} (n={count:,})")
        
        # Identify cluster characteristics
        print(f"\nCluster characteristics:")
        for cluster_id in range(optimal_k_demo):
            cluster_subset = cluster_data[cluster_data['demographic_cluster'] == cluster_id]
            
            # Find most common values in each demographic dimension
            characteristics = []
            for i, feature in enumerate(demographic_features):
                if feature in cluster_subset.columns:
                    # Get most common encoded value and map back to original
                    most_common_encoded = cluster_subset[feature].mode().iloc[0] if len(cluster_subset[feature].mode()) > 0 else 0
                    
                    # Map back to original values
                    original_col = feature.replace('_encoded', '')
                    if original_col in df.columns:
                        original_values = df[original_col].dropna().unique()
                        if most_common_encoded < len(original_values):
                            characteristics.append(f"{demographic_feature_names[i]}: {original_values[most_common_encoded]}")
            
            culture_score = cluster_means[cluster_id]
            print(f"\n  Cluster {cluster_id} (Score: {culture_score:.2f}):")
            for char in characteristics[:4]:  # Show top 4 characteristics
                print(f"    - {char}")
        
        # Statistical significance between clusters
        if optimal_k_demo >= 2:
            cluster_groups = [cluster_data[cluster_data['demographic_cluster'] == i]['overall_culture_score'] 
                            for i in range(optimal_k_demo)]
            f_stat, p_value = f_oneway(*cluster_groups)
            print(f"\nCluster differences ANOVA: F={f_stat:.3f}, p={p_value:.3e}")
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            print(f"Statistical significance: {significance}")

else:
    print("Multi-dimensional demographic clustering skipped - insufficient features")


# ## 8. DEMOGRAPHIC DISPARITY ANALYSIS
# 
# Statistical analysis of demographic inequities and gaps in workplace culture experiences with comprehensive disparity metrics and significance testing.

# In[44]:


# 8.1 Comprehensive Demographic Disparity Matrix
print("=== COMPREHENSIVE DEMOGRAPHIC DISPARITY ANALYSIS ===")

# Create disparity analysis framework
disparity_results = {}
demographic_columns = {}

# Collect available demographic columns
if age_col and age_col in df.columns:
    demographic_columns['Age'] = age_col
if gender_col and gender_col in df.columns:
    demographic_columns['Gender'] = gender_col
if race_col and race_col in df.columns:
    demographic_columns['Race/Ethnicity'] = race_col
if education_col and education_col in df.columns:
    demographic_columns['Education'] = education_col
if position_col and position_col in df.columns:
    demographic_columns['Position Level'] = position_col
if 'department' in df.columns:
    demographic_columns['Department'] = 'department'
if supervision_col and supervision_col in df.columns:
    demographic_columns['Supervision'] = supervision_col

demographic_columns['Domain'] = 'domain'

print(f"Analyzing disparities across {len(demographic_columns)} demographic dimensions")
print(f"Dimensions: {list(demographic_columns.keys())}")

# Function to calculate disparity metrics
def calculate_disparity_metrics(group_means, group_counts):
    """Calculate comprehensive disparity metrics"""
    if len(group_means) < 2:
        return {}
    
    metrics = {}
    
    # Basic disparity measures
    metrics['range'] = group_means.max() - group_means.min()
    metrics['coefficient_of_variation'] = group_means.std() / group_means.mean()
    metrics['max_group'] = group_means.idxmax()
    metrics['min_group'] = group_means.idxmin()
    metrics['max_score'] = group_means.max()
    metrics['min_score'] = group_means.min()
    
    # Relative disparity (compared to overall mean)
    overall_mean = (group_means * group_counts).sum() / group_counts.sum()
    metrics['relative_disparity'] = abs(group_means - overall_mean).max()
    
    # Gini coefficient (inequality measure)
    sorted_scores = np.sort(group_means.values)
    n = len(sorted_scores)
    cumsum = np.cumsum(sorted_scores)
    metrics['gini_coefficient'] = (n + 1 - 2 * sum((n + 1 - i) * sorted_scores[i-1] for i in range(1, n+1)) / cumsum[-1]) / n
    
    return metrics

# Analyze disparities for each demographic dimension
for demo_name, demo_col in demographic_columns.items():
    print(f"\n=== {demo_name.upper()} DISPARITY ANALYSIS ===")
    
    # Clean data for this demographic
    demo_data = df[df[demo_col].notna()].copy()
    
    if len(demo_data) == 0:
        print(f"No data available for {demo_name}")
        continue
    
    # Calculate group statistics
    group_stats = demo_data.groupby(demo_col).agg({
        'overall_culture_score': ['count', 'mean', 'std', 'median'],
        **{col: 'mean' for col in section_score_cols}
    })
    
    # Filter groups with sufficient size
    min_group_size = 30 if demo_name != 'Department' else 20
    valid_groups = group_stats['overall_culture_score']['count'] >= min_group_size
    
    if valid_groups.sum() < 2:
        print(f"Insufficient groups for {demo_name} analysis")
        continue
    
    # Get valid group data
    valid_group_data = group_stats[valid_groups]
    group_means = valid_group_data['overall_culture_score']['mean']
    group_counts = valid_group_data['overall_culture_score']['count']
    
    # Calculate disparity metrics
    disparity_metrics = calculate_disparity_metrics(group_means, group_counts)
    disparity_results[demo_name] = disparity_metrics
    
    print(f"Groups analyzed: {len(group_means)}")
    print(f"Disparity range: {disparity_metrics['range']:.3f} points")
    print(f"Best performing group: {disparity_metrics['max_group']} ({disparity_metrics['max_score']:.3f})")
    print(f"Worst performing group: {disparity_metrics['min_group']} ({disparity_metrics['min_score']:.3f})")
    print(f"Coefficient of variation: {disparity_metrics['coefficient_of_variation']:.3f}")
    print(f"Gini coefficient: {disparity_metrics['gini_coefficient']:.3f}")
    
    # Statistical significance testing
    demo_groups = []
    demo_group_names = []
    
    for group in group_means.index:
        group_data = demo_data[demo_data[demo_col] == group]['overall_culture_score'].dropna()
        if len(group_data) >= min_group_size:
            demo_groups.append(group_data)
            demo_group_names.append(group)
    
    if len(demo_groups) >= 2:
        # ANOVA test
        f_stat, p_value = f_oneway(*demo_groups)
        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        print(f"ANOVA: F={f_stat:.3f}, p={p_value:.3e} {significance}")
        
        # Effect size (eta-squared)
        total_n = sum(len(group) for group in demo_groups)
        overall_mean = np.concatenate(demo_groups).mean()
        
        # Between-group sum of squares
        ss_between = sum(len(group) * (group.mean() - overall_mean)**2 for group in demo_groups)
        # Total sum of squares
        ss_total = sum(np.sum((group - overall_mean)**2) for group in demo_groups)
        
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        print(f"Effect size (η²): {eta_squared:.3f}")
        
        disparity_results[demo_name]['f_statistic'] = f_stat
        disparity_results[demo_name]['p_value'] = p_value
        disparity_results[demo_name]['eta_squared'] = eta_squared
        disparity_results[demo_name]['significance'] = significance

# Create comprehensive disparity visualization
print(f"\n=== DISPARITY SUMMARY DASHBOARD ===")

if disparity_results:
    # Prepare data for visualization
    disparity_df = pd.DataFrame(disparity_results).T
    
    # Sort by disparity range
    disparity_df_sorted = disparity_df.sort_values('range', ascending=False)
    
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            'Disparity Range by Demographic',
            'Statistical Significance',
            'Effect Sizes (η²)',
            'Inequality Measures (Gini)',
            'Best vs Worst Performing Groups',
            'Coefficient of Variation'
        ],
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "bar"}]]
    )
    
    # Disparity range
    fig.add_trace(
        go.Bar(
            x=disparity_df_sorted.index,
            y=disparity_df_sorted['range'],
            marker_color='red',
            text=[f"{x:.2f}" for x in disparity_df_sorted['range']],
            textposition='auto',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Statistical significance
    significance_colors = []
    significance_values = []
    for idx in disparity_df_sorted.index:
        if 'p_value' in disparity_df_sorted.columns:
            p_val = disparity_df_sorted.loc[idx, 'p_value']
            if pd.notna(p_val):
                if p_val < 0.001:
                    significance_colors.append('darkred')
                    significance_values.append(-np.log10(p_val))
                elif p_val < 0.01:
                    significance_colors.append('red')
                    significance_values.append(-np.log10(p_val))
                elif p_val < 0.05:
                    significance_colors.append('orange')
                    significance_values.append(-np.log10(p_val))
                else:
                    significance_colors.append('lightgray')
                    significance_values.append(-np.log10(p_val))
            else:
                significance_colors.append('lightgray')
                significance_values.append(0)
        else:
            significance_colors.append('lightgray')
            significance_values.append(0)
    
    fig.add_trace(
        go.Bar(
            x=disparity_df_sorted.index,
            y=significance_values,
            marker_color=significance_colors,
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Effect sizes
    if 'eta_squared' in disparity_df_sorted.columns:
        eta_values = disparity_df_sorted['eta_squared'].fillna(0)
        fig.add_trace(
            go.Bar(
                x=disparity_df_sorted.index,
                y=eta_values,
                marker_color='purple',
                text=[f"{x:.3f}" for x in eta_values],
                textposition='auto',
                showlegend=False
            ),
            row=2, col=1
        )
    
    # Gini coefficients
    fig.add_trace(
        go.Bar(
            x=disparity_df_sorted.index,
            y=disparity_df_sorted['gini_coefficient'],
            marker_color='orange',
            text=[f"{x:.3f}" for x in disparity_df_sorted['gini_coefficient']],
            textposition='auto',
            showlegend=False
        ),
        row=2, col=2
    )
    
    # Best vs worst performing groups scatter
    fig.add_trace(
        go.Scatter(
            x=disparity_df_sorted['min_score'],
            y=disparity_df_sorted['max_score'],
            mode='markers+text',
            text=disparity_df_sorted.index,
            textposition='top center',
            marker=dict(
                size=disparity_df_sorted['range'] * 10,
                color=disparity_df_sorted['range'],
                colorscale='Reds',
                showscale=True
            ),
            showlegend=False
        ),
        row=3, col=1
    )
    
    # Add diagonal line for reference
    min_val = min(disparity_df_sorted['min_score'].min(), disparity_df_sorted['max_score'].min())
    max_val = max(disparity_df_sorted['min_score'].max(), disparity_df_sorted['max_score'].max())
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(dash='dash', color='gray'),
            showlegend=False
        ),
        row=3, col=1
    )
    
    # Coefficient of variation
    fig.add_trace(
        go.Bar(
            x=disparity_df_sorted.index,
            y=disparity_df_sorted['coefficient_of_variation'],
            marker_color='green',
            text=[f"{x:.3f}" for x in disparity_df_sorted['coefficient_of_variation']],
            textposition='auto',
            showlegend=False
        ),
        row=3, col=2
    )
    
    fig.update_layout(height=1200, title_text="Comprehensive Demographic Disparity Analysis")
    fig.show()
    
    # Summary insights
    print(f"\nDISPARITY ANALYSIS SUMMARY:")
    print(f"Demographic dimensions analyzed: {len(disparity_results)}")
    
    # Rank by disparity severity
    print(f"\nDisparity ranking (by range):")
    for i, (demo, range_val) in enumerate(disparity_df_sorted['range'].items(), 1):
        significance = disparity_results[demo].get('significance', 'ns')
        print(f"{i}. {demo}: {range_val:.3f} points {significance}")
    
    # Identify most concerning disparities
    high_disparity_threshold = disparity_df_sorted['range'].median()
    high_disparities = disparity_df_sorted[disparity_df_sorted['range'] > high_disparity_threshold]
    
    print(f"\nHigh-priority disparities (above median):")
    for demo in high_disparities.index:
        metrics = disparity_results[demo]
        print(f"  {demo}: {metrics['min_group']} vs {metrics['max_group']} ({metrics['range']:.3f} point gap)")

else:
    print("No disparity analysis could be performed - insufficient data")


# In[45]:


# 8.2 Section-Specific Disparity Analysis
print("=== SECTION-SPECIFIC DISPARITY ANALYSIS ===")

# Analyze disparities within each cultural section
section_disparities = {}

for section_name, questions in sections.items():
    print(f"\n--- {section_name.upper()} DISPARITIES ---")
    
    section_col = f'{section_name}_score'
    section_disparities[section_name] = {}
    
    # Analyze each demographic dimension for this section
    for demo_name, demo_col in demographic_columns.items():
        # Clean data
        demo_section_data = df[[demo_col, section_col]].dropna()
        
        if len(demo_section_data) == 0:
            continue
        
        # Calculate group means
        group_means = demo_section_data.groupby(demo_col)[section_col].mean()
        group_counts = demo_section_data.groupby(demo_col)[section_col].count()
        
        # Filter for sufficient group sizes
        min_size = 30 if demo_name != 'Department' else 20
        valid_groups = group_counts >= min_size
        
        if valid_groups.sum() < 2:
            continue
        
        filtered_means = group_means[valid_groups]
        filtered_counts = group_counts[valid_groups]
        
        # Calculate disparity metrics
        disparity = calculate_disparity_metrics(filtered_means, filtered_counts)
        section_disparities[section_name][demo_name] = disparity
        
        print(f"  {demo_name}: {disparity['range']:.3f} point gap ({disparity['min_group']} vs {disparity['max_group']})")

# Create section-specific disparity heatmap
if section_disparities:
    # Prepare data for heatmap
    disparity_matrix = []
    section_names = []
    demo_names = list(demographic_columns.keys())
    
    for section_name, demo_disparities in section_disparities.items():
        section_names.append(section_name.replace(' & ', ' &\n'))
        section_row = []
        for demo_name in demo_names:
            if demo_name in demo_disparities:
                section_row.append(demo_disparities[demo_name]['range'])
            else:
                section_row.append(0)  # No data available
        disparity_matrix.append(section_row)
    
    # Convert to numpy array
    disparity_matrix = np.array(disparity_matrix)
    
    # Create heatmap visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Section-Demographic Disparity Heatmap',
            'Most Severe Section Disparities',
            'Most Equitable Sections',
            'Cross-Section Disparity Comparison'
        ],
        specs=[[{"type": "xy"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "scatter"}]]
    )
    
    # Disparity heatmap
    fig.add_trace(
        go.Heatmap(
            z=disparity_matrix,
            x=demo_names,
            y=section_names,
            colorscale='Reds',
            text=np.round(disparity_matrix, 2),
            texttemplate="%{text}",
            textfont={"size": 8},
            showscale=True,
            colorbar=dict(title="Disparity Range")
        ),
        row=1, col=1
    )
    
    # Most severe section disparities
    max_disparities = disparity_matrix.max(axis=1)
    severe_sections = sorted(zip(section_names, max_disparities), key=lambda x: x[1], reverse=True)
    
    severe_section_names = [x[0] for x in severe_sections[:8]]
    severe_section_values = [x[1] for x in severe_sections[:8]]
    
    fig.add_trace(
        go.Bar(
            x=severe_section_values,
            y=severe_section_names,
            orientation='h',
            marker_color='red',
            text=[f"{x:.2f}" for x in severe_section_values],
            textposition='auto',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Most equitable sections (lowest average disparity)
    avg_disparities = disparity_matrix.mean(axis=1)
    equitable_sections = sorted(zip(section_names, avg_disparities), key=lambda x: x[1])
    
    equitable_section_names = [x[0] for x in equitable_sections[:6]]
    equitable_section_values = [x[1] for x in equitable_sections[:6]]
    
    fig.add_trace(
        go.Bar(
            x=equitable_section_values,
            y=equitable_section_names,
            orientation='h',
            marker_color='green',
            text=[f"{x:.2f}" for x in equitable_section_values],
            textposition='auto',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Cross-section disparity comparison
    section_means = [disparity_matrix[i].mean() for i in range(len(section_names))]
    section_maxes = [disparity_matrix[i].max() for i in range(len(section_names))]
    
    fig.add_trace(
        go.Scatter(
            x=section_means,
            y=section_maxes,
            mode='markers+text',
            text=section_names,
            textposition='top center',
            marker=dict(
                size=10,
                color=section_means,
                colorscale='RdYlBu_r',
                showscale=False
            ),
            showlegend=False
        ),
        row=2, col=2
    )
    
    fig.update_layout(height=1000, title_text="Section-Specific Disparity Analysis")
    fig.show()
    
    # Summary insights
    print(f"\nSECTION DISPARITY INSIGHTS:")
    
    # Section with highest overall disparity
    most_disparate_section = severe_sections[0][0].replace('\n', ' ')
    most_disparate_value = severe_sections[0][1]
    print(f"Most disparate section: {most_disparate_section} ({most_disparate_value:.3f} max disparity)")
    
    # Section with lowest overall disparity
    most_equitable_section = equitable_sections[0][0].replace('\n', ' ')
    most_equitable_value = equitable_sections[0][1]
    print(f"Most equitable section: {most_equitable_section} ({most_equitable_value:.3f} avg disparity)")
    
    # Demographic dimension with highest average disparity across sections
    demo_avg_disparities = disparity_matrix.mean(axis=0)
    most_disparate_demo_idx = np.argmax(demo_avg_disparities)
    most_disparate_demo = demo_names[most_disparate_demo_idx]
    print(f"Most disparate demographic: {most_disparate_demo} ({demo_avg_disparities[most_disparate_demo_idx]:.3f} avg disparity)")
    
    # Create priority matrix
    print(f"\nPRIORITY INTERVENTIONS (Section × Demographic combinations with highest disparities):")
    
    # Find top 10 section-demographic combinations with highest disparities
    top_disparities = []
    for i, section_name in enumerate(section_names):
        for j, demo_name in enumerate(demo_names):
            if disparity_matrix[i, j] > 0:  # Has data
                top_disparities.append((section_name.replace('\n', ' '), demo_name, disparity_matrix[i, j]))
    
    top_disparities.sort(key=lambda x: x[2], reverse=True)
    
    for i, (section, demo, disparity) in enumerate(top_disparities[:10], 1):
        priority_level = "Critical" if disparity > 1.0 else "High" if disparity > 0.5 else "Medium"
        print(f"{i:2d}. {section} × {demo}: {disparity:.3f} ({priority_level})")

else:
    print("Section-specific disparity analysis could not be performed")


# ## 9. PROFESSIONAL PROGRESSION ANALYSIS
# 
# Analysis of career advancement patterns and their impact on workplace culture experiences, including tenure progression, position mobility, and leadership development insights.

# In[46]:


# 9.1 Career Journey Analysis: Tenure × Position Progression
print("=== CAREER JOURNEY ANALYSIS ===")

# Check for tenure data
tenure_cols = ['q30', 'tenure', 'tenure_range', 'years_at_organization']
tenure_col = None
for col in tenure_cols:
    if col in df.columns:
        tenure_col = col
        break

if tenure_col and position_col:
    print(f"Analyzing career progression using {tenure_col} and {position_col}")
    
    # Clean and prepare progression data
    progression_data = df[[tenure_col, position_col, 'overall_culture_score'] + section_score_cols + ['domain', 'department']].dropna()
    
    print(f"Career progression dataset: {len(progression_data):,} complete records")
    
    # Define tenure progression stages
    tenure_order = ['<1_year', '1-3_years', '3-7_years', '7-15_years', '15+_years']
    available_tenure = [t for t in tenure_order if t in progression_data[tenure_col].unique()]
    if not available_tenure:
        available_tenure = sorted(progression_data[tenure_col].unique())
    
    # Define position hierarchy
    position_hierarchy = ['Entry Level', 'Mid Level', 'Senior Level', 'Executive Level', 'C-Suite']
    available_positions = [p for p in position_hierarchy if p in progression_data[position_col].unique()]
    if not available_positions:
        available_positions = sorted(progression_data[position_col].unique())
    
    print(f"Tenure stages: {available_tenure}")
    print(f"Position levels: {available_positions}")
    
    # Career progression analysis
    progression_stats = progression_data.groupby([tenure_col, position_col]).agg({
        'overall_culture_score': ['count', 'mean', 'std'],
        **{col: 'mean' for col in section_score_cols}
    }).round(3)
    
    # Filter combinations with sufficient data
    min_combination_size = 15
    valid_combinations = progression_stats['overall_culture_score']['count'] >= min_combination_size
    valid_progression_stats = progression_stats[valid_combinations]
    
    print(f"Valid tenure-position combinations (n≥{min_combination_size}): {valid_combinations.sum()}")
    
    if valid_combinations.sum() > 0:
        # Visualization
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Career Progression Matrix (Tenure × Position)',
                'Culture Score Evolution by Tenure',
                'Position-Specific Tenure Effects',
                'Career Advancement Patterns',
                'Progression Success Factors',
                'Cross-Domain Career Patterns'
            ],
            specs=[[{"type": "xy"}, {"type": "xy"}],
                   [{"type": "box"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Career progression heatmap
        progression_pivot = progression_data.pivot_table(
            values='overall_culture_score',
            index=position_col,
            columns=tenure_col,
            aggfunc='mean'
        )
        
        # Reorder if possible
        if len(available_tenure) > 1 and len(available_positions) > 1:
            try:
                progression_pivot = progression_pivot.reindex(index=available_positions, columns=available_tenure)
            except:
                pass  # Use original order if reindexing fails
        
        fig.add_trace(
            go.Heatmap(
                z=progression_pivot.values,
                x=progression_pivot.columns,
                y=progression_pivot.index,
                colorscale='RdYlBu_r',
                text=np.round(progression_pivot.values, 2),
                texttemplate="%{text}",
                textfont={"size": 10},
                showscale=True,
                colorbar=dict(title="Culture Score")
            ),
            row=1, col=1
        )
        
        # Culture score evolution by tenure
        tenure_evolution = progression_data.groupby(tenure_col)['overall_culture_score'].mean()
        if len(available_tenure) > 1:
            try:
                tenure_evolution = tenure_evolution.reindex(available_tenure)
            except:
                pass
        
        fig.add_trace(
            go.Scatter(
                x=tenure_evolution.index,
                y=tenure_evolution.values,
                mode='lines+markers',
                name='Overall Trend',
                line=dict(width=3),
                marker=dict(size=8),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Add position-specific trends
        colors_pos = px.colors.qualitative.Set1
        for i, pos in enumerate(available_positions[:4]):  # Limit to 4 positions for clarity
            pos_tenure_data = progression_data[progression_data[position_col] == pos]
            pos_tenure_evolution = pos_tenure_data.groupby(tenure_col)['overall_culture_score'].mean()
            
            fig.add_trace(
                go.Scatter(
                    x=pos_tenure_evolution.index,
                    y=pos_tenure_evolution.values,
                    mode='lines+markers',
                    name=pos[:15],  # Truncate long names
                    line=dict(color=colors_pos[i % len(colors_pos)], width=2),
                    marker=dict(size=6),
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # Position-specific tenure effects (box plots)
        for i, pos in enumerate(available_positions[:4]):
            pos_data = progression_data[progression_data[position_col] == pos]['overall_culture_score']
            fig.add_trace(
                go.Box(
                    y=pos_data,
                    name=pos[:15],
                    marker_color=colors_pos[i % len(colors_pos)],
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # Career advancement patterns (position distribution by tenure)
        advancement_patterns = pd.crosstab(progression_data[tenure_col], progression_data[position_col], normalize='index')
        
        # Show stacked bar for tenure stages
        for i, pos in enumerate(available_positions):
            if pos in advancement_patterns.columns:
                fig.add_trace(
                    go.Bar(
                        x=advancement_patterns.index,
                        y=advancement_patterns[pos],
                        name=pos[:15],
                        marker_color=colors_pos[i % len(colors_pos)],
                        showlegend=False
                    ),
                    row=2, col=2
                )
        
        # Progression success factors (tenure vs culture score scatter)
        # Create numeric tenure for correlation
        tenure_numeric = {tenure: i for i, tenure in enumerate(available_tenure)}
        progression_data['tenure_numeric'] = progression_data[tenure_col].map(tenure_numeric)
        
        # Calculate correlation by position
        progression_correlations = []
        position_names = []
        
        for pos in available_positions:
            pos_data = progression_data[progression_data[position_col] == pos]
            if len(pos_data) >= 10 and pos_data['tenure_numeric'].nunique() >= 3:
                corr, p_val = stats.pearsonr(pos_data['tenure_numeric'], pos_data['overall_culture_score'])
                progression_correlations.append(corr)
                position_names.append(pos[:15])
        
        if progression_correlations:
            fig.add_trace(
                go.Scatter(
                    x=progression_correlations,
                    y=position_names,
                    mode='markers',
                    marker=dict(
                        size=15,
                        color=progression_correlations,
                        colorscale='RdYlBu_r',
                        showscale=False
                    ),
                    text=[f"r={x:.3f}" for x in progression_correlations],
                    textposition='middle right',
                    showlegend=False
                ),
                row=3, col=1
            )
        
        # Cross-domain career patterns
        domain_progression = progression_data.groupby(['domain', tenure_col])['overall_culture_score'].mean().unstack()
        
        for domain in domains:
            if domain in domain_progression.index:
                fig.add_trace(
                    go.Bar(
                        x=domain_progression.columns,
                        y=domain_progression.loc[domain],
                        name=domain,
                        marker_color=domain_colors[domain],
                        showlegend=False
                    ),
                    row=3, col=2
                )
        
        fig.update_layout(height=1400, title_text="Career Journey Analysis Dashboard")
        fig.show()
        
        # Statistical analysis of career progression
        print(f"\nCAREER PROGRESSION INSIGHTS:")
        
        # Overall tenure effect
        if len(available_tenure) >= 3:
            tenure_groups = [progression_data[progression_data[tenure_col] == t]['overall_culture_score'].dropna() 
                           for t in available_tenure]
            tenure_groups = [group for group in tenure_groups if len(group) >= 10]
            
            if len(tenure_groups) >= 2:
                f_stat, p_value = f_oneway(*tenure_groups)
                print(f"Tenure effect ANOVA: F={f_stat:.3f}, p={p_value:.3e}")
                
                # Linear trend test
                tenure_means = [group.mean() for group in tenure_groups]
                tenure_indices = list(range(len(tenure_means)))
                trend_corr, trend_p = stats.pearsonr(tenure_indices, tenure_means)
                print(f"Career progression trend: r={trend_corr:.3f}, p={trend_p:.3e}")
                
                trend_direction = "improves" if trend_corr > 0 else "declines" if trend_corr < 0 else "remains stable"
                print(f"Culture score {trend_direction} with tenure")
        
        # Position advancement analysis
        print(f"\nPosition advancement patterns:")
        for tenure in available_tenure[1:]:  # Skip first tenure stage
            current_positions = progression_data[progression_data[tenure_col] == tenure][position_col].value_counts()
            if len(current_positions) > 0:
                top_position = current_positions.index[0]
                percentage = current_positions.iloc[0] / current_positions.sum() * 100
                print(f"  {tenure}: {percentage:.1f}% in {top_position}")
        
        # Career success metrics
        print(f"\nCareer success indicators:")
        
        # Leadership attainment by tenure
        leadership_keywords = ['Senior', 'Executive', 'Manager', 'Director', 'VP', 'C-Suite']
        progression_data['is_leadership'] = progression_data[position_col].apply(
            lambda x: any(keyword in str(x) for keyword in leadership_keywords)
        )
        
        leadership_by_tenure = progression_data.groupby(tenure_col)['is_leadership'].mean()
        print(f"Leadership attainment by tenure:")
        for tenure, rate in leadership_by_tenure.items():
            print(f"  {tenure}: {rate*100:.1f}% in leadership roles")

else:
    print("Career journey analysis skipped - insufficient data for tenure and position analysis")


# In[47]:


# 9.2 Leadership Development Pipeline Analysis
print("=== LEADERSHIP DEVELOPMENT PIPELINE ANALYSIS ===")

# Analyze the leadership pipeline and development patterns
if position_col and 'department' in df.columns:
    print(f"Analyzing leadership pipeline across positions and departments")
    
    # Define leadership categories
    leadership_keywords = ['Senior', 'Executive', 'Manager', 'Director', 'VP', 'C-Suite', 'Lead', 'Head', 'Chief']
    
    def categorize_leadership_level(position):
        if pd.isna(position):
            return 'Unknown'
        position_str = str(position).lower()
        
        if any(keyword.lower() in position_str for keyword in ['c-suite', 'ceo', 'cfo', 'cto', 'chief']):
            return 'C-Suite'
        elif any(keyword.lower() in position_str for keyword in ['executive', 'vp', 'vice president']):
            return 'Executive'
        elif any(keyword.lower() in position_str for keyword in ['director', 'head']):
            return 'Director'
        elif any(keyword.lower() in position_str for keyword in ['manager', 'lead']):
            return 'Manager'
        elif any(keyword.lower() in position_str for keyword in ['senior']):
            return 'Senior IC'
        else:
            return 'Individual Contributor'
    
    # Apply leadership categorization
    df['leadership_level'] = df[position_col].apply(categorize_leadership_level)
    
    # Leadership pipeline analysis
    leadership_counts = df['leadership_level'].value_counts()
    print(f"Leadership Pipeline Distribution:")
    for level, count in leadership_counts.items():
        pct = count / len(df) * 100
        print(f"  {level}: {count:,} ({pct:.1f}%)")
    
    # Leadership effectiveness analysis
    leadership_stats = df.groupby('leadership_level').agg({
        'overall_culture_score': ['count', 'mean', 'std', 'median'],
        **{col: 'mean' for col in section_score_cols}
    }).round(3)
    
    print(f"\nLeadership Effectiveness by Level:")
    print(leadership_stats['overall_culture_score'])
    
    # Comprehensive leadership visualization
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            'Leadership Pipeline Distribution',
            'Leadership Effectiveness by Level',
            'Department Leadership Distribution',
            'Leadership Culture Impact',
            'Cross-Domain Leadership Comparison',
            'Leadership Development Success Factors'
        ],
        specs=[[{"type": "pie"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "box"}],
               [{"type": "bar"}, {"type": "scatter"}]]
    )
    
    # Leadership pipeline pie chart
    fig.add_trace(
        go.Pie(
            labels=leadership_counts.index,
            values=leadership_counts.values,
            name="Leadership Distribution"
        ),
        row=1, col=1
    )
    
    # Leadership effectiveness bar chart
    leadership_means = df.groupby('leadership_level')['overall_culture_score'].mean().sort_values(ascending=False)
    colors_leadership = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    fig.add_trace(
        go.Bar(
            x=leadership_means.index,
            y=leadership_means.values,
            marker_color=colors_leadership[:len(leadership_means)],
            text=[f"{x:.2f}" for x in leadership_means.values],
            textposition='auto',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Department leadership distribution
    dept_leadership = pd.crosstab(df['department'], df['leadership_level'], normalize='index')
    top_departments = df['department'].value_counts().head(8).index
    dept_leadership_top = dept_leadership.loc[top_departments]
    
    # Show management vs IC ratio
    management_levels = ['Manager', 'Director', 'Executive', 'C-Suite']
    dept_leadership_top['Management'] = dept_leadership_top[
        [col for col in management_levels if col in dept_leadership_top.columns]
    ].sum(axis=1)
    
    fig.add_trace(
        go.Bar(
            x=dept_leadership_top.index,
            y=dept_leadership_top['Management'],
            name='Management %',
            marker_color='red',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Leadership culture impact (box plots by level)
    leadership_hierarchy = ['Individual Contributor', 'Senior IC', 'Manager', 'Director', 'Executive', 'C-Suite']
    available_leadership = [level for level in leadership_hierarchy if level in df['leadership_level'].unique()]
    
    for i, level in enumerate(available_leadership):
        level_data = df[df['leadership_level'] == level]['overall_culture_score']
        fig.add_trace(
            go.Box(
                y=level_data,
                name=level,
                marker_color=colors_leadership[i % len(colors_leadership)],
                showlegend=False
            ),
            row=2, col=2
        )
    
    # Cross-domain leadership comparison
    domain_leadership = df.groupby(['domain', 'leadership_level'])['overall_culture_score'].mean().unstack()
    
    for domain in domains:
        if domain in domain_leadership.index:
            management_score = domain_leadership.loc[domain, management_levels].mean() if any(level in domain_leadership.columns for level in management_levels) else 0
            ic_score = domain_leadership.loc[domain, 'Individual Contributor'] if 'Individual Contributor' in domain_leadership.columns else 0
            
            fig.add_trace(
                go.Bar(
                    x=[f"{domain}\nManagement", f"{domain}\nIC"],
                    y=[management_score, ic_score],
                    name=domain,
                    marker_color=domain_colors[domain],
                    showlegend=False
                ),
                row=3, col=1
            )
    
    # Leadership development success factors
    if tenure_col:
        # Analyze leadership attainment by tenure and other factors
        leadership_success = df.copy()
        leadership_success['is_leadership'] = leadership_success['leadership_level'].apply(
            lambda x: x in ['Manager', 'Director', 'Executive', 'C-Suite']
        )
        
        # Leadership rate by tenure
        tenure_leadership = leadership_success.groupby(tenure_col)['is_leadership'].mean()
        
        # Leadership rate by education (if available)
        education_leadership = None
        if education_col:
            education_leadership = leadership_success.groupby(education_col)['is_leadership'].mean()
        
        # Tenure effect
        fig.add_trace(
            go.Scatter(
                x=list(range(len(tenure_leadership))),
                y=tenure_leadership.values,
                mode='lines+markers',
                text=tenure_leadership.index,
                textposition='top center',
                name='Leadership Rate by Tenure',
                showlegend=False
            ),
            row=3, col=2
        )
    
    fig.update_layout(height=1400, title_text="Leadership Development Pipeline Analysis")
    fig.show()
    
    # Statistical analysis of leadership pipeline
    print(f"\nLEADERSHIP PIPELINE INSIGHTS:")
    
    # Leadership effectiveness ranking
    print(f"Leadership effectiveness ranking:")
    for i, (level, score) in enumerate(leadership_means.items(), 1):
        count = leadership_counts[level]
        print(f"{i}. {level}: {score:.3f} (n={count:,})")
    
    # Leadership vs IC comparison
    management_data = df[df['leadership_level'].isin(['Manager', 'Director', 'Executive', 'C-Suite'])]
    ic_data = df[df['leadership_level'].isin(['Individual Contributor', 'Senior IC'])]
    
    if len(management_data) > 0 and len(ic_data) > 0:
        mgmt_score = management_data['overall_culture_score'].mean()
        ic_score = ic_data['overall_culture_score'].mean()
        
        # Statistical test
        t_stat, p_value = stats.ttest_ind(
            management_data['overall_culture_score'].dropna(),
            ic_data['overall_culture_score'].dropna()
        )
        
        print(f"\nManagement vs Individual Contributors:")
        print(f"  Management (n={len(management_data):,}): {mgmt_score:.3f}")
        print(f"  Individual Contributors (n={len(ic_data):,}): {ic_score:.3f}")
        print(f"  Difference: {mgmt_score - ic_score:+.3f}")
        print(f"  Statistical test: t={t_stat:.3f}, p={p_value:.3e}")
    
    # Department leadership effectiveness
    print(f"\nDepartment leadership effectiveness:")
    dept_mgmt_scores = {}
    for dept in top_departments:
        dept_mgmt = df[(df['department'] == dept) & 
                      (df['leadership_level'].isin(['Manager', 'Director', 'Executive', 'C-Suite']))]
        if len(dept_mgmt) >= 10:
            dept_mgmt_scores[dept] = dept_mgmt['overall_culture_score'].mean()
    
    dept_mgmt_sorted = sorted(dept_mgmt_scores.items(), key=lambda x: x[1], reverse=True)
    for dept, score in dept_mgmt_sorted[:5]:
        print(f"  {dept}: {score:.3f}")
    
    # Leadership development recommendations
    print(f"\nLeadership development recommendations:")
    
    # Find departments with leadership gaps
    dept_leadership_rates = df.groupby('department')['leadership_level'].apply(
        lambda x: (x.isin(['Manager', 'Director', 'Executive', 'C-Suite'])).mean()
    ).sort_values()
    
    low_leadership_depts = dept_leadership_rates.head(3)
    print(f"Departments needing leadership development:")
    for dept, rate in low_leadership_depts.items():
        print(f"  {dept}: {rate*100:.1f}% in leadership roles")
    
    # Find leadership levels with culture challenges
    leadership_problems = leadership_means.sort_values().head(3)
    print(f"Leadership levels needing culture improvement:")
    for level, score in leadership_problems.items():
        print(f"  {level}: {score:.3f} (vs overall {df['overall_culture_score'].mean():.3f})")

else:
    print("Leadership development pipeline analysis skipped - insufficient position or department data")


# In[48]:


# 9.3 Comprehensive Professional Progression Summary
print("=== COMPREHENSIVE PROFESSIONAL PROGRESSION SUMMARY ===")

# Create final executive summary of all professional progression findings
progression_summary = {
    'demographic_analysis': {},
    'professional_analysis': {},
    'intersectional_findings': {},
    'disparity_metrics': {},
    'leadership_insights': {},
    'recommendations': []
}

# Collect key findings from previous analyses
print("Synthesizing professional progression findings...")

# Overall progression metrics
if tenure_col and position_col:
    print(f"\n=== EXECUTIVE SUMMARY: PROFESSIONAL PROGRESSION ===")
    
    # Key demographic findings
    demo_findings = []
    if age_col:
        demo_findings.append("Age-based culture variations identified")
    if gender_col:
        demo_findings.append("Gender-based culture disparities documented")
    if race_col:
        demo_findings.append("Racial/ethnic culture gaps analyzed")
    if education_col:
        demo_findings.append("Education-culture correlations established")
    
    print(f"Demographic dimensions analyzed: {len(demo_findings)}")
    for finding in demo_findings:
        print(f"  ✓ {finding}")
    
    # Professional progression insights
    print(f"\nProfessional progression insights:")
    
    # Calculate overall progression trends
    if len(available_tenure) >= 3 and len(available_positions) >= 3:
        # Career advancement rates
        advancement_rates = {}
        for i, tenure in enumerate(available_tenure[1:], 1):
            tenure_data = df[df[tenure_col] == tenure]
            leadership_rate = tenure_data['leadership_level'].apply(
                lambda x: x in ['Manager', 'Director', 'Executive', 'C-Suite']
            ).mean()
            advancement_rates[tenure] = leadership_rate
        
        print(f"  Leadership advancement by tenure:")
        for tenure, rate in advancement_rates.items():
            print(f"    {tenure}: {rate*100:.1f}% in leadership roles")
    
    # Department effectiveness ranking
    if 'department' in df.columns:
        dept_effectiveness = df.groupby('department')['overall_culture_score'].mean().sort_values(ascending=False)
        print(f"\n  Top 5 departments by culture score:")
        for i, (dept, score) in enumerate(dept_effectiveness.head(5).items(), 1):
            print(f"    {i}. {dept}: {score:.3f}")
        
        print(f"\n  Bottom 5 departments by culture score:")
        for i, (dept, score) in enumerate(dept_effectiveness.tail(5).items(), 1):
            print(f"    {i}. {dept}: {score:.3f}")
    
    # Cross-domain progression patterns
    print(f"\n  Cross-domain progression patterns:")
    domain_progression_summary = df.groupby('domain').agg({
        'overall_culture_score': 'mean',
        'leadership_level': lambda x: (x.isin(['Manager', 'Director', 'Executive', 'C-Suite'])).mean()
    }).round(3)
    
    for domain in domains:
        if domain in domain_progression_summary.index:
            culture_score = domain_progression_summary.loc[domain, 'overall_culture_score']
            leadership_rate = domain_progression_summary.loc[domain, 'leadership_level']
            print(f"    {domain}: Culture {culture_score:.3f}, Leadership {leadership_rate*100:.1f}%")
    
    # Key disparity findings
    if disparity_results:
        print(f"\n  Key disparity findings:")
        disparity_df = pd.DataFrame(disparity_results).T
        worst_disparities = disparity_df.nlargest(3, 'range')
        
        for demo, row in worst_disparities.iterrows():
            print(f"    {demo}: {row['range']:.3f} point gap ({row['min_group']} vs {row['max_group']})")
    
    # Actionable recommendations
    print(f"\n=== STRATEGIC RECOMMENDATIONS ===")
    
    recommendations = [
        "DEMOGRAPHIC EQUITY INITIATIVES:",
        "• Implement targeted support programs for underrepresented demographic groups",
        "• Establish mentorship programs connecting junior and senior employees",
        "• Create demographic-specific professional development tracks",
        "",
        "LEADERSHIP DEVELOPMENT:",
        "• Expand leadership pipeline programs for underrepresented groups",
        "• Implement 360-degree feedback systems for all leadership levels",
        "• Create cross-department leadership rotation programs",
        "",
        "DEPARTMENT-SPECIFIC INTERVENTIONS:",
        "• Conduct deep-dive culture assessments in low-scoring departments",
        "• Implement best practice sharing between high and low-performing departments",
        "• Establish department-specific culture improvement targets",
        "",
        "CAREER PROGRESSION SUPPORT:",
        "• Create clear career progression pathways for all roles",
        "• Implement skills-based advancement criteria",
        "• Establish cross-domain mobility programs",
        "",
        "MONITORING AND ACCOUNTABILITY:",
        "• Establish quarterly culture score tracking by demographic groups",
        "• Create department-level culture score targets and accountability measures",
        "• Implement regular intersectional analysis reporting"
    ]
    
    for rec in recommendations:
        print(f"{rec}")
    
    # Create final comprehensive dashboard
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            'Overall Progression Summary',
            'Demographic Disparity Overview',
            'Leadership Pipeline Health',
            'Department Performance Matrix',
            'Cross-Domain Comparison',
            'Priority Intervention Areas'
        ],
        specs=[[{"type": "scatter"}, {"type": "bar"}],
               [{"type": "xy"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Overall progression summary (tenure vs position vs culture score)
    if len(available_tenure) >= 3 and len(available_positions) >= 3:
        progression_summary_data = df.groupby([tenure_col, position_col])['overall_culture_score'].mean().reset_index()
        
        # Create bubble chart
        fig.add_trace(
            go.Scatter(
                x=[available_tenure.index(t) if t in available_tenure else 0 for t in progression_summary_data[tenure_col]],
                y=[available_positions.index(p) if p in available_positions else 0 for p in progression_summary_data[position_col]],
                mode='markers',
                marker=dict(
                    size=progression_summary_data['overall_culture_score'] * 10,
                    color=progression_summary_data['overall_culture_score'],
                    colorscale='RdYlBu_r',
                    showscale=True
                ),
                text=[f"{t}-{p}: {s:.2f}" for t, p, s in zip(
                    progression_summary_data[tenure_col],
                    progression_summary_data[position_col],
                    progression_summary_data['overall_culture_score']
                )],
                showlegend=False
            ),
            row=1, col=1
        )
    
    # Demographic disparity overview
    if disparity_results:
        disparity_ranges = [disparity_results[demo]['range'] for demo in disparity_results.keys()]
        demo_names = list(disparity_results.keys())
        
        fig.add_trace(
            go.Bar(
                x=demo_names,
                y=disparity_ranges,
                marker_color='red',
                text=[f"{x:.2f}" for x in disparity_ranges],
                textposition='auto',
                showlegend=False
            ),
            row=1, col=2
        )
    
    # Leadership pipeline health
    if 'leadership_level' in df.columns:
        leadership_health = df['leadership_level'].value_counts()
        
        fig.add_trace(
            go.Bar(
                x=leadership_health.index,
                y=leadership_health.values,
                marker_color='blue',
                showlegend=False
            ),
            row=1, col=3
        )
    
    # Department performance matrix
    if 'department' in df.columns:
        dept_perf = df.groupby('department').agg({
            'overall_culture_score': 'mean',
            'response_id': 'count'
        }).round(3)
        
        top_depts = dept_perf.nlargest(15, 'response_id')
        
        fig.add_trace(
            go.Scatter(
                x=top_depts['response_id'],
                y=top_depts['overall_culture_score'],
                mode='markers+text',
                text=top_depts.index,
                textposition='top center',
                marker=dict(
                    size=8,
                    color=top_depts['overall_culture_score'],
                    colorscale='RdYlBu_r'
                ),
                showlegend=False
            ),
            row=2, col=1
        )
    
    # Cross-domain comparison
    domain_summary = df.groupby('domain')['overall_culture_score'].mean()
    fig.add_trace(
        go.Bar(
            x=domain_summary.index,
            y=domain_summary.values,
            marker_color=[domain_colors[d] for d in domain_summary.index],
            text=[f"{x:.2f}" for x in domain_summary.values],
            textposition='auto',
            showlegend=False
        ),
        row=2, col=2
    )
    
    # Priority intervention areas
    if disparity_results:
        priority_interventions = pd.Series(disparity_ranges, index=demo_names).nlargest(5)
        
        fig.add_trace(
            go.Bar(
                x=priority_interventions.values,
                y=priority_interventions.index,
                orientation='h',
                marker_color='orange',
                text=[f"{x:.2f}" for x in priority_interventions.values],
                textposition='auto',
                showlegend=False
            ),
            row=2, col=3
        )
    
    fig.update_layout(height=1000, title_text="Professional Progression Analysis - Executive Summary Dashboard")
    fig.show()
    
    # Final statistics
    print(f"\n=== FINAL STATISTICS ===")
    print(f"Total survey responses analyzed: {len(df):,}")
    print(f"Demographic dimensions covered: {len(demographic_columns)}")
    print(f"Statistical tests performed: {sum(1 for d in disparity_results.values() if 'p_value' in d)}")
    print(f"Organizations included: {df['organization_name'].nunique():,}")
    print(f"Departments analyzed: {df['department'].nunique() if 'department' in df.columns else 'N/A'}")
    print(f"Domains covered: {', '.join(domains)}")
    
    overall_culture_mean = df['overall_culture_score'].mean()
    overall_culture_std = df['overall_culture_score'].std()
    print(f"Overall culture score: {overall_culture_mean:.3f} ± {overall_culture_std:.3f}")
    
    if disparity_results:
        avg_disparity = pd.DataFrame(disparity_results).T['range'].mean()
        max_disparity = pd.DataFrame(disparity_results).T['range'].max()
        print(f"Average demographic disparity: {avg_disparity:.3f} points")
        print(f"Maximum demographic disparity: {max_disparity:.3f} points")
    
    print(f"\n" + "="*80)
    print("PROFESSIONAL PROGRESSION ANALYSIS COMPLETE")
    print("="*80)
    print("This comprehensive analysis provides actionable insights for:")
    print("• Targeted demographic equity initiatives")
    print("• Leadership development program optimization") 
    print("• Department-specific culture improvement strategies")
    print("• Career progression pathway enhancement")
    print("• Data-driven diversity, equity, and inclusion efforts")
    print("\nReady for executive presentation and strategic implementation.")

else:
    print("Comprehensive professional progression summary skipped - insufficient core data")


# In[ ]:




