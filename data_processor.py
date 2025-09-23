#!/usr/bin/env python3
"""
HSEG Data Processor - Converts your Python analysis to API-ready data
Based on your hseg_comprehensive_analysis.py file
"""

import pandas as pd
import numpy as np
import json
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class HSEGDataProcessor:
    def __init__(self, csv_file_path='hseg_final_dataset.csv'):
        """Initialize with your CSV file"""
        self.csv_path = csv_file_path
        self.df = None
        self.sections = {
            'Power Abuse & Suppression': ['q1', 'q2', 'q3', 'q4'],
            'Discrimination & Exclusion': ['q5', 'q6', 'q7'],
            'Manipulative Work Culture': ['q8', 'q9', 'q10'],
            'Failure of Accountability': ['q11', 'q12', 'q13', 'q14'],
            'Mental Health Harm': ['q15', 'q16', 'q17', 'q18'],
            'Erosion of Voice & Autonomy': ['q19', 'q20', 'q21', 'q22']
        }
        self.load_and_process_data()
    
    def load_and_process_data(self):
        """Load and process the CSV data"""
        try:
            self.df = pd.read_csv(self.csv_path)
            print(f"Loaded {len(self.df):,} records from {self.csv_path}")
            
            # Calculate section scores (from your original code)
            for section_name, questions in self.sections.items():
                available_questions = [q for q in questions if q in self.df.columns]
                if available_questions:
                    self.df[f'{section_name}_score'] = self.df[available_questions].mean(axis=1)
                    self.df[f'{section_name}_sum'] = self.df[available_questions].sum(axis=1)
            
            # Overall culture score
            all_questions = [q for section_questions in self.sections.values() for q in section_questions]
            available_all_questions = [q for q in all_questions if q in self.df.columns]
            if available_all_questions:
                self.df['overall_culture_score'] = self.df[available_all_questions].mean(axis=1)
            
            print("Data processing completed successfully!")
            
        except FileNotFoundError:
            print(f"Error: Could not find {self.csv_path}")
            print("Please ensure your CSV file is in the same directory")
        except Exception as e:
            print(f"Error processing data: {e}")
    
    def get_overview_stats(self):
        """Generate overview statistics"""
        if self.df is None:
            return {}
            
        return {
            'totalResponses': len(self.df),
            'organizations': self.df['organization_name'].nunique() if 'organization_name' in self.df.columns else 0,
            'domains': self.df['domain'].unique().tolist() if 'domain' in self.df.columns else [],
            'departments': self.df['department'].nunique() if 'department' in self.df.columns else 0,
            'averageCultureScore': float(self.df['overall_culture_score'].mean()) if 'overall_culture_score' in self.df.columns else 0,
            'responseRate': 100.0  # Placeholder - calculate based on your methodology
        }
    
    def get_domain_analysis(self):
        """Domain analysis from your original code"""
        if self.df is None or 'domain' not in self.df.columns:
            return []
            
        domain_stats = self.df.groupby('domain').agg({
            'overall_culture_score': ['count', 'mean', 'std'] if 'overall_culture_score' in self.df.columns else ['count']
        }).round(3)
        
        result = []
        for domain in domain_stats.index:
            count = int(domain_stats.loc[domain, ('overall_culture_score', 'count')])
            percentage = round((count / len(self.df)) * 100, 1)
            
            domain_data = {
                'domain': domain,
                'count': count,
                'percentage': percentage
            }
            
            if 'overall_culture_score' in self.df.columns:
                domain_data['avgScore'] = float(domain_stats.loc[domain, ('overall_culture_score', 'mean')])
                domain_data['stdScore'] = float(domain_stats.loc[domain, ('overall_culture_score', 'std')])
            
            result.append(domain_data)
        
        return result
    
    def get_section_analysis(self):
        """Section analysis from your original code"""
        if self.df is None:
            return []
            
        result = []
        section_score_cols = [f'{section}_score' for section in self.sections.keys()]
        
        for section_name in self.sections.keys():
            section_col = f'{section_name}_score'
            if section_col not in self.df.columns:
                continue
                
            # Overall section stats
            section_data = {
                'section': section_name,
                'overallScore': float(self.df[section_col].mean()),
                'overallStd': float(self.df[section_col].std()),
                'domainBreakdown': []
            }
            
            # Domain breakdown
            if 'domain' in self.df.columns:
                domain_breakdown = self.df.groupby('domain')[section_col].agg(['mean', 'std', 'count']).round(3)
                
                for domain in domain_breakdown.index:
                    section_data['domainBreakdown'].append({
                        'domain': domain,
                        'score': float(domain_breakdown.loc[domain, 'mean']),
                        'std': float(domain_breakdown.loc[domain, 'std']),
                        'count': int(domain_breakdown.loc[domain, 'count'])
                    })
            
            result.append(section_data)
        
        return result
    
    def get_correlation_analysis(self):
        """Correlation analysis from your original code"""
        if self.df is None:
            return []
            
        section_score_cols = [f'{section}_score' for section in self.sections.keys()]
        available_cols = [col for col in section_score_cols if col in self.df.columns]
        
        if len(available_cols) < 2:
            return []
            
        corr_matrix = self.df[available_cols].corr()
        
        correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                correlations.append({
                    'section1': corr_matrix.columns[i].replace('_score', ''),
                    'section2': corr_matrix.columns[j].replace('_score', ''),
                    'correlation': float(corr_matrix.iloc[i, j])
                })
        
        return sorted(correlations, key=lambda x: abs(x['correlation']), reverse=True)
    
    def get_organization_benchmarks(self, min_responses=50):
        """Organization benchmarking from your original code"""
        if self.df is None or 'organization_name' not in self.df.columns:
            return []
            
        org_stats = self.df.groupby('organization_name').agg({
            'overall_culture_score': ['count', 'mean', 'std'] if 'overall_culture_score' in self.df.columns else ['count'],
            'domain': 'first'
        }).round(3)
        
        # Filter organizations with sufficient responses
        valid_orgs = org_stats[org_stats[('overall_culture_score', 'count')] >= min_responses]
        
        result = []
        for org in valid_orgs.index:
            org_data = {
                'organization': org,
                'domain': valid_orgs.loc[org, ('domain', 'first')],
                'responses': int(valid_orgs.loc[org, ('overall_culture_score', 'count')])
            }
            
            if 'overall_culture_score' in self.df.columns:
                org_data['score'] = float(valid_orgs.loc[org, ('overall_culture_score', 'mean')])
                org_data['std'] = float(valid_orgs.loc[org, ('overall_culture_score', 'std')])
            
            result.append(org_data)
        
        # Sort by score (best first)
        if 'overall_culture_score' in self.df.columns:
            result.sort(key=lambda x: x['score'])
            
            # Add rankings
            for i, org in enumerate(result):
                org['rank'] = i + 1
        
        return result
    
    def get_demographic_analysis(self):
        """Demographic analysis from your original code"""
        if self.df is None:
            return []
            
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
        
        result = []
        
        for col, label in demographic_cols.items():
            if col not in self.df.columns:
                continue
                
            demo_stats = self.df.groupby(col).agg({
                'overall_culture_score': ['count', 'mean', 'std'] if 'overall_culture_score' in self.df.columns else ['count']
            }).round(3)
            
            demo_data = {
                'demographic': label,
                'groups': []
            }
            
            for group in demo_stats.index:
                if pd.isna(group):
                    continue
                    
                group_data = {
                    'group': str(group),
                    'count': int(demo_stats.loc[group, ('overall_culture_score', 'count')])
                }
                
                if 'overall_culture_score' in self.df.columns:
                    group_data['score'] = float(demo_stats.loc[group, ('overall_culture_score', 'mean')])
                    group_data['std'] = float(demo_stats.loc[group, ('overall_culture_score', 'std')])
                
                demo_data['groups'].append(group_data)
            
            result.append(demo_data)
        
        return result
    
    def generate_all_data(self):
        """Generate all processed data for the dashboard"""
        if self.df is None:
            return {"error": "No data loaded"}
            
        return {
            'overview': self.get_overview_stats(),
            'domains': self.get_domain_analysis(),
            'sections': self.get_section_analysis(), 
            'correlations': self.get_correlation_analysis(),
            'organizations': self.get_organization_benchmarks(),
            'demographics': self.get_demographic_analysis(),
            'metadata': {
                'processedAt': pd.Timestamp.now().isoformat(),
                'dataShape': self.df.shape,
                'columns': list(self.df.columns) if self.df is not None else []
            }
        }
    
    def save_processed_data(self, output_file='processed_hseg_data.json'):
        """Save processed data to JSON file"""
        data = self.generate_all_data()
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"Processed data saved to {output_file}")
        return data

if __name__ == "__main__":
    # Process the data
    processor = HSEGDataProcessor('hseg_final_dataset.csv')
    processed_data = processor.save_processed_data()
    print("Data processing complete!")
    if 'overview' in processed_data:
        print(f"Generated data for {processed_data['overview']['totalResponses']} responses")
