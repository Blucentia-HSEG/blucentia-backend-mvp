from flask import Flask, jsonify, request, send_from_directory, Response, stream_template
from flask_cors import CORS
import json
import os
from utils.data_visualization import main as generate_visualizations
import numpy as np
import math
from functools import wraps
import time
import hashlib
from collections import defaultdict

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

app = Flask(__name__, static_folder='static', static_url_path='/static')
CORS(app)

# Content Security Policy configuration
@app.after_request
def add_security_headers(response):
    # CSP configuration for dashboard with visualization libraries
    # Note: Some charting libraries may require 'unsafe-eval' for dynamic features
    # This is a balanced approach prioritizing functionality while maintaining security

    # Get environment to determine CSP strictness
    is_development = os.environ.get('FLASK_ENV') == 'development'

    if is_development:
        # More permissive CSP for development (includes unsafe-eval if needed)
        csp = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' "
            "https://cdn.jsdelivr.net "
            "https://cdnjs.cloudflare.com "
            "https://cdn.plot.ly "
            "https://unpkg.com "
            "https://ajax.googleapis.com; "
            "style-src 'self' 'unsafe-inline' "
            "https://cdn.jsdelivr.net "
            "https://cdnjs.cloudflare.com "
            "https://fonts.googleapis.com; "
            "font-src 'self' "
            "https://cdn.jsdelivr.net "
            "https://cdnjs.cloudflare.com "
            "https://fonts.gstatic.com; "
            "img-src 'self' data: blob:; "
            "connect-src 'self' "
            "https://cdn.jsdelivr.net "
            "https://cdnjs.cloudflare.com; "
            "worker-src 'self' blob:; "
            "child-src 'self' blob:; "
            "object-src 'none'; "
            "base-uri 'self';"
        )
    else:
        # Production CSP - more restrictive but may need unsafe-eval for charting libraries
        csp = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' "
            "https://cdn.jsdelivr.net "
            "https://cdnjs.cloudflare.com "
            "https://cdn.plot.ly "
            "https://unpkg.com "
            "https://ajax.googleapis.com; "
            "style-src 'self' 'unsafe-inline' "
            "https://cdn.jsdelivr.net "
            "https://cdnjs.cloudflare.com "
            "https://fonts.googleapis.com; "
            "font-src 'self' "
            "https://cdn.jsdelivr.net "
            "https://cdnjs.cloudflare.com "
            "https://fonts.gstatic.com; "
            "img-src 'self' data: blob:; "
            "connect-src 'self' "
            "https://cdn.jsdelivr.net "
            "https://cdnjs.cloudflare.com; "
            "worker-src 'self' blob:; "
            "child-src 'self' blob:; "
            "object-src 'none'; "
            "base-uri 'self';"
        )
    response.headers['Content-Security-Policy'] = csp

    # Additional security headers
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'

    return response

# Global data store
processed_data = None
raw_data = None
data_cache = {}

def cache_with_request_params(maxsize=128):
    """Cache decorator that includes request parameters in cache key"""
    def decorator(func):
        cache = {}
        access_order = []

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and query parameters
            query_params = dict(request.args) if request else {}
            cache_key = f"{func.__name__}_{hashlib.md5(str(sorted(query_params.items())).encode()).hexdigest()}"

            # Check if result is in cache
            if cache_key in cache:
                # Move to end (most recently used)
                access_order.remove(cache_key)
                access_order.append(cache_key)
                return cache[cache_key]

            # Execute function and cache result
            result = func(*args, **kwargs)

            # Manage cache size
            if len(cache) >= maxsize:
                # Remove least recently used item
                oldest_key = access_order.pop(0)
                del cache[oldest_key]

            cache[cache_key] = result
            access_order.append(cache_key)
            return result

        # Add cache clearing method
        wrapper.clear_cache = lambda: cache.clear() or access_order.clear()
        return wrapper
    return decorator

# HSEG Scoring Framework Constants
HSEG_CATEGORIES = {
    'Power Abuse & Suppression': {
        'questions': ['q1', 'q2', 'q3', 'q4'],
        'weight': 3.0,
        'risk_level': 'Critical'
    },
    'Failure of Accountability': {
        'questions': ['q11', 'q12', 'q13', 'q14'],
        'weight': 3.0,
        'risk_level': 'Critical'
    },
    'Discrimination & Exclusion': {
        'questions': ['q5', 'q6', 'q7'],
        'weight': 2.5,
        'risk_level': 'Severe'
    },
    'Mental Health Harm': {
        'questions': ['q15', 'q16', 'q17', 'q18'],
        'weight': 2.5,
        'risk_level': 'Severe'
    },
    'Manipulative Work Culture': {
        'questions': ['q8', 'q9', 'q10'],
        'weight': 2.0,
        'risk_level': 'Moderate'
    },
    'Erosion of Voice & Autonomy': {
        'questions': ['q19', 'q20', 'q21', 'q22'],
        'weight': 2.0,
        'risk_level': 'Moderate'
    }
}

HSEG_TIERS = {
    'Crisis': {'range': (7, 12), 'color': '#ef4444', 'description': 'Immediate Action Required', 'icon': '🔴'},
    'At Risk': {'range': (13, 16), 'color': '#f97316', 'description': 'Preventive Intervention Needed', 'icon': '🟠'},
    'Mixed': {'range': (17, 20), 'color': '#6b7280', 'description': 'Strategic Improvement Focus', 'icon': '⚫'},
    'Safe': {'range': (21, 24), 'color': '#3b82f6', 'description': 'Optimization and Maintenance', 'icon': '🔵'},
    'Thriving': {'range': (25, 28), 'color': '#22c55e', 'description': 'Excellence and Innovation', 'icon': '🟢'}
}

def calculate_hseg_score(record):
    """Calculate HSEG weighted score for a single record"""
    total_weighted_score = 0

    for category_name, category_info in HSEG_CATEGORIES.items():
        questions = category_info['questions']
        weight = category_info['weight']

        # Get scores for this category
        scores = [record.get(q, 0) for q in questions if record.get(q) is not None]
        if scores:
            category_avg = sum(scores) / len(scores)
            category_weighted_score = category_avg * weight
            total_weighted_score += category_weighted_score

    # Normalize to 28-point scale (max possible: 55.5, min possible: 13.875)
    normalized_score = (total_weighted_score / 55.5) * 28
    return round(normalized_score, 1)

def get_hseg_tier(score):
    """Get HSEG tier classification for a score"""
    for tier_name, tier_info in HSEG_TIERS.items():
        min_score, max_score = tier_info['range']
        if min_score <= score <= max_score:
            return {
                'tier': tier_name,
                'color': tier_info['color'],
                'description': tier_info['description'],
                'icon': tier_info['icon'],
                'score': score
            }

    # Fallback for edge cases
    if score < 7:
        return {'tier': 'Crisis', 'color': '#ef4444', 'description': 'Critical', 'icon': '🔴', 'score': score}
    else:
        return {'tier': 'Thriving', 'color': '#22c55e', 'description': 'Exceptional', 'icon': '🟢', 'score': score}

def calculate_category_scores(record):
    """Calculate individual category scores for a record"""
    category_scores = {}

    for category_name, category_info in HSEG_CATEGORIES.items():
        questions = category_info['questions']
        weight = category_info['weight']

        scores = [record.get(q, 0) for q in questions if record.get(q) is not None]
        if scores:
            category_avg = sum(scores) / len(scores)
            category_weighted_score = category_avg * weight

            category_scores[category_name] = {
                'average': round(category_avg, 2),
                'weighted_score': round(category_weighted_score, 2),
                'weight': weight,
                'question_count': len(scores),
                'risk_level': category_info['risk_level']
            }

    return category_scores

def merge_data_chunks_if_needed():
    """Merge data chunks if the full dataset file doesn't exist"""
    if not os.path.exists('hseg_final_dataset.json') and os.path.exists('data/metadata.json'):
        print("Full dataset not found. Merging data chunks...")
        try:
            import subprocess
            result = subprocess.run(['python', 'utils/merge_json.py'],
                                  capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print("Successfully merged data chunks!")
            else:
                print(f"Error merging chunks: {result.stderr}")
                return False
        except Exception as e:
            print(f"Failed to merge data chunks: {e}")
            return False
    return True

def load_data():
    """Load processed data and raw data"""
    global processed_data, raw_data

    # Load raw data - prioritize chunked data for GitHub compatibility
    raw_data = []

    if os.path.exists('data/metadata.json'):
        print("Loading data from chunks (GitHub-compatible mode)...")
        try:
            with open('data/metadata.json', 'r') as f:
                metadata = json.load(f)

            for chunk_file in metadata['chunk_files']:
                chunk_path = os.path.join('data', chunk_file)
                if os.path.exists(chunk_path):
                    print(f"Loading {chunk_file}...")
                    with open(chunk_path, 'r') as f:
                        chunk_data = json.load(f)
                        if isinstance(chunk_data, list):
                            raw_data.extend(chunk_data)
                        else:
                            raw_data.append(chunk_data)
            print(f"Successfully loaded {len(raw_data)} records from {len(metadata['chunk_files'])} chunks")
        except Exception as e:
            print(f"Error loading data from chunks: {e}")
            # Fallback to merged file if chunks fail
            if os.path.exists('hseg_final_dataset.json'):
                print("Falling back to merged dataset...")
                with open('hseg_final_dataset.json', 'r') as f:
                    raw_data = json.load(f)
                print(f"Loaded {len(raw_data)} raw records from merged file")
    elif os.path.exists('hseg_final_dataset.json'):
        print("Loading raw dataset from merged file...")
        with open('hseg_final_dataset.json', 'r') as f:
            raw_data = json.load(f)
        print(f"Loaded {len(raw_data)} raw records")
    else:
        print("Warning: No data files found. Please run 'python utils/merge_json.py' first.")
        raw_data = []

    # Try to load from processed JSON first
    if os.path.exists('processed_hseg_data.json'):
        with open('processed_hseg_data.json', 'r') as f:
            processed_data = json.load(f)
        print("Loaded processed data from JSON")
    else:
        # Process data from CSV
        try:
            processed_data = generate_visualizations()
            with open('processed_hseg_data.json', 'w') as f:
                json.dump(processed_data, f, indent=2, cls=NpEncoder)
            print("Processed data from CSV and saved to JSON")
        except Exception as e:
            print(f"Error processing data: {e}")
            processed_data = {"error": str(e)}

# Load data on startup
load_data()

@app.route('/')
def serve_index():
    """Serve the main dashboard HTML file"""
    return send_from_directory(app.root_path, 'index.html')

@app.route('/favicon.ico')
def favicon():
    # Avoid 404 noise for missing favicon; optionally serve a static icon later
    return ('', 204)

@app.route('/api/overview', methods=['GET'])
@cache_with_request_params(maxsize=128)
def get_overview():
    """Get overview statistics"""
    if processed_data and 'overview' in processed_data:
        return jsonify(processed_data['overview'])
    return jsonify({"error": "No data available"})

@app.route('/api/domains', methods=['GET'])
@cache_with_request_params(maxsize=128)
def get_domains():
    """Get domain analysis"""
    if not raw_data:
        return jsonify({})

    # Generate domain analysis from raw_data
    domain_data = {}

    for record in raw_data:
        domain = record.get('domain', 'Unknown')
        if domain not in domain_data:
            domain_data[domain] = {
                'count': 0,
                'culture_scores': [],
                'organizations': set(),
                'departments': set()
            }

        domain_data[domain]['count'] += 1
        domain_data[domain]['organizations'].add(record.get('organization_name', 'Unknown'))
        domain_data[domain]['departments'].add(record.get('department', 'Unknown'))

        # Calculate HSEG weighted culture score
        hseg_score = calculate_hseg_score(record)
        if hseg_score > 0:
            domain_data[domain]['culture_scores'].append(hseg_score)

    # Convert to final format
    result = {}
    for domain, data in domain_data.items():
        avg_score = sum(data['culture_scores']) / len(data['culture_scores']) if data['culture_scores'] else 0
        result[domain] = {
            'count': data['count'],
            'avg_culture_score': round(avg_score, 2),
            'organizations': len(data['organizations']),
            'departments': len(data['departments'])
        }

    return jsonify(result)

@app.route('/api/sections', methods=['GET'])
@cache_with_request_params(maxsize=128)
def get_sections():
    """Get section analysis based on survey structure"""
    domain = request.args.get('domain', 'all')

    if not raw_data:
        return jsonify({})

    # Use HSEG Categories for consistent scoring
    sections = {name: info['questions'] for name, info in HSEG_CATEGORIES.items()}

    # Calculate section scores
    section_data = {}

    for section_name, questions in sections.items():
        section_scores = []
        domain_scores = {}

        for record in raw_data:
            # Skip if domain filter is applied and doesn't match
            if domain.lower() != 'all' and record.get('domain', '').lower() != domain.lower():
                continue

            # Calculate section score for this record
            scores = [record.get(q, 0) for q in questions if record.get(q) is not None]
            if scores:
                section_score = sum(scores) / len(scores)
                section_scores.append(section_score)

                # Track by domain
                record_domain = record.get('domain', 'Unknown')
                if record_domain not in domain_scores:
                    domain_scores[record_domain] = []
                domain_scores[record_domain].append(section_score)

        if section_scores:
            # Calculate raw average score (1-4 scale) for radar charts
            raw_avg = sum(section_scores) / len(section_scores)

            # Calculate weighted HSEG contribution for this category
            category_info = HSEG_CATEGORIES[section_name]
            weighted_contribution = raw_avg * category_info['weight']

            # Calculate what this represents as a percentage of max possible weighted score
            max_weighted_for_category = 4 * category_info['weight']
            weighted_percentage = (weighted_contribution / max_weighted_for_category) * 100

            section_data[section_name] = {
                'overall_score': round(raw_avg, 3),  # Keep for compatibility with existing radar charts
                'weighted_score': round(weighted_contribution, 3),  # New: weighted HSEG contribution
                'weighted_percentage': round(weighted_percentage, 1),  # New: percentage of category max
                'category_weight': category_info['weight'],  # New: category weight
                'risk_level': category_info['risk_level'],  # New: risk classification
                'overall_std': round(np.std(section_scores), 3),
                'count': len(section_scores),
                'domain_breakdown': {
                    domain: {
                        'avg_score': round(sum(scores) / len(scores), 3),
                        'weighted_score': round((sum(scores) / len(scores)) * category_info['weight'], 3),
                        'count': len(scores)
                    } for domain, scores in domain_scores.items() if scores
                }
            }

    return jsonify(section_data)

@app.route('/api/sections/distributions', methods=['GET'])
@cache_with_request_params(maxsize=128)
def get_section_distributions():
    """Histogram distributions for each section score"""
    bins = int(request.args.get('bins', 20))
    if not raw_data:
        return jsonify({})

    sections = {
        'Power Abuse & Suppression': ['q1', 'q2', 'q3', 'q4'],
        'Discrimination & Exclusion': ['q5', 'q6', 'q7'],
        'Manipulative Work Culture': ['q8', 'q9', 'q10'],
        'Failure of Accountability': ['q11', 'q12', 'q13', 'q14'],
        'Mental Health Harm': ['q15', 'q16', 'q17', 'q18'],
        'Erosion of Voice & Autonomy': ['q19', 'q20', 'q21', 'q22']
    }

    section_scores = defaultdict(list)
    for record in raw_data:
        for name, qs in sections.items():
            scores = [record.get(q) for q in qs if record.get(q) is not None]
            if scores:
                section_scores[name].append(sum(scores) / len(scores))

    # Update to use HSEG scores instead of raw section scores
    hseg_scores_by_section = defaultdict(list)
    for record in raw_data:
        hseg_score = calculate_hseg_score(record)
        if hseg_score > 0:
            # Calculate which section scores contribute most to overall HSEG score
            category_contributions = calculate_category_scores(record)
            for section_name in sections.keys():
                if section_name in category_contributions:
                    # Use weighted contribution to HSEG score
                    contrib = category_contributions[section_name]['weighted_score']
                    hseg_scores_by_section[section_name].append(hseg_score)

    # Use 28-point scale for HSEG distribution
    edges = np.linspace(7, 28, bins + 1).tolist()
    dists = {}
    for name, scores in hseg_scores_by_section.items():
        if scores:
            counts, _ = np.histogram(scores, bins=bins, range=(7, 28))
            dists[name] = {
                'bins': edges,
                'counts': counts.tolist()
            }

    return jsonify({'bins': edges, 'sections': dists})

@app.route('/api/sections/correlation', methods=['GET'])
@cache_with_request_params(maxsize=128)
def get_sections_correlation():
    """Correlation matrix between section average scores per response"""
    if not raw_data:
        return jsonify({'labels': [], 'matrix': []})

    sections = {
        'Power Abuse & Suppression': ['q1', 'q2', 'q3', 'q4'],
        'Discrimination & Exclusion': ['q5', 'q6', 'q7'],
        'Manipulative Work Culture': ['q8', 'q9', 'q10'],
        'Failure of Accountability': ['q11', 'q12', 'q13', 'q14'],
        'Mental Health Harm': ['q15', 'q16', 'q17', 'q18'],
        'Erosion of Voice & Autonomy': ['q19', 'q20', 'q21', 'q22']
    }

    labels = list(sections.keys())
    rows = []
    for record in raw_data:
        row = []
        valid = True
        for _, qs in sections.items():
            vals = [record.get(q) for q in qs if record.get(q) is not None]
            if not vals:
                valid = False
                break
            row.append(sum(vals) / len(vals))
        if valid:
            rows.append(row)

    if len(rows) < 5:
        return jsonify({'labels': labels, 'matrix': []})

    arr = np.array(rows)
    corr = np.corrcoef(arr.T)
    return jsonify({'labels': labels, 'matrix': corr.tolist()})

@app.route('/api/correlations', methods=['GET'])
@cache_with_request_params(maxsize=128)
def get_correlations():
    """Get correlation analysis between questions"""
    if not raw_data:
        return jsonify([])

    # Create correlation matrix for all quantitative questions (Q1-Q22)
    questions = [f'q{i}' for i in range(1, 23)]

    # Prepare data matrix
    data_matrix = []
    for record in raw_data:
        row = []
        valid_row = True
        for q in questions:
            value = record.get(q)
            if value is not None and isinstance(value, (int, float)):
                row.append(float(value))
            else:
                valid_row = False
                break
        if valid_row:
            data_matrix.append(row)

    if len(data_matrix) < 10:  # Need minimum data for correlation
        return jsonify([])

    # Convert to numpy array and calculate correlations
    data_array = np.array(data_matrix)
    corr_matrix = np.corrcoef(data_array.T)

    # Extract correlation pairs
    correlations = []
    for i in range(len(questions)):
        for j in range(i + 1, len(questions)):
            corr_value = corr_matrix[i, j]
            if not np.isnan(corr_value):
                correlations.append({
                    'Question 1': questions[i].upper(),
                    'Question 2': questions[j].upper(),
                    'Correlation': round(float(corr_value), 4)
                })

    # Sort by absolute correlation value
    correlations.sort(key=lambda x: abs(x['Correlation']), reverse=True)

    return jsonify(correlations[:50])  # Return top 50 correlations

@app.route('/api/organizations', methods=['GET'])
@cache_with_request_params(maxsize=128)
def get_organizations():
    """Get organization analysis data with advanced filtering"""
    limit = int(request.args.get('limit', 20))
    min_responses = int(request.args.get('min_responses', 5))
    domain_filter = request.args.get('domain', 'all')
    score_range = request.args.get('score_range', 'all')
    size_filter = request.args.get('size_filter', 'all')

    if not raw_data:
        return jsonify([])

    # Generate organization data from raw_data
    org_data = {}

    # Group by organization and calculate metrics
    for record in raw_data:
        org_name = record.get('organization_name', 'Unknown')
        org_domain = record.get('domain', 'Unknown')

        # Apply domain filter early
        if domain_filter != 'all' and org_domain.lower() != domain_filter.lower():
            continue

        if org_name not in org_data:
            org_data[org_name] = {
                'name': org_name,
                'domain': org_domain,
                'employee_count': record.get('employee_count', 0),
                'responses': [],
                'culture_scores': []
            }

        # Calculate HSEG weighted culture score for this response
        hseg_score = calculate_hseg_score(record)
        if hseg_score > 0:
            org_data[org_name]['culture_scores'].append(hseg_score)
            org_data[org_name]['responses'].append(record)

    # Calculate final metrics and apply filters
    result = []
    for org_name, data in org_data.items():
        if len(data['culture_scores']) >= min_responses:
            avg_score = sum(data['culture_scores']) / len(data['culture_scores'])
            employee_count = data['employee_count']

            # Apply size filter
            if size_filter != 'all':
                if size_filter == 'large' and employee_count < 1000:
                    continue
                elif size_filter == 'medium' and not (100 <= employee_count < 1000):
                    continue
                elif size_filter == 'small' and employee_count >= 100:
                    continue

            # Apply HSEG score range filter (7-28 scale)
            if score_range != 'all':
                if score_range == 'high' and avg_score < 21.0:  # Safe + Thriving (21-28)
                    continue
                elif score_range == 'medium' and not (17.0 <= avg_score < 21.0):  # Mixed (17-20)
                    continue
                elif score_range == 'low' and avg_score >= 17.0:  # Crisis + At Risk (7-16)
                    continue

            result.append({
                'name': org_name,
                'domain': data['domain'],
                'employee_count': employee_count,
                'response_count': len(data['responses']),
                'culture_score': round(avg_score, 2),
                'score_std': round(np.std(data['culture_scores']), 2) if len(data['culture_scores']) > 1 else 0
            })

    # Sort by culture score and return top organizations
    result.sort(key=lambda x: x['culture_score'])
    return jsonify(result[:limit])

@app.route('/api/demographics', methods=['GET'])
@cache_with_request_params(maxsize=128)
def get_demographics():
    """Get comprehensive demographic analysis"""
    demographic_type = request.args.get('type', 'all')

    if not raw_data:
        return jsonify({})

    # Define demographic mappings using actual column names
    demographic_mappings = {
        'age_range': 'age_range',
        'gender_identity': 'gender_identity',
        'tenure_range': 'tenure_range',
        'tenure': 'tenure_range',  # alias for tenure_range
        'position_level': 'position_level',
        'domain_role': 'q32_domain_role',
        'supervises_others': 'supervises_others',
        'race_ethnicity': 'q28_race_ethnicity',
        'education_level': 'q29_education_level'
    }

    # Ensure columns exist in data
    if raw_data and len(raw_data) > 0:
        available_columns = set(raw_data[0].keys())
        demographic_mappings = {k: v for k, v in demographic_mappings.items() if v in available_columns}

    result = {}

    # Process each demographic type
    for demo_name, demo_column in demographic_mappings.items():
        if demographic_type != 'all' and demo_name != demographic_type:
            continue

        demographic_data = {}

        for record in raw_data:
            demo_value = record.get(demo_column, 'Unknown')
            if demo_value is None or demo_value == '':
                demo_value = 'Unknown'

            if demo_value not in demographic_data:
                demographic_data[demo_value] = {
                    'count': 0,
                    'culture_scores': [],
                    'domains': {},
                    'organizations': set()
                }

            demographic_data[demo_value]['count'] += 1
            demographic_data[demo_value]['organizations'].add(record.get('organization_name', 'Unknown'))

            # Track by domain
            domain = record.get('domain', 'Unknown')
            if domain not in demographic_data[demo_value]['domains']:
                demographic_data[demo_value]['domains'][domain] = 0
            demographic_data[demo_value]['domains'][domain] += 1

            # Calculate HSEG weighted culture score
            hseg_score = calculate_hseg_score(record)
            if hseg_score > 0:
                demographic_data[demo_value]['culture_scores'].append(hseg_score)

        # Calculate summary statistics
        summary = {}
        for demo_value, data in demographic_data.items():
            if data['culture_scores']:
                avg_score = sum(data['culture_scores']) / len(data['culture_scores'])
                summary[demo_value] = {
                    'count': data['count'],
                    'avg_culture_score': round(avg_score, 3),
                    'score_std': round(np.std(data['culture_scores']), 3) if len(data['culture_scores']) > 1 else 0,
                    'organizations': len(data['organizations']),
                    'domain_distribution': data['domains']
                }

        result[demo_name] = summary

    return jsonify(result if demographic_type == 'all' else result.get(demographic_type, {}))

@app.route('/api/metadata', methods=['GET'])
@cache_with_request_params(maxsize=128)
def get_metadata():
    """Get data metadata"""
    if processed_data and 'metadata' in processed_data:
        return jsonify(processed_data['metadata'])
    return jsonify({})

@app.route('/api/advanced/treemap', methods=['GET'])
@cache_with_request_params(maxsize=128)
def get_treemap_data():
    """Get hierarchical treemap data for domain > organization > department"""
    if not raw_data:
        return jsonify([])

    min_count = int(request.args.get('min_count', 3))
    domain_filter = request.args.get('domain', 'all')

    # Group by domain, organization, department
    grouped_data = {}
    for record in raw_data:
        domain = record.get('domain', 'Unknown')
        org = record.get('organization_name', 'Unknown')
        dept = record.get('department', 'Unknown')

        if domain_filter != 'all' and domain != domain_filter:
            continue

        key = (domain, org, dept)
        if key not in grouped_data:
            grouped_data[key] = {
                'responses': [],
                'count': 0
            }

        hseg_score = calculate_hseg_score(record)
        if hseg_score > 0:
            grouped_data[key]['responses'].append(hseg_score)
            grouped_data[key]['count'] += 1

    # Build treemap structure
    treemap_data = []
    for (domain, org, dept), data in grouped_data.items():
        if data['count'] >= min_count:
            avg_score = sum(data['responses']) / len(data['responses']) if data['responses'] else 0
            treemap_data.append({
                'domain': domain,
                'organization': org,
                'department': dept,
                'count': data['count'],
                'avg_score': round(avg_score, 2),
                'path': f"{domain}/{org}/{dept}"
            })

    # Sort by count for better visualization
    treemap_data.sort(key=lambda x: x['count'], reverse=True)

    return jsonify(treemap_data)

@app.route('/api/advanced/violin', methods=['GET'])
@cache_with_request_params(maxsize=128)
def get_violin_data():
    """Get violin plot data for section score distributions by domain"""
    if not raw_data:
        return jsonify({})

    domain_filter = request.args.get('domain', 'all')
    section_filter = request.args.get('section', 'all')

    # HSEG sections with their question ranges
    sections = {
        'Power Abuse Suppression': list(range(1, 5)),
        'Discrimination Exclusion': list(range(5, 8)),
        'Manipulative Work Culture': list(range(8, 11)),
        'Failure of Accountability': list(range(11, 15)),
        'Mental Health Harm': list(range(15, 19)),
        'Erosion of Voice Autonomy': list(range(19, 23))
    }

    violin_data = {}

    for record in raw_data:
        domain = record.get('domain', 'Unknown')
        if domain_filter != 'all' and domain != domain_filter:
            continue

        if domain not in violin_data:
            violin_data[domain] = {}

        for section_name, questions in sections.items():
            if section_filter != 'all' and section_name != section_filter:
                continue

            if section_name not in violin_data[domain]:
                violin_data[domain][section_name] = []

            # Calculate section score
            section_scores = [record.get(f'q{q}') for q in questions if record.get(f'q{q}') is not None]
            if section_scores:
                section_avg = sum(section_scores) / len(section_scores)
                violin_data[domain][section_name].append(section_avg)

    # Convert to format suitable for frontend
    result = {}
    for domain, sections_data in violin_data.items():
        result[domain] = {}
        for section, scores in sections_data.items():
            if scores:
                # Calculate distribution statistics
                scores_array = np.array(scores)
                result[domain][section] = {
                    'scores': scores,
                    'mean': float(np.mean(scores_array)),
                    'median': float(np.median(scores_array)),
                    'std': float(np.std(scores_array)),
                    'min': float(np.min(scores_array)),
                    'max': float(np.max(scores_array)),
                    'q25': float(np.percentile(scores_array, 25)),
                    'q75': float(np.percentile(scores_array, 75)),
                    'count': len(scores)
                }

    return jsonify(result)

@app.route('/api/debug/columns', methods=['GET'])
def debug_columns():
    """Debug: Get all available columns in raw data"""
    if not raw_data:
        return jsonify({"error": "No raw data available"})

    if len(raw_data) > 0:
        sample_record = raw_data[0]
        columns = list(sample_record.keys())

        # Find demographic-related columns
        demographic_columns = [col for col in columns if any(keyword in col.lower() for keyword in
                              ['age', 'gender', 'race', 'ethnic', 'tenure', 'position', 'level', 'supervisor', 'q2', 'q3', 'demo'])]

        return jsonify({
            "all_columns": columns,
            "demographic_columns": demographic_columns,
            "total_records": len(raw_data),
            "sample_record": sample_record
        })

    return jsonify({"error": "No data records found"})

@app.route('/api/data/paginated', methods=['GET'])
@cache_with_request_params(maxsize=128)
def get_paginated_data():
    """Get paginated raw data for virtual scrolling"""
    if not raw_data:
        return jsonify({"error": "No raw data available", "data": [], "total": 0})

    page = int(request.args.get('page', 1))
    limit = int(request.args.get('limit', 50))
    search = request.args.get('search', '').lower()
    organization = request.args.get('organization', '')

    # Filter data
    filtered_data = raw_data

    if search:
        filtered_data = [
            record for record in filtered_data
            if search in record.get('organization_name', '').lower()
            or search in record.get('domain', '').lower()
            or search in record.get('department', '').lower()
        ]

    if organization and organization != 'all':
        filtered_data = [
            record for record in filtered_data
            if record.get('organization_name', '').lower() == organization.lower()
        ]

    # Pagination
    total = len(filtered_data)
    start = (page - 1) * limit
    end = start + limit
    paginated_data = filtered_data[start:end]

    # Format response data (only include essential fields for performance)
    formatted_data = []
    for record in paginated_data:
        formatted_record = {
            'response_id': record.get('response_id'),
            'organization_name': record.get('organization_name'),
            'domain': record.get('domain'),
            'department': record.get('department'),
            'position_level': record.get('position_level'),
            'submission_date': record.get('submission_date'),
            'culture_score': calculate_hseg_score(record)
        }
        formatted_data.append(formatted_record)

    return jsonify({
        "data": formatted_data,
        "total": total,
        "page": page,
        "pages": math.ceil(total / limit),
        "has_next": end < total,
        "has_prev": page > 1
    })

@app.route('/api/data/stream')
def stream_data():
    """Stream data in chunks for real-time loading"""
    chunk_size = int(request.args.get('chunk_size', 100))

    def generate():
        if not raw_data:
            yield f"data: {json.dumps({'error': 'No data available'})}\n\n"
            return

        total_chunks = math.ceil(len(raw_data) / chunk_size)

        for i in range(0, len(raw_data), chunk_size):
            chunk = raw_data[i:i + chunk_size]
            chunk_data = {
                'chunk': i // chunk_size + 1,
                'total_chunks': total_chunks,
                'data': chunk[:10],  # Send only first 10 items per chunk for demo
                'progress': min(100, ((i + chunk_size) / len(raw_data)) * 100)
            }
            yield f"data: {json.dumps(chunk_data, cls=NpEncoder)}\n\n"
            time.sleep(0.1)  # Small delay to simulate streaming

        yield f"data: {json.dumps({'complete': True})}\n\n"

    return Response(generate(), mimetype='text/event-stream',
                   headers={'Cache-Control': 'no-cache'})

@app.route('/api/organizations/list', methods=['GET'])
@cache_with_request_params(maxsize=128)
def get_organizations_list():
    """Get list of organizations for filters"""
    if not raw_data:
        return jsonify([])

    organizations = list(set(record.get('organization_name', '') for record in raw_data if record.get('organization_name')))
    organizations.sort()

    return jsonify([{'value': org, 'label': org} for org in organizations])

@app.route('/api/domains/list', methods=['GET'])
@cache_with_request_params(maxsize=128)
def get_domains_list():
    """Get list of domains for filters"""
    if not raw_data:
        return jsonify([])

    domains = list(set(record.get('domain', '') for record in raw_data if record.get('domain')))
    domains.sort()

    return jsonify([{'value': domain, 'label': domain} for domain in domains])

@app.route('/api/stats/quick', methods=['GET'])
@cache_with_request_params(maxsize=128)
def get_quick_stats():
    """Get quick stats for KPI cards"""
    start_time = time.time()

    if not raw_data:
        return jsonify({"error": "No data available"})

    # Calculate quick stats
    total_responses = len(raw_data)
    organizations = len(set(record.get('organization_name') for record in raw_data if record.get('organization_name')))
    domains = len(set(record.get('domain') for record in raw_data if record.get('domain')))

    # Calculate average HSEG weighted culture score
    hseg_scores = []
    for record in raw_data:
        hseg_score = calculate_hseg_score(record)
        if hseg_score > 0:
            hseg_scores.append(hseg_score)

    avg_culture_score = round(sum(hseg_scores) / len(hseg_scores), 1) if hseg_scores else 0

    # Calculate response time
    response_time = round((time.time() - start_time) * 1000, 1)

    return jsonify({
        "total_responses": total_responses,
        "num_organizations": organizations,
        "num_domains": domains,
        "overall_culture_score": avg_culture_score,
        "response_time_ms": response_time,
        "data_freshness": "Live",
        "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
    })

@app.route('/api/analytics/trend', methods=['GET'])
@cache_with_request_params(maxsize=128)
def get_trend_data():
    """Get trend data for charts"""
    days = int(request.args.get('days', 30))
    metric = request.args.get('metric', 'culture_score')
    granularity = request.args.get('granularity', 'weekly')
    smoothing = request.args.get('smoothing', 'none')

    if not raw_data:
        return jsonify({"labels": [], "datasets": []})

    # Group data by submission date
    from collections import defaultdict
    import datetime

    # Define survey sections
    sections = {
        'Power Abuse & Suppression': ['q1', 'q2', 'q3', 'q4'],
        'Discrimination & Exclusion': ['q5', 'q6', 'q7'],
        'Manipulative Work Culture': ['q8', 'q9', 'q10'],
        'Failure of Accountability': ['q11', 'q12', 'q13', 'q14'],
        'Mental Health Harm': ['q15', 'q16', 'q17', 'q18'],
        'Erosion of Voice & Autonomy': ['q19', 'q20', 'q21', 'q22']
    }

    date_scores = defaultdict(list)
    section_date_scores = defaultdict(lambda: defaultdict(list))

    for record in raw_data:
        date_str = record.get('submission_date', '')
        if date_str:
            try:
                date_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d')

                # Apply granularity grouping
                if granularity == 'monthly':
                    date_key = date_obj.strftime('%Y-%m')
                elif granularity == 'weekly':
                    # Get start of week (Monday)
                    week_start = date_obj - datetime.timedelta(days=date_obj.weekday())
                    date_key = week_start.strftime('%Y-%m-%d')
                else:  # daily
                    date_key = date_str

                if metric == 'response_count':
                    date_scores[date_key].append(1)
                elif metric == 'section_scores':
                    # Calculate HSEG weighted section scores for this record
                    for section_name, questions in sections.items():
                        scores = [record.get(q, 0) for q in questions if record.get(q) is not None]
                        if scores:
                            category_avg = sum(scores) / len(scores)
                            # Apply HSEG weighting to get proper scale
                            category_weight = HSEG_CATEGORIES.get(section_name, {}).get('weight', 1.0)
                            weighted_score = category_avg * category_weight
                            # Convert to HSEG scale: each category contributes to the overall 28-point scale
                            # The weighted score should be normalized to show its contribution
                            hseg_contribution = (weighted_score / 55.5) * 28
                            section_date_scores[section_name][date_key].append(round(hseg_contribution, 2))
                else:  # culture_score
                    hseg_score = calculate_hseg_score(record)
                    if hseg_score > 0:
                        date_scores[date_key].append(hseg_score)
            except:
                pass

    # Apply smoothing function
    def apply_smoothing(data, method):
        if method == 'none' or len(data) < 3:
            return data
        elif method == 'ma7':
            window = min(7, len(data))
        elif method == 'ma30':
            window = min(30, len(data))
        else:
            return data

        smoothed = []
        for i in range(len(data)):
            start_idx = max(0, i - window // 2)
            end_idx = min(len(data), i + window // 2 + 1)
            smoothed.append(sum(data[start_idx:end_idx]) / (end_idx - start_idx))
        return smoothed

    # Create trend data
    if metric == 'section_scores':
        # Return multiple datasets for each section
        datasets = []
        colors = ['#ef4444', '#f97316', '#eab308', '#84cc16', '#22c55e', '#06b6d4']

        all_dates = set()
        for section_scores in section_date_scores.values():
            all_dates.update(section_scores.keys())

        labels = sorted(list(all_dates))[-days:]

        for i, (section_name, section_scores) in enumerate(section_date_scores.items()):
            trend_data = []
            for date in labels:
                if date in section_scores and section_scores[date]:
                    avg_score = sum(section_scores[date]) / len(section_scores[date])
                    trend_data.append(round(avg_score, 2))
                else:
                    trend_data.append(0)

            # Apply smoothing
            trend_data = apply_smoothing(trend_data, smoothing)

            datasets.append({
                "label": section_name,
                "data": trend_data,
                "borderColor": colors[i % len(colors)],
                "backgroundColor": f"{colors[i % len(colors)]}20",
                "tension": 0.3
            })
    else:
        labels = sorted(date_scores.keys())[-days:]
        if metric == 'response_count':
            trend_data = [
                sum(date_scores[date]) if date_scores[date] else 0
                for date in labels
            ]
            trend_data = apply_smoothing(trend_data, smoothing)
            datasets = [{"label": "Responses", "data": trend_data, "borderColor": "#10b981", "backgroundColor": "rgba(16,185,129,0.1)", "tension": 0.3}]
        else:
            trend_data = [
                round(sum(date_scores[date]) / len(date_scores[date]), 2) if date_scores[date] else 0
                for date in labels
            ]
            trend_data = apply_smoothing(trend_data, smoothing)
            datasets = [{"label": "Culture Score", "data": trend_data, "borderColor": "#2563eb", "backgroundColor": "rgba(37, 99, 235, 0.1)", "tension": 0.4}]

    return jsonify({
        "labels": labels,
        "datasets": datasets
    })

@app.route('/api/advanced/hierarchical', methods=['GET'])
@cache_with_request_params(maxsize=128)
def get_hierarchical_data():
    """Get hierarchical data for treemap and sunburst visualizations"""
    if not raw_data:
        return jsonify([])

    # Create hierarchical structure: Domain > Organization > Department
    min_count = int(request.args.get('min_count', 3))
    domain_filter = request.args.get('domain')
    hierarchy = {}

    for record in raw_data:
        domain = record.get('domain', 'Unknown')
        if domain_filter and domain_filter.lower() != 'all' and domain.lower() != domain_filter.lower():
            continue
        org = record.get('organization_name', 'Unknown')
        dept = record.get('department', 'Unknown')

        if domain not in hierarchy:
            hierarchy[domain] = {}
        if org not in hierarchy[domain]:
            hierarchy[domain][org] = {}
        if dept not in hierarchy[domain][org]:
            hierarchy[domain][org][dept] = {
                'count': 0,
                'culture_scores': []
            }

        hierarchy[domain][org][dept]['count'] += 1

        # Calculate HSEG weighted culture score
        hseg_score = calculate_hseg_score(record)
        if hseg_score > 0:
            hierarchy[domain][org][dept]['culture_scores'].append(hseg_score)

    # Convert to flat array for visualization
    result = []
    for domain, orgs in hierarchy.items():
        for org, depts in orgs.items():
            for dept, data in depts.items():
                if data['count'] >= min_count:  # Minimum threshold
                    avg_score = sum(data['culture_scores']) / len(data['culture_scores']) if data['culture_scores'] else 0
                    result.append({
                        'domain': domain,
                        'organization': org,
                        'department': dept,
                        'count': data['count'],
                        'avg_culture_score': round(avg_score, 2)
                    })

    return jsonify(result)

@app.route('/api/advanced/sunburst', methods=['GET'])
@cache_with_request_params(maxsize=128)
def get_sunburst_data():
    """Sunburst-like nested counts: Domain > Position Level > Department"""
    if not raw_data:
        return jsonify([])

    min_count = int(request.args.get('min_count', 3))
    domain_filter = request.args.get('domain')

    counts = defaultdict(int)
    avg_scores = defaultdict(list)
    for record in raw_data:
        dom = record.get('domain', 'Unknown')
        if domain_filter and domain_filter.lower() != 'all' and dom.lower() != domain_filter.lower():
            continue
        pos = record.get('position_level', 'Unknown')
        dept = record.get('department', 'Unknown')
        scores = [record.get(f'q{i}') for i in range(1,23) if record.get(f'q{i}') is not None]
        s = sum(scores)/len(scores) if scores else None
        counts[(dom,pos,dept)] += 1
        if s is not None:
            avg_scores[(dom,pos,dept)].append(s)

    # build flat list
    results = []
    for (dom,pos,dept), cnt in counts.items():
        if cnt < min_count:
            continue
        scores = avg_scores.get((dom,pos,dept), [])
        avg = sum(scores)/len(scores) if scores else 0
        results.append({'domain': dom, 'position_level': pos, 'department': dept, 'count': cnt, 'avg_culture_score': round(avg,2)})

    return jsonify(results)

@app.route('/api/advanced/ridge', methods=['GET'])
@cache_with_request_params(maxsize=128)
def get_ridge_data():
    """Get ridge plot data for section distributions by domain (6.1) - Enhanced version"""
    domain_filter = request.args.get('domain', 'all')
    bins = int(request.args.get('bins', 30))

    if not raw_data:
        return jsonify({'sections': [], 'domains': []})

    # Define sections
    sections = {
        'Power Abuse & Suppression': ['q1', 'q2', 'q3', 'q4'],
        'Discrimination & Exclusion': ['q5', 'q6', 'q7'],
        'Manipulative Work Culture': ['q8', 'q9', 'q10'],
        'Failure of Accountability': ['q11', 'q12', 'q13', 'q14'],
        'Mental Health Harm': ['q15', 'q16', 'q17', 'q18'],
        'Erosion of Voice & Autonomy': ['q19', 'q20', 'q21', 'q22']
    }

    # Get unique domains
    domains = sorted(list(set(record.get('domain', 'Unknown') for record in raw_data)))
    if domain_filter != 'all':
        domains = [d for d in domains if d.lower() == domain_filter.lower()]

    ridge_data = {'domains': [], 'sections': list(sections.keys())}

    for domain in domains:
        domain_records = [r for r in raw_data if r.get('domain', 'Unknown') == domain]

        domain_data = {'domain': domain, 'distributions': []}

        for section_name, questions in sections.items():
            scores = []
            for record in domain_records:
                section_scores = [record.get(q, 0) for q in questions if record.get(q) is not None]
                if section_scores:
                    scores.append(sum(section_scores) / len(section_scores))

            if scores:
                # Create histogram
                hist, bin_edges = np.histogram(scores, bins=bins, range=(1, 4))
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                domain_data['distributions'].append({
                    'section': section_name,
                    'x': bin_centers.tolist(),
                    'y': hist.tolist(),
                    'density': (hist / np.max(hist)).tolist() if np.max(hist) > 0 else hist.tolist(),
                    'count': len(scores),
                    'mean': np.mean(scores),
                    'std': np.std(scores)
                })
            else:
                domain_data['distributions'].append({
                    'section': section_name,
                    'x': [],
                    'y': [],
                    'density': [],
                    'count': 0,
                    'mean': 0,
                    'std': 0
                })

        ridge_data['domains'].append(domain_data)

    return jsonify(ridge_data)

@app.route('/api/tenure/matrix', methods=['GET'])
@cache_with_request_params(maxsize=128)
def get_tenure_matrix():
    """Mean overall culture score heatmap for Domain x Tenure"""
    if not raw_data:
        return jsonify({'domains': [], 'tenure': [], 'matrix': []})

    tenure_order = ['<1_year', '1-3_years', '3-7_years', '7-15_years', '15+_years']
    by_key = defaultdict(list)
    for record in raw_data:
        dom = record.get('domain') or 'Unknown'
        ten = record.get('tenure_range') or record.get('q30') or 'Unknown'
        # Use HSEG score calculation instead of raw average
        hseg_score = calculate_hseg_score(record)
        if hseg_score <= 0:
            continue
        key = (dom, ten)
        by_key[key].append(hseg_score)

    domains = sorted(list({k[0] for k in by_key.keys()}))
    tenures = [t for t in tenure_order if any(k[1]==t for k in by_key.keys())]
    if not tenures:
        tenures = sorted(list({k[1] for k in by_key.keys()}))

    matrix = []
    for d in domains:
        row = []
        for t in tenures:
            vals = by_key.get((d,t), [])
            row.append(round(float(sum(vals)/len(vals)), 3) if vals else None)
        matrix.append(row)

    return jsonify({'domains': domains, 'tenure': tenures, 'matrix': matrix})

@app.route('/api/advanced/network', methods=['GET'])
@cache_with_request_params(maxsize=128)
def get_network_data():
    """Get organization network data for similarity analysis"""
    if not raw_data:
        return jsonify({'nodes': [], 'links': []})

    # Calculate organization profiles for network analysis
    min_responses = int(request.args.get('min_responses', 20))

    org_profiles = {}
    for record in raw_data:
        org = record.get('organization_name', 'Unknown')
        if org not in org_profiles:
            org_profiles[org] = {
                'domain': record.get('domain', 'Unknown'),
                'employee_count': record.get('employee_count', 0),
                'responses': 0,
                'section_scores': {f'q{i}': [] for i in range(1, 23)}
            }

        org_profiles[org]['responses'] += 1

        # Collect scores for each question
        for i in range(1, 23):
            q_val = record.get(f'q{i}')
            if q_val is not None:
                org_profiles[org]['section_scores'][f'q{i}'].append(q_val)

    # Filter organizations with sufficient responses
    filtered_orgs = {org: data for org, data in org_profiles.items()
                    if data['responses'] >= min_responses}

    # Calculate average scores for each organization
    nodes = []
    for org, data in filtered_orgs.items():
        avg_scores = {}
        for q, scores in data['section_scores'].items():
            if scores:
                avg_scores[q] = sum(scores) / len(scores)

        # Calculate HSEG score for the organization
        org_hseg_scores = []
        # Get all records for this organization and calculate HSEG scores
        for record in raw_data:
            if record.get('organization_name') == org:
                hseg_score = calculate_hseg_score(record)
                if hseg_score > 0:
                    org_hseg_scores.append(hseg_score)

        overall_score = sum(org_hseg_scores) / len(org_hseg_scores) if org_hseg_scores else 0

        nodes.append({
            'id': org,
            'domain': data['domain'],
            'employee_count': data['employee_count'],
            'responses': data['responses'],
            'culture_score': round(overall_score, 1),
            'size': min(50, max(10, data['responses'] / 10))  # Scale for visualization
        })

    # Calculate similarity links (simplified version)
    links = []
    orgs = list(filtered_orgs.keys())
    for i, org1 in enumerate(orgs):
        for j, org2 in enumerate(orgs[i+1:], i+1):
            # Calculate simple similarity based on culture scores
            score1 = nodes[i]['culture_score']
            score2 = nodes[j]['culture_score']
            similarity = 1 - abs(score1 - score2) / 4  # Normalize by max possible difference

            if similarity > 0.8:  # Only include high similarity links
                links.append({
                    'source': org1,
                    'target': org2,
                    'similarity': round(similarity, 3),
                    'weight': similarity
                })

    return jsonify({'nodes': nodes, 'links': links})

def _simple_kmeans(X, k=3, max_iters=100, seed=42):
    """Lightweight KMeans on numpy arrays (for elbow only)."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    centroids = X[rng.choice(n, k, replace=False)]
    for _ in range(max_iters):
        dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        new_centroids = np.array([X[labels==i].mean(axis=0) if np.any(labels==i) else centroids[i] for i in range(k)])
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
    inertia = float(((X - centroids[labels])**2).sum())
    return labels, centroids, inertia

@app.route('/api/advanced/clustering', methods=['GET'])
@cache_with_request_params(maxsize=128)
def get_clustering_data():
    """Get clustering analysis data for PCA and t-SNE visualizations"""
    if not raw_data:
        return jsonify({})

    # Prepare data for clustering
    data_matrix = []
    labels = []

    questions = [f'q{i}' for i in range(1, 23)]

    for record in raw_data:
        row = []
        valid_row = True
        for q in questions:
            value = record.get(q)
            if value is not None and isinstance(value, (int, float)):
                row.append(float(value))
            else:
                valid_row = False
                break

        if valid_row:
            data_matrix.append(row)

            # Calculate HSEG score for this record
            hseg_score = calculate_hseg_score(record)
            hseg_tier_info = get_hseg_tier(hseg_score)

            labels.append({
                'domain': record.get('domain', 'Unknown'),
                'organization': record.get('organization_name', 'Unknown'),
                'department': record.get('department', 'Unknown'),
                'hseg_score': round(hseg_score, 2),
                'hseg_tier': hseg_tier_info['tier'],
                'hseg_tier_info': hseg_tier_info
            })

    if len(data_matrix) < 100:  # Need sufficient data
        return jsonify({'error': 'Insufficient data for clustering analysis'})

    # Sample data for performance (random sample)
    import random
    if len(data_matrix) > 2000:
        indices = random.sample(range(len(data_matrix)), 2000)
        data_matrix = [data_matrix[i] for i in indices]
        labels = [labels[i] for i in indices]

    # Simple PCA using numpy (since sklearn might not be available)
    data_array = np.array(data_matrix)

    # Center the data
    data_centered = data_array - np.mean(data_array, axis=0)

    # Compute covariance matrix
    cov_matrix = np.cov(data_centered.T)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort by eigenvalues (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Project data onto first 2 principal components
    pca_result = data_centered @ eigenvectors[:, :2]

    # Simple elbow on PCA result for k=2..8
    inertias = []
    try:
        X = pca_result[:, :2] if pca_result.shape[1] >= 2 else pca_result
        for k in range(2, 9):
            _, _, inertia = _simple_kmeans(X, k=k)
            inertias.append({'k': k, 'inertia': inertia})
    except Exception:
        inertias = []

    # Prepare response
    result = {
        'pca_data': [
            {
                'x': float(pca_result[i, 0]),
                'y': float(pca_result[i, 1]),
                'domain': labels[i]['domain'],
                'organization': labels[i]['organization'],
                'department': labels[i]['department'],
                'hseg_score': labels[i]['hseg_score'],
                'hseg_tier': labels[i]['hseg_tier'],
                'hseg_tier_info': labels[i]['hseg_tier_info']
            }
            for i in range(len(pca_result))
        ],
        'explained_variance': [float(val) for val in eigenvalues[:5]],
        'loadings': {
            'pc1': [float(eigenvectors[i, 0]) for i in range(len(questions))],
            'pc2': [float(eigenvectors[i, 1]) for i in range(len(questions))],
            'questions': questions
        },
        'elbow': inertias,
        'hseg_context': {
            'tier_counts': {tier: sum(1 for label in labels if label['hseg_tier'] == tier)
                          for tier in ['Crisis', 'At Risk', 'Mixed', 'Safe', 'Thriving']},
            'score_stats': {
                'min': min([labels[i]['hseg_score'] for i in range(len(labels))]),
                'max': max([labels[i]['hseg_score'] for i in range(len(labels))]),
                'mean': round(sum([labels[i]['hseg_score'] for i in range(len(labels))]) / len(labels), 2)
            },
            'total_samples': len(labels)
        }
    }

    return jsonify(result)

@app.route('/api/organizations/summary', methods=['GET'])
@cache_with_request_params(maxsize=128)
def get_org_summary():
    """Department and position distributions for an organization"""
    org = request.args.get('organization', 'all')
    if not raw_data:
        return jsonify({'departments': {}, 'positions': {}})

    if org.lower() == 'all':
        data = raw_data
    else:
        data = [r for r in raw_data if (r.get('organization_name') or '').lower() == org.lower()]

    dept = defaultdict(int)
    pos = defaultdict(int)
    for r in data:
        dept[r.get('department') or 'Unknown'] += 1
        pos[r.get('position_level') or 'Unknown'] += 1

    return jsonify({
        'departments': dict(sorted(dept.items(), key=lambda x: x[1], reverse=True)[:15]),
        'positions': dict(sorted(pos.items(), key=lambda x: x[1], reverse=True)[:15])
    })

@app.route('/api/distributions/overall', methods=['GET'])
@cache_with_request_params(maxsize=128)
def get_overall_distribution():
    """Histogram for overall culture score, and per-domain distributions"""
    bins = int(request.args.get('bins', 20))
    if not raw_data:
        return jsonify({'bins': [], 'overall': [], 'by_domain': {}})

    scores = []
    by_domain = defaultdict(list)
    for r in raw_data:
        vals = [r.get(f'q{i}') for i in range(1,23) if r.get(f'q{i}') is not None]
        if not vals:
            continue
        s = sum(vals)/len(vals)
        scores.append(s)
        d = r.get('domain') or 'Unknown'
        by_domain[d].append(s)

    # Update to use HSEG scoring for distribution analysis
    hseg_scores = []
    hseg_by_domain = defaultdict(list)

    for record in raw_data:
        hseg_score = calculate_hseg_score(record)
        if hseg_score > 0:
            hseg_scores.append(hseg_score)
            domain = record.get('domain') or 'Unknown'
            hseg_by_domain[domain].append(hseg_score)

    # Use 28-point HSEG scale instead of 1-4 scale
    edges = np.linspace(7, 28, bins + 1).tolist()
    counts, _ = np.histogram(hseg_scores, bins=bins, range=(7, 28))
    per_domain = {}
    for d, arr in hseg_by_domain.items():
        if arr:
            c, _ = np.histogram(arr, bins=bins, range=(7, 28))
            per_domain[d] = c.tolist()

    return jsonify({'bins': edges, 'overall': counts.tolist(), 'by_domain': per_domain})

@app.route('/api/distributions/responses', methods=['GET'])
@cache_with_request_params(maxsize=128)
def get_response_distribution():
    """Histogram of response counts per organization (distribution of participation)."""
    bins = int(request.args.get('bins', 20))
    if not raw_data:
        return jsonify({'bins': [], 'counts': []})

    from collections import defaultdict
    org_counts = defaultdict(int)
    for r in raw_data:
        org = r.get('organization_name') or 'Unknown'
        org_counts[org] += 1

    values = list(org_counts.values())
    if not values:
        return jsonify({'bins': [], 'counts': []})

    edges = np.linspace(min(values), max(values), bins + 1).tolist()
    counts, _ = np.histogram(values, bins=bins, range=(min(values), max(values)))
    return jsonify({'bins': edges, 'counts': counts.tolist()})

@app.route('/api/organizations/sections', methods=['GET'])
@cache_with_request_params(maxsize=128)
def get_organization_sections():
    """Get section analysis for specific organization"""
    org_name = request.args.get('organization', 'all')

    if not raw_data:
        return jsonify({})

    # Use HSEG Categories for consistent scoring
    sections = {name: info['questions'] for name, info in HSEG_CATEGORIES.items()}

    # Filter data for specific organization
    if org_name.lower() == 'all':
        org_data = raw_data
    else:
        org_data = [record for record in raw_data if record.get('organization_name', '').lower() == org_name.lower()]

    if not org_data:
        return jsonify({})

    # Calculate section scores for this organization
    section_results = {}

    for section_name, questions in sections.items():
        section_scores = []

        for record in org_data:
            scores = [record.get(q, 0) for q in questions if record.get(q) is not None]
            if scores:
                section_score = sum(scores) / len(scores)
                section_scores.append(section_score)

        if section_scores:
            # Calculate raw average score (1-4 scale)
            raw_avg = sum(section_scores) / len(section_scores)

            # Calculate weighted HSEG contribution for this category
            category_info = HSEG_CATEGORIES[section_name]
            weighted_contribution = raw_avg * category_info['weight']

            # Calculate what this represents as a percentage of max possible weighted score
            max_weighted_for_category = 4 * category_info['weight']
            weighted_percentage = (weighted_contribution / max_weighted_for_category) * 100

            section_results[section_name.replace(' & ', ' ')] = {
                'score': round(raw_avg, 3),  # Keep for compatibility
                'weighted_score': round(weighted_contribution, 3),  # HSEG weighted contribution
                'weighted_percentage': round(weighted_percentage, 1),  # Percentage of category max
                'category_weight': category_info['weight'],  # Category weight
                'risk_level': category_info['risk_level'],  # Risk classification
                'std': round(np.std(section_scores), 3),
                'count': len(section_scores)
            }

    return jsonify(section_results)


@app.route('/api/refresh', methods=['POST'])
def refresh_data():
    """Refresh data from CSV"""
    global processed_data, raw_data, data_cache
    try:
        # Clear cache
        data_cache.clear()

        # Reload data
        load_data()

        return jsonify({"success": True, "message": "Data refreshed successfully"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/insights/hseg', methods=['GET'])
@cache_with_request_params(maxsize=32)
def get_hseg_insights():
    """Get comprehensive HSEG framework insights"""
    if not raw_data:
        return jsonify({})

    domain_filter = request.args.get('domain', 'all')
    org_filter = request.args.get('organization', 'all')

    # Filter data
    filtered_data = raw_data
    if domain_filter.lower() != 'all':
        filtered_data = [r for r in filtered_data if r.get('domain', '').lower() == domain_filter.lower()]
    if org_filter.lower() != 'all':
        filtered_data = [r for r in filtered_data if r.get('organization_name', '').lower() == org_filter.lower()]

    if not filtered_data:
        return jsonify({'error': 'No data available for filters'})

    # Calculate overall HSEG scores and tier distribution
    hseg_scores = []
    tier_distribution = {tier: 0 for tier in HSEG_TIERS.keys()}
    category_scores = {cat: [] for cat in HSEG_CATEGORIES.keys()}

    for record in filtered_data:
        # Calculate HSEG score
        hseg_score = calculate_hseg_score(record)
        if hseg_score > 0:
            hseg_scores.append(hseg_score)

            # Get tier classification
            tier_info = get_hseg_tier(hseg_score)
            tier_distribution[tier_info['tier']] += 1

            # Calculate category scores
            for category_name, category_info in HSEG_CATEGORIES.items():
                questions = category_info['questions']
                scores = [record.get(q, 0) for q in questions if record.get(q) is not None]
                if scores:
                    category_avg = sum(scores) / len(scores)
                    category_scores[category_name].append(category_avg)

    # Calculate summary statistics
    if not hseg_scores:
        return jsonify({'error': 'No valid scores found'})

    overall_score = sum(hseg_scores) / len(hseg_scores)
    overall_tier = get_hseg_tier(overall_score)

    # Calculate category insights
    category_insights = {}
    for category_name, scores in category_scores.items():
        if scores:
            category_info = HSEG_CATEGORIES[category_name]
            avg_score = sum(scores) / len(scores)
            weighted_score = avg_score * category_info['weight']

            category_insights[category_name] = {
                'average_score': round(avg_score, 2),
                'weighted_score': round(weighted_score, 2),
                'weight': category_info['weight'],
                'risk_level': category_info['risk_level'],
                'response_count': len(scores),
                'score_range': f"{min(scores):.1f} - {max(scores):.1f}",
                'standard_deviation': round(np.std(scores), 2)
            }

    # Risk assessment
    at_risk_count = tier_distribution['Crisis'] + tier_distribution['At Risk']
    risk_percentage = (at_risk_count / len(hseg_scores)) * 100

    # Key insights and recommendations
    key_insights = []
    recommendations = []

    # Overall tier assessment
    if overall_tier['tier'] == 'Crisis':
        key_insights.append("🚨 Organization is in CRISIS state requiring immediate intervention")
        recommendations.append("Conduct emergency leadership review and implement crisis response protocols")
    elif overall_tier['tier'] == 'At Risk':
        key_insights.append("⚠️ Organization shows warning signs requiring preventive action")
        recommendations.append("Develop targeted improvement plan focusing on highest-risk categories")
    elif overall_tier['tier'] == 'Mixed':
        key_insights.append("📊 Mixed results indicate uneven experiences across organization")
        recommendations.append("Focus on reducing disparities and strengthening weak areas")
    elif overall_tier['tier'] == 'Safe':
        key_insights.append("✅ Organization shows strong foundation with room for optimization")
        recommendations.append("Implement continuous improvement programs and protect vulnerable groups")
    else:  # Thriving
        key_insights.append("🌟 Organization demonstrates excellence across all HSEG dimensions")
        recommendations.append("Share best practices and maintain high standards through innovation")

    # Category-specific insights
    critical_categories = [cat for cat, info in category_insights.items()
                          if HSEG_CATEGORIES[cat]['risk_level'] == 'Critical' and info['average_score'] < 2.5]

    if critical_categories:
        key_insights.append(f"🔴 Critical concerns in: {', '.join(critical_categories)}")
        recommendations.append(f"Prioritize immediate action for {', '.join(critical_categories)}")

    # Trend indicators (if we have timestamp data)
    trend_analysis = "Trend analysis requires historical data collection"

    return jsonify({
        'overall_assessment': {
            'score': round(overall_score, 1),
            'tier': overall_tier['tier'],
            'tier_color': overall_tier['color'],
            'tier_description': overall_tier['description'],
            'tier_icon': overall_tier['icon'],
            'score_range': f"{min(hseg_scores):.1f} - {max(hseg_scores):.1f}",
            'standard_deviation': round(np.std(hseg_scores), 2)
        },
        'tier_distribution': {
            'percentages': {tier: round((count / len(hseg_scores)) * 100, 1)
                          for tier, count in tier_distribution.items()},
            'counts': tier_distribution,
            'total_responses': len(hseg_scores)
        },
        'category_analysis': category_insights,
        'risk_assessment': {
            'at_risk_percentage': round(risk_percentage, 1),
            'at_risk_count': at_risk_count,
            'total_assessed': len(hseg_scores),
            'risk_level': 'High' if risk_percentage > 30 else 'Moderate' if risk_percentage > 15 else 'Low'
        },
        'key_insights': key_insights,
        'recommendations': recommendations,
        'trend_analysis': trend_analysis,
        'methodology': {
            'framework': 'HSEG Five-Tier Cultural Risk Assessment',
            'categories': len(HSEG_CATEGORIES),
            'total_questions': 22,
            'weighting_system': 'Risk-proportional category weights',
            'score_range': '7-28 points (normalized)',
            'documentation': 'HSEG_Comprehensive_Scoring_Documentation.md'
        }
    })

if __name__ == '__main__':
    app.run(debug=True, port=8999, host='0.0.0.0')
