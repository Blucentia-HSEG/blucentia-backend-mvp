from flask import Flask, jsonify, request, send_from_directory, Response, stream_template
from flask_cors import CORS
import json
import os
from data_visualization import main as generate_visualizations
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

def load_data():
    """Load processed data and raw data"""
    global processed_data, raw_data

    # Load raw data for streaming
    if os.path.exists('hseg_final_dataset.json'):
        print("Loading raw dataset...")
        with open('hseg_final_dataset.json', 'r') as f:
            raw_data = json.load(f)
        print(f"Loaded {len(raw_data)} raw records")

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

        # Calculate culture score
        scores = [record.get(f'q{i}', 0) for i in range(1, 23) if record.get(f'q{i}') is not None]
        if scores:
            culture_score = sum(scores) / len(scores)
            domain_data[domain]['culture_scores'].append(culture_score)

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

    # Define survey sections as per hseg_comprehensive_analysis.py
    sections = {
        'Power Abuse & Suppression': ['q1', 'q2', 'q3', 'q4'],
        'Discrimination & Exclusion': ['q5', 'q6', 'q7'],
        'Manipulative Work Culture': ['q8', 'q9', 'q10'],
        'Failure of Accountability': ['q11', 'q12', 'q13', 'q14'],
        'Mental Health Harm': ['q15', 'q16', 'q17', 'q18'],
        'Erosion of Voice & Autonomy': ['q19', 'q20', 'q21', 'q22']
    }

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
            section_data[section_name] = {
                'overall_score': round(sum(section_scores) / len(section_scores), 3),
                'overall_std': round(np.std(section_scores), 3),
                'count': len(section_scores),
                'domain_breakdown': {
                    domain: {
                        'avg_score': round(sum(scores) / len(scores), 3),
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

    edges = np.linspace(1, 4, bins + 1).tolist()
    dists = {}
    for name, scores in section_scores.items():
        counts, _ = np.histogram(scores, bins=bins, range=(1, 4))
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
    """Get organization analysis data"""
    limit = int(request.args.get('limit', 20))

    if not raw_data:
        return jsonify([])

    # Generate organization data from raw_data
    org_data = {}

    # Group by organization and calculate metrics
    for record in raw_data:
        org_name = record.get('organization_name', 'Unknown')
        if org_name not in org_data:
            org_data[org_name] = {
                'name': org_name,
                'domain': record.get('domain', 'Unknown'),
                'employee_count': record.get('employee_count', 0),
                'responses': [],
                'culture_scores': []
            }

        # Calculate culture score for this response
        scores = [record.get(f'q{i}', 0) for i in range(1, 23) if record.get(f'q{i}') is not None]
        if scores:
            culture_score = sum(scores) / len(scores)
            org_data[org_name]['culture_scores'].append(culture_score)
            org_data[org_name]['responses'].append(record)

    # Calculate final metrics
    result = []
    for org_name, data in org_data.items():
        if len(data['culture_scores']) >= 5:  # Only include orgs with enough responses
            avg_score = sum(data['culture_scores']) / len(data['culture_scores'])
            result.append({
                'name': org_name,
                'domain': data['domain'],
                'employee_count': data['employee_count'],
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

    # Define demographic mappings based on hseg_comprehensive_analysis.py
    demographic_mappings = {
        'age_range': 'q26',
        'gender_identity': 'q27',
        'race_ethnicity': 'q28',
        'education_level': 'q29',
        'tenure': 'q30',
        'position_level': 'q31',
        'domain_role': 'q32',
        'supervises_others': 'q33'
    }

    # Also check for direct demographic columns
    for demo_name in ['age_range', 'gender_identity', 'tenure_range', 'position_level']:
        if demo_name in raw_data[0] if raw_data else {}:
            demographic_mappings[demo_name] = demo_name

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

            # Calculate culture score
            scores = [record.get(f'q{i}', 0) for i in range(1, 23) if record.get(f'q{i}') is not None]
            if scores:
                culture_score = sum(scores) / len(scores)
                demographic_data[demo_value]['culture_scores'].append(culture_score)

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
            'culture_score': round(sum([record.get(f'q{i}', 0) for i in range(1, 23)]) / 22, 2)
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

    # Calculate average culture score
    total_score = 0
    valid_responses = 0

    for record in raw_data:
        scores = [record.get(f'q{i}', 0) for i in range(1, 23) if record.get(f'q{i}') is not None]
        if scores:
            total_score += sum(scores) / len(scores)
            valid_responses += 1

    avg_culture_score = round(total_score / valid_responses, 2) if valid_responses > 0 else 0

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

    if not raw_data:
        return jsonify({"labels": [], "datasets": []})

    # Group data by submission date
    from collections import defaultdict
    import datetime

    date_scores = defaultdict(list)

    for record in raw_data:
        date_str = record.get('submission_date', '')
        if date_str:
            try:
                date_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d')
                if metric == 'response_count':
                    date_scores[date_str].append(1)
                else:
                    scores = [record.get(f'q{i}', 0) for i in range(1, 23) if record.get(f'q{i}') is not None]
                    if scores:
                        avg_score = sum(scores) / len(scores)
                        date_scores[date_str].append(avg_score)
            except:
                pass

    # Create trend data
    labels = sorted(date_scores.keys())[-days:]
    if metric == 'response_count':
        trend_data = [
            sum(date_scores[date]) if date_scores[date] else 0
            for date in labels
        ]
        dataset = {"label": "Responses", "data": trend_data, "borderColor": "#10b981", "backgroundColor": "rgba(16,185,129,0.1)", "tension": 0.3}
    else:
        trend_data = [
            round(sum(date_scores[date]) / len(date_scores[date]), 2) if date_scores[date] else 0
            for date in labels
        ]
        dataset = {"label": "Culture Score", "data": trend_data, "borderColor": "#2563eb", "backgroundColor": "rgba(37, 99, 235, 0.1)", "tension": 0.4}

    return jsonify({
        "labels": labels,
        "datasets": [dataset]
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

        # Calculate culture score
        scores = [record.get(f'q{i}', 0) for i in range(1, 23) if record.get(f'q{i}') is not None]
        if scores:
            culture_score = sum(scores) / len(scores)
            hierarchy[domain][org][dept]['culture_scores'].append(culture_score)

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
    """Ridge-like distribution data: for a given domain, per-section normalized histograms."""
    if not raw_data:
        return jsonify({'bins': [], 'sections': {}})

    domain = request.args.get('domain', 'all')
    bins = int(request.args.get('bins', 30))

    sections = {
        'Power Abuse & Suppression': ['q1', 'q2', 'q3', 'q4'],
        'Discrimination & Exclusion': ['q5', 'q6', 'q7'],
        'Manipulative Work Culture': ['q8', 'q9', 'q10'],
        'Failure of Accountability': ['q11', 'q12', 'q13', 'q14'],
        'Mental Health Harm': ['q15', 'q16', 'q17', 'q18'],
        'Erosion of Voice & Autonomy': ['q19', 'q20', 'q21', 'q22']
    }

    # collect section scores for the selected domain or all
    section_scores = {name: [] for name in sections.keys()}
    for r in raw_data:
        if domain.lower() != 'all' and (r.get('domain') or '').lower() != domain.lower():
            continue
        for name, qs in sections.items():
            vals = [r.get(q) for q in qs if r.get(q) is not None]
            if vals:
                section_scores[name].append(sum(vals)/len(vals))

    edges = np.linspace(1, 4, bins + 1)
    centers = ((edges[:-1] + edges[1:]) / 2.0).tolist()
    result = {}
    for name, arr in section_scores.items():
        if not arr:
            continue
        counts, _ = np.histogram(arr, bins=bins, range=(1,4), density=True)
        result[name] = {
            'x': centers,
            'y': counts.tolist()
        }

    return jsonify({'sections': result, 'x': centers})

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
        scores = [record.get(f'q{i}') for i in range(1,23) if record.get(f'q{i}') is not None]
        if not scores:
            continue
        key = (dom, ten)
        by_key[key].append(sum(scores)/len(scores))

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

        overall_score = sum(avg_scores.values()) / len(avg_scores) if avg_scores else 0

        nodes.append({
            'id': org,
            'domain': data['domain'],
            'employee_count': data['employee_count'],
            'responses': data['responses'],
            'culture_score': round(overall_score, 2),
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
            labels.append({
                'domain': record.get('domain', 'Unknown'),
                'organization': record.get('organization_name', 'Unknown'),
                'department': record.get('department', 'Unknown')
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
                'department': labels[i]['department']
            }
            for i in range(len(pca_result))
        ],
        'explained_variance': [float(val) for val in eigenvalues[:5]],
        'loadings': {
            'pc1': [float(eigenvectors[i, 0]) for i in range(len(questions))],
            'pc2': [float(eigenvectors[i, 1]) for i in range(len(questions))],
            'questions': questions
        },
        'elbow': inertias
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

    edges = np.linspace(1,4,bins+1).tolist()
    counts, _ = np.histogram(scores, bins=bins, range=(1,4))
    per_domain = {}
    for d, arr in by_domain.items():
        c, _ = np.histogram(arr, bins=bins, range=(1,4))
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

    # Define survey sections as per hseg_comprehensive_analysis.py
    sections = {
        'Power Abuse & Suppression': ['q1', 'q2', 'q3', 'q4'],
        'Discrimination & Exclusion': ['q5', 'q6', 'q7'],
        'Manipulative Work Culture': ['q8', 'q9', 'q10'],
        'Failure of Accountability': ['q11', 'q12', 'q13', 'q14'],
        'Mental Health Harm': ['q15', 'q16', 'q17', 'q18'],
        'Erosion of Voice & Autonomy': ['q19', 'q20', 'q21', 'q22']
    }

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
            section_results[section_name.replace(' & ', ' ')] = {
                'score': round(sum(section_scores) / len(section_scores), 3),
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

if __name__ == '__main__':
    app.run(debug=True, port=8999, host='0.0.0.0')
