# HSEG Analytics Dashboard

A comprehensive Flask-based web application for visualizing HSEG (Healthy System Evaluation Guide) workplace culture survey data with advanced analytics and interactive visualizations.

## 🚀 Key Features

- **Advanced Analytics**: PCA analysis, clustering, hierarchical visualization, and statistical analysis
- **Interactive Visualizations**: 20+ chart types using Chart.js and Plotly.js with zoom and drill-down capabilities
- **Real-time Filtering**: Dynamic filtering across demographics, organizations, and time ranges
- **Comprehensive Documentation**: Full user guides, technical documentation, and business insights
- **Security Hardened**: Content Security Policy implementation with OWASP compliance
- **Large Dataset Optimized**: Efficiently handles 49,550+ survey responses with chunked data loading
- **Mobile Responsive**: Fully responsive design for all devices

## 📁 Project Structure

```
├── app.py                  # Main Flask application with security headers and API endpoints
├── index.html              # Comprehensive dashboard HTML template
├── static/                 # Frontend assets
│   ├── dashboard.js        # Advanced dashboard JavaScript with PCA, clustering, hierarchical analysis
│   └── style.css          # Responsive application styling
├── data/                   # Chunked dataset (GitHub-compatible)
│   ├── hseg_data_part_01.json  # Data chunk 1 (17.7MB)
│   ├── hseg_data_part_02.json  # Data chunk 2 (17.7MB)
│   ├── hseg_data_part_03.json  # Data chunk 3 (17.7MB)
│   ├── hseg_data_part_04.json  # Data chunk 4 (17.7MB)
│   ├── hseg_data_part_05.json  # Data chunk 5 (2.8MB)
│   └── metadata.json          # Chunk information and integrity data
├── docs/                   # Comprehensive documentation
│   ├── README.md           # Documentation overview
│   ├── user-guide.md       # Complete user instructions
│   ├── technical-guide.md   # Implementation details
│   ├── business-insights.md # Business interpretation guide
│   ├── visualizations.md   # Chart explanations
│   └── security-configuration.md # Security implementation guide
├── utils/                  # Data processing utilities
│   ├── merge_json.py       # Merge data chunks
│   ├── split_json.py       # Split large JSON files
│   └── data_visualization.py # Data processing helpers
├── requirements.txt        # Python dependencies
├── package.json           # Project metadata and npm scripts
└── processed_hseg_data.json # Sample/processed data file
```

## 🛠️ Setup & Installation

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Quick Start

#### GitHub-Ready Workflow (Recommended)
The application now works directly with chunked JSON files, making it GitHub-compatible without any manual merging:

```bash
git clone https://github.com/Blucentia-HSEG/blucentia-backend-mvp.git
cd blucentia-backend-mvp
npm run setup          # Install dependencies
npm run start          # Start application (auto-loads from chunks)
```

#### Alternative Methods

**Option 1: Direct Python execution**
```bash
git clone https://github.com/Blucentia-HSEG/blucentia-backend-mvp.git
cd blucentia-backend-mvp
pip install -r requirements.txt
python app.py          # Automatically loads from data/*.json chunks
```

**Option 2: With manual merge**
```bash
git clone https://github.com/Blucentia-HSEG/blucentia-backend-mvp.git
cd blucentia-backend-mvp
npm run setup
npm run build          # Merge data chunks (optional)
npm run start
```

**Access the dashboard:** Open your browser and navigate to `http://localhost:8999`

## 📊 Data Management

### GitHub-Compatible Data Management

The application uses an intelligent data loading system that works seamlessly with GitHub's file size restrictions:

#### Automatic Chunk Loading
- **Chunked data**: 5 files in `data/` folder (~17MB each, GitHub-compatible)
- **Auto-detection**: Application automatically loads from chunks when available
- **No manual merging required**: Just clone and run!

#### Data Structure
```
data/
├── hseg_data_part_01.json  # 16.9MB - Records 1-11,918
├── hseg_data_part_02.json  # 16.9MB - Records 11,919-23,836
├── hseg_data_part_03.json  # 16.9MB - Records 23,837-35,754
├── hseg_data_part_04.json  # 16.9MB - Records 35,755-47,672
├── hseg_data_part_05.json  # 2.7MB  - Records 47,673-49,550
└── metadata.json           # Chunk information
```

#### Development Utilities

**Split large files** (for developers with new datasets):
```bash
python utils/split_json.py    # Creates GitHub-compatible chunks
```

**Manual merge** (optional, for development):
```bash
python utils/merge_json.py    # Reconstructs full dataset
```

#### Production Deployment Workflow

```bash
# 1. Clone from GitHub
git clone https://github.com/Blucentia-HSEG/blucentia-backend-mvp.git
cd blucentia-backend-mvp

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start application (loads chunks automatically)
python app.py
```

**That's it!** No manual data merging required.

## 🚀 Deployment

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Merge data chunks
python utils/merge_json.py

# Run development server
python app.py
```

### Production Deployment

#### Option 1: Standard Deployment
```bash
# Clone repository
git clone https://github.com/Blucentia-HSEG/blucentia-backend-mvp.git
cd blucentia-backend-mvp

# Install dependencies
pip install -r requirements.txt

# Merge data chunks for production
python utils/merge_json.py

# Run with production server (example with gunicorn)
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

#### Option 2: Docker Deployment
```dockerfile
# Example Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt
RUN python utils/merge_json.py

EXPOSE 5000
CMD ["python", "app.py"]
```

### Environment Variables

The application uses the following optional environment variables:

- `FLASK_ENV`: Set to `production` for production deployment
- `PORT`: Port number (default: 5000)
- `DEBUG`: Set to `False` for production

## 📈 Dashboard Features

### Dashboard Sections

1. **Core Analytics**
   - Response trends and time-series analysis
   - Section analysis with radar charts
   - Score distribution analysis with multiple bin options

2. **Organization Insights**
   - Organizational benchmarking and performance comparisons
   - Department-level performance analysis
   - Top organizations leaderboard with filtering
   - Performance vs. size analysis with domain filtering

3. **Demographics Analysis**
   - Multi-dimensional demographic breakdown
   - Experience heatmaps across tenure and positions
   - Equity analysis across employee groups

4. **Advanced Analytics**
   - **PCA Analysis**: Principal component analysis with configurable parameters
   - **Clustering**: K-means, hierarchical, and DBSCAN clustering algorithms
   - **Hierarchical Analysis**: Interactive treemap visualization for organizational structure
   - **Section Distribution**: Ridge plots for cross-domain comparison

### Visualization Technologies

- **Chart.js 4.4.0**: Standard interactive charts with zoom functionality
- **Plotly.js 2.27.0**: Advanced statistical visualizations (PCA, clustering, treemaps)
- **Bootstrap 5**: Responsive UI framework
- **Custom JavaScript**: Advanced filtering, data processing, and chart interactions

### Security Features

- **Content Security Policy (CSP)**: Comprehensive CSP implementation with environment-aware configuration
- **Security Headers**: X-Frame-Options, X-Content-Type-Options, X-XSS-Protection
- **Input Sanitization**: Protected against XSS with data validation
- **CSRF Protection**: Built-in CSRF protection for form submissions

### Documentation Suite

The `/docs/` folder contains comprehensive documentation:
- **[User Guide](docs/user-guide.md)**: Complete instructions for using all dashboard features
- **[Technical Guide](docs/technical-guide.md)**: Implementation details and architecture
- **[Business Insights](docs/business-insights.md)**: How to interpret data for business decisions
- **[Visualizations](docs/visualizations.md)**: Detailed explanation of each chart type
- **[Security Configuration](docs/security-configuration.md)**: Security implementation and best practices

## 🔧 Development

### File Structure for Development

- **Main Application**: `app.py`
- **Frontend**: `static/dashboard.js`, `static/style.css`
- **Template**: `index.html`
- **Utilities**: `utils/split_json.py`, `utils/merge_json.py`, `utils/data_visualization.py`

### Adding New Visualizations

1. Add chart configuration in `static/dashboard.js`
2. Update the HTML template in `index.html`
3. Add corresponding route in `app.py` if needed

### Code Architecture

- **Modular Design**: Separate API endpoints, data processing, and frontend logic
- **Error Handling**: Comprehensive error handling with user-friendly messages
- **Performance Optimized**: Efficient data loading and caching mechanisms
- **Extensible**: Easy to add new visualizations and analysis methods

## 🐛 Troubleshooting

### Common Issues

1. **Large file not found error**:
   ```bash
   # Run the merge script to reconstruct the dataset
   python utils/merge_json.py
   ```

2. **Memory issues with large dataset**:
   - The application is optimized for large datasets
   - Consider increasing system memory for very large files
   - Use data pagination if needed

3. **Port already in use**:
   ```bash
   # Change port in app.py or use environment variable
   export PORT=8080
   python app.py
   ```

### Data Integrity Verification

After merging chunks, verify data integrity:
```bash
# Check if merge was successful
python -c "import json; data=json.load(open('hseg_final_dataset.json')); print(f'Total items: {len(data)}')"
# Should output: Total items: 49550
```

## 📝 License

[Add your license information here]

## 🤝 Contributing

[Add contributing guidelines here]

## 📞 Support

[Add contact information or support details here]

---

## ⚡ Quick Commands Reference

### Using npm scripts (Recommended):
```bash
# GitHub-ready workflow (no merging needed)
npm run setup          # Install Python dependencies
npm run start          # Start application (auto-loads chunks)

# Development utilities
npm run dev            # Run development server
npm run build          # Merge data chunks (optional)
npm run split-data     # Split large JSON file (for new datasets)
npm run clean          # Remove merged dataset file
```

### Using Python directly:
```bash
# GitHub-ready workflow
pip install -r requirements.txt
python app.py          # Automatically loads from data/*.json chunks

# Development utilities
python utils/split_json.py    # Split large JSON file (for new datasets)
python utils/merge_json.py    # Merge JSON chunks (optional)

# Data verification
ls -lh data/                  # Check chunk sizes (GitHub-compatible)
ls -lh hseg_final_dataset.json # Check merged file (if created)
```

**Note**: The application now works directly with chunked data files - no manual merging required after cloning from GitHub!