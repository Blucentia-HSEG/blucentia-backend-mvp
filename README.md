# HSEG Workplace Culture Analytics Dashboard

A comprehensive dashboard for analyzing workplace culture intelligence survey data with 49,550+ responses across Healthcare, University, and Business domains.

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- CSV file named `hseg_final_dataset.csv`

### 1. Setup Environment
```bash
# Clone or create project directory
mkdir hseg-dashboard
cd hseg-dashboard

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Prepare Your Data
Place your `hseg_final_dataset.csv` file in the root directory.

**Required CSV columns:**
- `q1` through `q22`: Survey questions (1-4 scale)
- `domain`: Healthcare/University/Business
- `organization_name`: Organization identifier
- `department`: Department/division
- `q26` through `q33`: Demographics

### 3. Process Data
```bash
# Process CSV data into JSON format
python data_processor.py
```
This creates `processed_hseg_data.json` with all calculated metrics.

### 4. Start Dashboard
```bash
# Start Flask server
python app.py
```

Visit `http://localhost:5000` to view your dashboard.

## 📊 Dashboard Features

### Data Processing
- **Section Scores**: Calculates 6 cultural dimension scores
- **Domain Analysis**: Healthcare vs University vs Business comparisons  
- **Organization Benchmarking**: Performance rankings with statistical significance
- **Demographic Breakdowns**: Analysis across age, gender, position, etc.
- **Correlation Analysis**: Inter-section relationship mapping

### Interactive Visualizations
- **Executive Overview**: KPI cards and domain distributions
- **Section Analysis**: Comparative bar charts and detailed breakdowns
- **Organization Rankings**: Top/bottom performer identification
- **Demographic Insights**: Culture patterns across demographic groups
- **Correlation Matrix**: Statistical relationships between dimensions

### Real-time Features
- **Domain Filtering**: Healthcare/University/Business/All
- **Data Refresh**: Reload from CSV without restart
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Export Ready**: Built-in report generation

## 📁 Project Structure

```
hseg-dashboard/
├── hseg_final_dataset.csv     # Your survey data (place here)
├── requirements.txt           # Python dependencies
├── data_processor.py         # Data processing logic
├── app.py                    # Flask web server
├── index.html               # Dashboard interface
├── dashboard.js             # Frontend logic
├── processed_hseg_data.json # Generated processed data
└── README.md               # This file
```

## 🔧 Configuration

### CSV Data Format
Your CSV should include these columns:

**Survey Questions (1-4 scale):**
- `q1-q4`: Power Abuse & Suppression
- `q5-q7`: Discrimination & Exclusion  
- `q8-q10`: Manipulative Work Culture
- `q11-q14`: Failure of Accountability
- `q15-q18`: Mental Health Harm
- `q19-q22`: Erosion of Voice & Autonomy

**Organizational Data:**
- `domain`: Healthcare, University, or Business
- `organization_name`: Organization identifier
- `department`: Department/division name

**Demographics (Q26-Q33):**
- `q26`: Age Range
- `q27`: Gender Identity
- `q28`: Race/Ethnicity
- `q29`: Education Level
- `q30`: Tenure
- `q31`: Position Level
- `q32`: Domain Role
- `q33`: Supervises Others

### API Endpoints

- `GET /api/overview` - Overall statistics
- `GET /api/domains` - Domain analysis
- `GET /api/sections?domain=X` - Section analysis (filtered)
- `GET /api/organizations?limit=N` - Organization benchmarks
- `GET /api/demographics?type=X` - Demographic analysis
- `GET /api/correlations` - Correlation matrix
- `POST /api/refresh` - Refresh data from CSV

## 🔄 Data Updates

To update with new survey data:

1. Replace `hseg_final_dataset.csv` with new data
2. Run `python data_processor.py` to reprocess
3. Click "Refresh" in dashboard or restart server

## 📈 Analytics Included

### Statistical Analysis
- **Correlation Analysis**: Pearson correlations between sections
- **Domain Comparisons**: ANOVA tests for significant differences
- **Demographic Analysis**: Group comparisons and effect sizes
- **Organization Benchmarking**: Rankings with confidence intervals

### Key Metrics
- **Section Scores**: Mean scores for 6 cultural dimensions
- **Domain Performance**: Cross-sector comparisons
- **Organization Rankings**: Best and worst performers
- **Demographic Patterns**: Culture variations across groups
- **Response Distribution**: Coverage analysis

## 🚀 Deployment Options

### Option 1: Local Development
```bash
python app.py
# Access at http://localhost:5000
```

### Option 2: Production Server
```bash
# Install production WSGI server
pip install gunicorn

# Run with gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Option 3: Cloud Deployment (Heroku/Railway/Render)
1. Create `Procfile`: `web: gunicorn app:app`
2. Push to Git repository
3. Deploy to cloud platform
4. Upload processed data file

## 🔍 Troubleshooting

### Common Issues

**"No data available" error:**
- Ensure `hseg_final_dataset.csv` is in root directory
- Check CSV column names match expected format
- Run `python data_processor.py` manually to see errors

**Charts not loading:**
- Check browser console for JavaScript errors
- Ensure Flask server is running
- Verify API endpoints return data

**Performance issues:**
- Large datasets (>100k rows) may need optimization
- Consider data sampling for very large files
- Increase server timeout settings

### Debug Commands
```bash
# Test data processing
python data_processor.py

# Check processed data
python -c "import json; print(json.load(open('processed_hseg_data.json'))['overview'])"

# Test API endpoints
curl http://localhost:5000/api/overview
```

## 📋 Data Privacy

- Remove personally identifiable information from CSV
- Use organization codes instead of names if needed
- Consider data aggregation for sensitive metrics
- Implement access controls for production deployment

## 🤝 Support

For issues:
1. Check CSV data format and column names
2. Review Flask server console logs
3. Test API endpoints individually
4. Verify processed data file is created correctly

## 📄 License

This project is configured for your specific HSEG analysis needs.

---

**Last Updated**: 2025
**Version**: 1.0.0
