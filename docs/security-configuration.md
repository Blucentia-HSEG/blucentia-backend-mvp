# Security Configuration Guide

## 🔒 Content Security Policy (CSP) Implementation

### Overview
The HSEG Dashboard implements a comprehensive Content Security Policy to protect against cross-site scripting (XSS) attacks while maintaining full functionality of visualization libraries.

### CSP Configuration Details

#### Development vs Production
The CSP configuration automatically adapts based on the environment:

```python
# Environment detection
is_development = os.environ.get('FLASK_ENV') == 'development'
```

#### Current CSP Settings

```
default-src 'self';
script-src 'self' 'unsafe-inline' 'unsafe-eval'
    https://cdn.jsdelivr.net
    https://cdnjs.cloudflare.com
    https://cdn.plot.ly
    https://unpkg.com
    https://ajax.googleapis.com;
style-src 'self' 'unsafe-inline'
    https://cdn.jsdelivr.net
    https://cdnjs.cloudflare.com
    https://fonts.googleapis.com;
font-src 'self'
    https://cdn.jsdelivr.net
    https://cdnjs.cloudflare.com
    https://fonts.gstatic.com;
img-src 'self' data: blob:;
connect-src 'self';
worker-src 'self' blob:;
child-src 'self' blob:;
object-src 'none';
base-uri 'self';
```

### Why `unsafe-eval` is Required

#### Chart.js and Plotly.js Requirements
Modern data visualization libraries require dynamic code evaluation for:

1. **Dynamic Chart Generation**: Creating charts based on runtime data structures
2. **Mathematical Calculations**: Complex statistical computations for PCA, clustering, etc.
3. **Performance Optimization**: Just-in-time compilation of rendering functions
4. **Plugin System**: Dynamic loading and execution of chart plugins

#### Security Mitigation Strategies

Despite using `unsafe-eval`, the dashboard maintains security through:

1. **Input Validation**: All data is sanitized before chart generation
2. **Trusted Sources**: External scripts only from verified CDNs
3. **No User-Generated Content**: No dynamic script generation from user input
4. **Additional Headers**: Comprehensive security header implementation

### Alternative Security Approaches

#### Option 1: Self-Hosting Libraries (Recommended for High Security)
```bash
# Download and host libraries locally to avoid external dependencies
wget https://cdn.plot.ly/plotly-2.27.0.min.js -P static/
wget https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.js -P static/
```

Then update CSP:
```python
csp = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-eval'; "  # Still needed for chart libraries
    "style-src 'self' 'unsafe-inline'; "
    # Remove external CDN sources
)
```

#### Option 2: Strict CSP with Nonces (Advanced)
```python
# Generate nonce for each request
import secrets

@app.before_request
def generate_csp_nonce():
    g.nonce = secrets.token_urlsafe(16)

# Use nonce in CSP
csp = f"script-src 'self' 'nonce-{g.nonce}';"
```

#### Option 3: Web Workers (Future Enhancement)
Move heavy chart computations to Web Workers to isolate eval usage:
```javascript
// In a separate worker file
self.addEventListener('message', function(e) {
    // Perform chart calculations in isolated context
    // Post results back to main thread
});
```

### Additional Security Headers

The dashboard implements multiple security layers:

```python
# Content type protection
response.headers['X-Content-Type-Options'] = 'nosniff'

# Frame protection
response.headers['X-Frame-Options'] = 'DENY'

# XSS protection
response.headers['X-XSS-Protection'] = '1; mode=block'

# Referrer policy
response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
```

### Monitoring and Alerts

#### CSP Violation Reporting
To monitor CSP violations in production, add a report-uri:

```python
csp += " report-uri /csp-violation-report;"

@app.route('/csp-violation-report', methods=['POST'])
def csp_violation_report():
    report = request.get_json()
    # Log violation for security analysis
    app.logger.warning(f"CSP Violation: {report}")
    return '', 204
```

#### Real-time Monitoring
Integrate with security monitoring services:
```python
import sentry_sdk

sentry_sdk.init(
    dsn="YOUR_SENTRY_DSN",
    integrations=[FlaskIntegration()],
    traces_sample_rate=1.0
)
```

### Best Practices for Production

#### 1. Regular Security Audits
- Review CSP configuration quarterly
- Audit external dependencies for security updates
- Monitor CSP violation reports

#### 2. Dependency Management
- Pin exact versions of external libraries
- Regular security updates for Chart.js and Plotly.js
- Use Subresource Integrity (SRI) hashes

#### 3. Input Sanitization
```python
from markupsafe import escape

def sanitize_chart_data(data):
    """Sanitize data before passing to charts"""
    if isinstance(data, str):
        return escape(data)
    elif isinstance(data, dict):
        return {k: sanitize_chart_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_chart_data(item) for item in data]
    return data
```

### Emergency CSP Disable

For critical production issues, temporarily disable CSP:

```python
# Emergency CSP bypass (use with extreme caution)
if os.environ.get('EMERGENCY_DISABLE_CSP') == 'true':
    # Skip CSP header - log this action
    app.logger.critical("CSP disabled by emergency flag")
    return response
```

### CSP Testing and Validation

#### Browser Testing
Test CSP compliance across browsers:
- Chrome DevTools → Security tab
- Firefox Developer Tools → Console
- Safari Web Inspector → Console

#### Automated Testing
```python
def test_csp_headers():
    response = client.get('/')
    assert 'Content-Security-Policy' in response.headers
    csp = response.headers['Content-Security-Policy']
    assert "'self'" in csp
    assert "unsafe-eval" in csp  # Required for charts
```

### Migration Path to Stricter CSP

#### Phase 1: Current Implementation (Done ✅)
- Implement permissive CSP with `unsafe-eval`
- Monitor violations and functionality

#### Phase 2: Library Analysis (Recommended)
- Investigate Chart.js alternatives that don't require `eval()`
- Consider server-side chart generation for sensitive data
- Evaluate WebAssembly-based visualization libraries

#### Phase 3: Strict CSP (Future)
- Remove `unsafe-eval` once compatible libraries are found
- Implement nonce-based script loading
- Move to self-hosted libraries only

### Compliance and Audit

The current CSP configuration balances:
- ✅ **Functionality**: All dashboard features work correctly
- ✅ **Security**: Protection against most XSS vectors
- ⚠️ **Compliance**: May require review for strict regulatory environments
- ✅ **Performance**: Minimal impact on load times

For environments requiring stricter compliance, consider the migration path outlined above or implement server-side chart generation.

---

*For implementation details, see [`technical-guide.md`](technical-guide.md)*
*For business context, see [`business-insights.md`](business-insights.md)*