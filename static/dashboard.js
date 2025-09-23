// Enterprise Dashboard JavaScript with Performance Optimizations
class DashboardApp {
    constructor() {
        this.currentSection = 'dashboard';
        this.charts = {};
        this.data = {};
        this.cache = new Map();
        this.eventSource = null;
        this.loadingTimeout = null;
        this.virtualTable = null;
        this.debounceTimeout = null;

        this.init();
    }

    init() {
        try {
            this.showLoadingOverlay();
            this.setupEventListeners();
            this.initializeApp();
        } catch (e) {
            console.error('Init failed:', e);
            this.showError('Initialization error. Some features may be unavailable.');
            this.hideLoadingOverlay();
        }

        // Global safety net so overlay never blocks the UI
        window.addEventListener('error', () => this.hideLoadingOverlay());
        window.addEventListener('unhandledrejection', () => this.hideLoadingOverlay());
        // Fallback timeout
        setTimeout(() => this.hideLoadingOverlay(), 5000);
    }

    setupEventListeners() {
        // Sidebar navigation
        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const section = link.dataset.section;
                this.navigateToSection(section);
            });
        });

        // Sidebar toggle
        const sidebarToggle = document.getElementById('sidebarToggle');
        const mobileSidebarToggle = document.getElementById('mobileSidebarToggle');

        if (sidebarToggle) {
            sidebarToggle.addEventListener('click', () => {
                document.getElementById('sidebar').classList.toggle('collapsed');
            });
        }

        if (mobileSidebarToggle) {
            mobileSidebarToggle.addEventListener('click', () => {
                document.getElementById('sidebar').classList.toggle('show');
            });
        }

        // Refresh button
        const refreshBtn = document.getElementById('refreshData');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => this.refreshData());
        }

        // Export button
        const exportBtn = document.getElementById('exportData');
        if (exportBtn) {
            exportBtn.addEventListener('click', () => this.exportData());
        }

        // Search and filters
        const searchInput = document.getElementById('dataSearch');
        if (searchInput) {
            searchInput.addEventListener('input', (e) => {
                this.debounceSearch(e.target.value);
            });
        }

        const tableFilter = document.getElementById('tableFilter');
        if (tableFilter) {
            tableFilter.addEventListener('change', (e) => {
                this.filterTable(e.target.value);
            });
        }

        // Trend time range
        const trendTimeRange = document.getElementById('trendTimeRange');
        const trendMetric = document.getElementById('trendMetric');
        if (trendTimeRange) trendTimeRange.addEventListener('change', (e) => { this.updateTrendChart(e.target.value); });
        if (trendMetric) trendMetric.addEventListener('change', () => { const val = (trendTimeRange||{}).value || '30d'; this.updateTrendChart(val); });

        // Analytics trends controls (separate from dashboard)
        const aRange = document.getElementById('analyticsTrendTimeRange');
        const aMetric = document.getElementById('analyticsTrendMetric');
        const aGranularity = document.getElementById('analyticsTrendGranularity');
        const aSmoothing = document.getElementById('analyticsTrendSmoothing');

        if (aRange) aRange.addEventListener('change', (e)=> this.updateAnalyticsTrendChart(e.target.value));
        if (aMetric) aMetric.addEventListener('change', ()=> { const val = (aRange||{}).value || '30d'; this.updateAnalyticsTrendChart(val); });
        if (aGranularity) aGranularity.addEventListener('change', ()=> { const val = (aRange||{}).value || '30d'; this.updateAnalyticsTrendChart(val); });
        if (aSmoothing) aSmoothing.addEventListener('change', ()=> { const val = (aRange||{}).value || '30d'; this.updateAnalyticsTrendChart(val); });

        // Section tab functionality
        this.setupSectionTabs();

        // Filter event listeners
        this.setupFilterEventListeners();
    }

    async initializeApp() {
        try {
            // Load initial data and setup dashboard
            await this.loadQuickStats();
            await this.loadOrganizationsList();
            await this.setupDashboard();

            // Debug: Test if Chart.js is available
            if (typeof Chart === 'undefined') {
                console.error('Chart.js is not loaded!');
                this.showError('Chart.js library failed to load. Please refresh the page or check your internet connection.');
                this.hideLoadingOverlay();
                return;
            } else {
                console.log('✓ Chart.js loaded successfully:', Chart.version);
                // Set global chart defaults to avoid looping animations across tabs/resizes
                this.setupChartDefaults();
            }

            this.hideLoadingOverlay();
        } catch (error) {
            console.error('Failed to initialize app:', error);
            this.showError('Failed to load dashboard. Please refresh the page.');
            this.hideLoadingOverlay();
        }
    }

    setupChartDefaults() {
        try {
            // Disable all animations globally
            Chart.defaults.animation = false;
            Chart.defaults.animations = {};
            Chart.defaults.animation = { duration: 0 };
            Chart.defaults.transitions = Chart.defaults.transitions || {};
            Chart.defaults.transitions.active = { animation: { duration: 0 } };
            Chart.defaults.transitions.show = { animation: { duration: 0 } };
            Chart.defaults.transitions.hide = { animation: { duration: 0 } };
            Chart.defaults.responsiveAnimationDuration = 0;
            Chart.defaults.plugins = Chart.defaults.plugins || {};
            Chart.defaults.plugins.tooltip = Chart.defaults.plugins.tooltip || {};
            Chart.defaults.plugins.tooltip.animation = { duration: 0 };
            Chart.defaults.responsive = true;
        } catch (e) {
            console.warn('Unable to set Chart.js global defaults:', e);
        }
    }

    showLoadingOverlay(message = 'Loading Analytics Dashboard...') {
        const overlay = document.getElementById('loadingOverlay');
        if (overlay) {
            const loadingText = overlay.querySelector('.loading-text');
            if (loadingText) loadingText.textContent = message;
            overlay.classList.remove('hidden');
        }
    }

    hideLoadingOverlay() {
        const overlay = document.getElementById('loadingOverlay');
        if (overlay) {
            overlay.classList.add('hidden');
            overlay.style.display = 'none';
        }
    }

    updateLoadingProgress(progress) {
        const progressBar = document.getElementById('loadingProgressBar');
        if (progressBar) {
            progressBar.style.width = `${progress}%`;
        }
    }

    async fetchWithCache(url, options = {}) {
        const cacheKey = url + JSON.stringify(options);

        if (this.cache.has(cacheKey)) {
            const cached = this.cache.get(cacheKey);
            if (Date.now() - cached.timestamp < 60000) { // 1 minute cache
                return cached.data;
            }
        }

        const response = await fetch(url, options);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        this.cache.set(cacheKey, { data, timestamp: Date.now() });
        return data;
    }

    clearCacheForPattern(pattern) {
        const keysToDelete = [];
        for (const key of this.cache.keys()) {
            if (key.includes(pattern)) {
                keysToDelete.push(key);
            }
        }
        keysToDelete.forEach(key => this.cache.delete(key));
    }

    async loadQuickStats() {
        try {
            const stats = await this.fetchWithCache('/api/stats/quick');
            this.updateKPICards(stats);
            this.updateSystemStatus(stats);
        } catch (error) {
            console.error('Failed to load quick stats:', error);
        }
    }

    updateKPICards(stats) {
        const kpiContainer = document.getElementById('kpiCards');
        if (!kpiContainer) return;

        const kpiData = [
            {
                title: 'Total Responses',
                value: this.formatNumber(stats.total_responses),
                icon: 'fas fa-users',
                change: '+5.2%',
                changeType: 'positive'
            },
            {
                title: 'Organizations',
                value: this.formatNumber(stats.num_organizations),
                icon: 'fas fa-building',
                change: '+2.1%',
                changeType: 'positive'
            },
            {
                title: 'Domains',
                value: this.formatNumber(stats.num_domains),
                icon: 'fas fa-chart-pie',
                change: '0%',
                changeType: 'neutral'
            },
            {
                title: 'Culture Score',
                value: stats.overall_culture_score || '0.00',
                icon: 'fas fa-star',
                change: '+1.8%',
                changeType: 'positive'
            }
        ];

        kpiContainer.innerHTML = kpiData.map(kpi => `
            <div class="col-xl-3 col-lg-6">
                <div class="kpi-card">
                    <div class="kpi-header">
                        <h6 class="kpi-title">${kpi.title}</h6>
                        <div class="kpi-icon">
                            <i class="${kpi.icon}"></i>
                        </div>
                    </div>
                    <div class="kpi-value">${kpi.value}</div>
                    <div class="kpi-change ${kpi.changeType}">
                        <i class="fas fa-arrow-${kpi.changeType === 'positive' ? 'up' : kpi.changeType === 'negative' ? 'down' : 'right'}"></i>
                        ${kpi.change}
                    </div>
                </div>
            </div>
        `).join('');
    }

    updateSystemStatus(stats) {
        const dataFreshness = document.getElementById('dataFreshness');
        const responseTime = document.getElementById('responseTime');

        if (dataFreshness) dataFreshness.textContent = stats.data_freshness || 'Live';
        if (responseTime) responseTime.textContent = `${stats.response_time_ms || '--'}ms`;
    }

    async loadOrganizationsList() {
        try {
            const organizations = await this.fetchWithCache('/api/organizations/list');
            const tableFilter = document.getElementById('tableFilter');

            if (tableFilter && organizations) {
                tableFilter.innerHTML = '<option value="all">All Organizations</option>' +
                    organizations.map(org => `<option value="${org.value}">${org.label}</option>`).join('');
            }
        } catch (error) {
            console.error('Failed to load organizations list:', error);
        }
    }

    async setupDashboard() {
        await this.setupTrendChart();
        await this.setupDomainChart();
        await this.setupVirtualTable();
    }

    async setupTrendChart() {
        try {
            const trendData = await this.fetchWithCache('/api/analytics/trend?days=30');
            const ctx = document.getElementById('trendChart');

            if (!ctx || !trendData.labels) return;

            if (this.charts.trendChart) {
                this.charts.trendChart.destroy();
            }

            this.charts.trendChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: trendData.labels,
                    datasets: trendData.datasets || []
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {
                        intersect: false,
                        mode: 'index'
                    },
                    plugins: {
                        legend: {
                            display: false
                        },
                        tooltip: {
                            backgroundColor: 'rgba(0, 0, 0, 0.8)',
                            titleColor: 'white',
                            bodyColor: 'white',
                            borderColor: '#2563eb',
                            borderWidth: 1
                        }
                    },
                    scales: {
                        x: {
                            display: true,
                            grid: {
                                display: false
                            }
                        },
                        y: {
                            display: true,
                            grid: {
                                color: 'rgba(0, 0, 0, 0.1)'
                            },
                            min: 1.5,
                            max: 3.0,
                            ticks: { stepSize: 0.25 }
                        }
                    },
                    elements: {
                        point: {
                            radius: 4,
                            hoverRadius: 6
                        }
                    }
                }
            });
        } catch (error) {
            console.error('Failed to setup trend chart:', error);
        }
    }

    async setupDomainChart() {
        try {
            const domainsData = await this.fetchWithCache('/api/domains');
            const ctx = document.getElementById('domainPieChart');

            if (!ctx || !domainsData) return;

            const domains = Object.keys(domainsData);
            const counts = domains.map(domain => domainsData[domain].count || 0);

            if (this.charts.domainChart) {
                this.charts.domainChart.destroy();
            }

            this.charts.domainChart = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: domains,
                    datasets: [{
                        data: counts,
                        backgroundColor: [
                            '#2563eb', '#06b6d4', '#10b981', '#f59e0b',
                            '#ef4444', '#8b5cf6', '#ec4899', '#6b7280'
                        ],
                        borderWidth: 0,
                        cutout: '60%'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: {
                                usePointStyle: true,
                                padding: 20
                            }
                        },
                        tooltip: {
                            backgroundColor: 'rgba(0, 0, 0, 0.8)',
                            titleColor: 'white',
                            bodyColor: 'white'
                        }
                    }
                }
            });
        } catch (error) {
            console.error('Failed to setup domain chart:', error);
        }
    }

    async setupVirtualTable() {
        const container = document.getElementById('dataTableContainer');
        if (!container) return;

        this.virtualTable = new VirtualTable(container, {
            onLoadData: (page, limit, search, filter) => this.loadTableData(page, limit, search, filter),
            columns: [
                { key: 'response_id', title: 'Response ID', width: '150px' },
                { key: 'organization_name', title: 'Organization', width: '200px' },
                { key: 'domain', title: 'Domain', width: '150px' },
                { key: 'department', title: 'Department', width: '150px' },
                { key: 'position_level', title: 'Level', width: '120px' },
                { key: 'submission_date', title: 'Date', width: '120px' },
                { key: 'culture_score', title: 'Score', width: '100px' }
            ]
        });

        await this.virtualTable.initialize();
    }

    async loadTableData(page = 1, limit = 50, search = '', organization = 'all') {
        try {
            const params = new URLSearchParams({
                page: page.toString(),
                limit: limit.toString(),
                search: search,
                organization: organization
            });

            const data = await this.fetchWithCache(`/api/data/paginated?${params}`);
            return data;
        } catch (error) {
            console.error('Failed to load table data:', error);
            return { data: [], total: 0, page: 1, pages: 1 };
        }
    }

    debounceSearch(searchTerm) {
        clearTimeout(this.debounceTimeout);
        this.debounceTimeout = setTimeout(() => {
            if (this.virtualTable) {
                this.virtualTable.search(searchTerm);
            }
        }, 300);
    }

    filterTable(organization) {
        if (this.virtualTable) {
            this.virtualTable.filter(organization);
        }
    }

    async updateTrendChart(timeRange) {
        const days = timeRange === '7d' ? 7 : timeRange === '30d' ? 30 : 90;
        const metric = (document.getElementById('trendMetric')||{}).value || 'culture_score';

        try {
            const trendData = await this.fetchWithCache(`/api/analytics/trend?days=${days}&metric=${metric}`);

            if (this.charts.trendChart && trendData.labels) {
                this.charts.trendChart.data.labels = trendData.labels;
                this.charts.trendChart.data.datasets = trendData.datasets || [];
                if (metric === 'culture_score' && this.charts.trendChart.options && this.charts.trendChart.options.scales && this.charts.trendChart.options.scales.y) {
                    this.charts.trendChart.options.scales.y.min = 1.5;
                    this.charts.trendChart.options.scales.y.max = 3.0;
                    this.charts.trendChart.options.scales.y.ticks = { stepSize: 0.25 };
                }
                this.charts.trendChart.update('none');
            }
        } catch (error) {
            console.error('Failed to update trend chart:', error);
        }
    }

    navigateToSection(section) {
        // Update active nav link
        document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.remove('active');
        });
        document.querySelector(`[data-section="${section}"]`).classList.add('active');

        // Hide all sections
        document.querySelectorAll('.content-section').forEach(sec => {
            sec.classList.remove('active');
        });

        // Show target section
        const targetSection = document.getElementById(`${section}-section`);
        if (targetSection) {
            targetSection.classList.add('active');
            this.currentSection = section;
            this.loadSectionData(section);
        }

        // Update page title
        const pageTitle = document.getElementById('pageTitle');
        if (pageTitle) {
            const titles = {
                dashboard: 'Workplace Culture Dashboard',
                analytics: 'Core Analytics',
                organizations: 'Organization Insights',
                demographics: 'Demographic Analysis',
                correlations: 'Correlation Analysis',
                insights: 'AI-Generated Insights',
                advanced: 'Advanced Analytics & PCA',
                network: 'Network Analysis & Flow'
            };
            pageTitle.textContent = titles[section] || 'Dashboard';
        }

        // Close mobile sidebar
        document.getElementById('sidebar').classList.remove('show');
    }

    async loadSectionData(section) {
        switch (section) {
            case 'analytics':
                await this.loadAnalyticsSection();
                break;
            case 'organizations':
                await this.loadOrganizationsSection();
                break;
            case 'demographics':
                await this.loadDemographicsSection();
                break;
            case 'correlations':
                await this.loadCorrelationsSection();
                break;
            case 'insights':
                await this.loadInsightsSection();
                break;
            case 'advanced':
                await this.loadAdvancedSection();
                break;
            case 'network':
                await this.loadNetworkSection();
                break;
        }
    }

    async loadAnalyticsSection() {
        try {
            // Get current domain filter if any
            const domainFilter = (document.getElementById('orgDomainFilter') || {}).value || 'all';
            const domainParam = domainFilter && domainFilter !== 'all' ? `?domain=${encodeURIComponent(domainFilter)}` : '';

            // Load section analysis for radar chart
            const sectionsData = await this.fetchWithCache(`/api/sections${domainParam}`);
            await this.populateSectionOrgFilter();
            this.setupSectionAnalysisChart(sectionsData);

            // Load distribution chart
            this.setupDistributionChart();
        } catch (error) {
            console.error('Failed to load analytics section:', error);
        }
    }

    async populateSectionOrgFilter() {
        try {
            const orgs = await this.fetchWithCache('/api/organizations/list');
            const sel = document.getElementById('sectionOrgFilter');
            const multi = document.getElementById('sectionOrgMulti');
            if (sel && Array.isArray(orgs)) {
                const existing = new Set(Array.from(sel.options).map(o=>o.value));
                orgs.forEach(o => { if (!existing.has(o.value)) { const opt=document.createElement('option'); opt.value=o.value; opt.textContent=o.label; sel.appendChild(opt);} });
            }
            if (multi && Array.isArray(orgs)) {
                multi.innerHTML = '';
                orgs.forEach(o => { const opt=document.createElement('option'); opt.value=o.value; opt.textContent=o.label; multi.appendChild(opt); });
            }

            // Toggle multi-select visibility based on comparison mode
            const cmp = document.getElementById('sectionComparisonMode');
            if (cmp) {
                const toggle = () => {
                    const isMulti = cmp.value === 'multi';
                    if (multi) multi.style.display = isMulti ? '' : 'none';
                    if (sel) sel.style.display = isMulti ? 'none' : '';
                };
                toggle();
                if (!cmp.dataset.bound) { cmp.addEventListener('change', toggle); cmp.dataset.bound = '1'; }
            }
        } catch {}
    }

    async setupSectionAnalysisChart(sectionsData) {
        const ctx = document.getElementById('sectionRadarChart');
        if (!ctx || !sectionsData) return;

        if (this.charts.sectionRadarChart) this.charts.sectionRadarChart.destroy();

        const chartType = (document.getElementById('sectionChartType') || {}).value || 'radar';
        const comparison = (document.getElementById('sectionComparisonMode') || {}).value || 'single';
        const selectedOrg = (document.getElementById('sectionOrgFilter') || {}).value || 'all';
        const multiSel = document.getElementById('sectionOrgMulti');

        // Clear cache for organization sections when switching comparison modes or organizations
        this.clearCacheForPattern('/api/organizations/sections');

        const labels = Object.keys(sectionsData).map(label => label.replace('_', ' ').toUpperCase());
        const overallValues = Object.keys(sectionsData).map(k => sectionsData[k].overall_score || 0);

        const datasets = [];
        if (comparison === 'single' && selectedOrg !== 'all') {
            const orgSec = await this.fetchWithCache(`/api/organizations/sections?organization=${encodeURIComponent(selectedOrg)}`);
            const orgVals = Object.keys(sectionsData).map(k => (orgSec[k]?.score) || (orgSec[k]?.overall_score) || 0);
            datasets.push({ label: selectedOrg, data: orgVals, backgroundColor: 'rgba(37,99,235,0.2)', borderColor: '#2563eb' });
        } else if (comparison === 'single' || (comparison !== 'multi' && selectedOrg === 'all')) {
            datasets.push({ label: 'Overall', data: overallValues, backgroundColor: 'rgba(37, 99, 235, 0.2)', borderColor: '#2563eb' });
        } else if (comparison === 'benchmark') {
            datasets.push({ label: 'Overall', data: overallValues, backgroundColor: 'rgba(148,163,184,0.2)', borderColor: '#94a3b8' });
        }

        // If benchmark mode and a specific org selected, overlay org values
        if (comparison === 'benchmark' && selectedOrg && selectedOrg !== 'all') {
            const orgSec = await this.fetchWithCache(`/api/organizations/sections?organization=${encodeURIComponent(selectedOrg)}`);
            const orgVals = Object.keys(sectionsData).map(k => (orgSec[k]?.score) || (orgSec[k]?.overall_score) || 0);
            datasets.push({ label: selectedOrg, data: orgVals, backgroundColor: 'rgba(16,185,129,0.2)', borderColor: '#10b981' });
        }

        // Multi-organization comparison
        if (comparison === 'multi' && multiSel) {
            const chosen = Array.from(multiSel.selectedOptions || []).map(o => o.value).slice(0, 5);
            const palette = ['#2563eb','#10b981','#f59e0b','#ef4444','#8b5cf6'];
            for (let i=0;i<chosen.length;i++) {
                const org = chosen[i];
                const orgSec = await this.fetchWithCache(`/api/organizations/sections?organization=${encodeURIComponent(org)}`);
                const orgVals = Object.keys(sectionsData).map(k => (orgSec[k]?.score) || (orgSec[k]?.overall_score) || 0);
                datasets.push({ label: org, data: orgVals, backgroundColor: palette[i]+'33', borderColor: palette[i] });
            }
        }

        const commonOptions = { responsive: true, maintainAspectRatio: false, animation: false };

        if (chartType === 'bar') {
            this.charts.sectionRadarChart = new Chart(ctx, {
                type: 'bar',
                data: { labels, datasets },
                options: { ...commonOptions, scales: { y: { min: 1, max: 4, ticks: { stepSize: 0.5 } } } }
            });
        } else if (chartType === 'horizontal') {
            this.charts.sectionRadarChart = new Chart(ctx, {
                type: 'bar',
                data: { labels, datasets },
                options: { ...commonOptions, indexAxis: 'y', scales: { x: { min: 1, max: 4, ticks: { stepSize: 0.5 } } } }
            });
        } else {
            this.charts.sectionRadarChart = new Chart(ctx, {
                type: 'radar',
                data: { labels, datasets },
                options: { ...commonOptions, scales: { r: { min: 1, max: 4, ticks: { stepSize: 0.5 }, animate: false } }, plugins: { legend: { display: true } } }
            });
        }
    }

    async loadOrganizationsSection() {
        try {
            await this.setupOrgBenchmarkChart();
            await this.setupOrgScatterChart();
            await this.setupTopOrgsChart();
            await this.setupOrgSizeChart();
            await this.setupDeptPerformanceChart();
            await this.setupOrgRadarChart();
            await this.setupOrgControls();
        } catch (error) {
            console.error('Failed to load organizations section:', error);
        }
    }

    // Apply client-side filters for organization-based charts
    getOrganizationFilters() {
        const minResponses = parseInt((document.getElementById('orgMinResponses') || {}).value || '0', 10);
        const domain = (document.getElementById('orgDomainFilter') || {}).value || 'all';
        const scoreRange = (document.getElementById('scoreRange') || {}).value || 'all';
        const sizeFilter = (document.getElementById('orgSizeFilter') || {}).value || 'all';
        return { minResponses, domain, scoreRange, sizeFilter };
    }

    filterOrganizations(data) {
        if (!Array.isArray(data)) return [];
        const { minResponses, domain, scoreRange, sizeFilter } = this.getOrganizationFilters();
        return data.filter(org => {
            if (minResponses && org.response_count < minResponses) return false;
            if (domain && domain !== 'all' && org.domain !== domain) return false;
            if (sizeFilter && sizeFilter !== 'all') {
                if (sizeFilter === 'large' && !(org.employee_count >= 500)) return false;
                if (sizeFilter === 'medium' && !(org.employee_count >= 100 && org.employee_count < 500)) return false;
                if (sizeFilter === 'small' && !(org.employee_count < 100)) return false;
            }
            if (scoreRange && scoreRange !== 'all') {
                if (scoreRange === 'high' && !(org.culture_score >= 3.0)) return false;
                if (scoreRange === 'medium' && !(org.culture_score >= 2.0 && org.culture_score < 3.0)) return false;
                if (scoreRange === 'low' && !(org.culture_score < 2.0)) return false;
            }
            return true;
        });
    }

    async setupOrgBenchmarkChart() {
        try {
            console.log('Setting up org benchmark chart...');
            const organizationsData = await this.fetchWithCache('/api/organizations?limit=100');
            const ctx = document.getElementById('orgBenchmarkChart');

            console.log('Org data:', organizationsData?.length || 0, 'items');
            console.log('Canvas element:', ctx ? 'found' : 'not found');

            if (!ctx || !organizationsData) {
                console.warn('Missing requirements for org benchmark chart:', { ctx: !!ctx, data: !!organizationsData });
                return;
            }

            if (this.charts.orgBenchmarkChart) {
                this.charts.orgBenchmarkChart.destroy();
            }

            // Apply filters and build organizational benchmarking visualization
            const filtered = this.filterOrganizations(organizationsData);
            const topOrgs = filtered.slice(0, 15);
            const orgNames = topOrgs.map(org => org.name.length > 20 ? org.name.substring(0, 17) + '...' : org.name);
            const cultureScores = topOrgs.map(org => org.culture_score);
            const responseCounts = topOrgs.map(org => org.response_count);

            this.charts.orgBenchmarkChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: orgNames,
                    datasets: [{
                        label: 'Culture Score',
                        data: cultureScores,
                        backgroundColor: cultureScores.map(score =>
                            score < 2.0 ? '#22c55e' : score < 2.5 ? '#f59e0b' : '#ef4444'
                        ),
                        borderWidth: 1,
                        yAxisID: 'y'
                    }, {
                        label: 'Response Count',
                        data: responseCounts,
                        type: 'line',
                        borderColor: '#3b82f6',
                        backgroundColor: 'transparent',
                        yAxisID: 'y1'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {
                        mode: 'index',
                        intersect: false,
                    },
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Organizations'
                            }
                        },
                        y: {
                            type: 'linear',
                            display: true,
                            position: 'left',
                            title: {
                                display: true,
                                text: 'Culture Score'
                            },
                            min: 1,
                            max: 4
                        },
                        y1: {
                            type: 'linear',
                            display: true,
                            position: 'right',
                            title: {
                                display: true,
                                text: 'Response Count'
                            },
                            grid: {
                                drawOnChartArea: false,
                            },
                        }
                    },
                    plugins: {
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    const org = topOrgs[context.dataIndex];
                                    if (context.datasetIndex === 0) {
                                        return `Culture Score: ${context.parsed.y} (${org.domain})`;
                                    } else {
                                        return `Responses: ${context.parsed.y} | Employees: ${org.employee_count}`;
                                    }
                                }
                            }
                        }
                    }
                }
            });
        } catch (error) {
            console.error('Failed to setup org benchmark chart:', error);
        }
    }

    async setupOrgScatterChart() {
        try {
            const organizationsData = await this.fetchWithCache('/api/organizations?limit=100');
            const ctx = document.getElementById('orgScatterChart');

            if (!ctx || !organizationsData) return;

            if (this.charts.orgScatterChart) {
                this.charts.orgScatterChart.destroy();
            }

            // Group by domain
            const domainColors = {
                'Healthcare': '#ff6b6b',
                'University': '#4ecdc4',
                'Business': '#45b7d1'
            };

            const filtered = this.filterOrganizations(organizationsData);
            const datasets = [];
            const domains = [...new Set(filtered.map(org => org.domain))];

            domains.forEach(domain => {
                const domainOrgs = filtered.filter(org => org.domain === domain);
                datasets.push({
                    label: domain,
                    data: domainOrgs.map(org => ({
                        x: org.response_count,
                        y: org.culture_score,
                        r: Math.sqrt(org.employee_count) / 50 + 5
                    })),
                    backgroundColor: domainColors[domain] || '#999',
                    borderColor: domainColors[domain] || '#999'
                });
            });

            this.charts.orgScatterChart = new Chart(ctx, {
                type: 'bubble',
                data: { datasets },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            title: { display: true, text: 'Response Count' }
                        },
                        y: {
                            title: { display: true, text: 'Culture Score' },
                            min: 1,
                            max: 4
                        }
                    },
                    plugins: {
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    const orgIndex = context.dataIndex;
                                    const domain = context.dataset.label;
                                    const domainOrgs = filtered.filter(org => org.domain === domain);
                                    const org = domainOrgs[orgIndex];
                                    return [
                                        `${org.name}`,
                                        `Domain: ${org.domain}`,
                                        `Culture Score: ${org.culture_score}`,
                                        `Responses: ${org.response_count}`,
                                        `Employees: ${org.employee_count}`
                                    ];
                                }
                            }
                        }
                    }
                }
            });
        } catch (error) {
            console.error('Failed to setup org scatter chart:', error);
        }
    }

    async setupTopOrgsChart() {
        try {
            const organizationsData = await this.fetchWithCache('/api/organizations?limit=100');
            const ctx = document.getElementById('topOrgsChart');

            if (!ctx || !organizationsData) return;

            if (this.charts.topOrgsChart) {
                this.charts.topOrgsChart.destroy();
            }

            // Apply filters
            const filtered = this.filterOrganizations(organizationsData);
            // Read sorting and top count controls
            const sortBy = (document.getElementById('orgSortBy') || {}).value || 'culture_score';
            let topCount = (document.getElementById('topOrgCount') || {}).value || '15';
            topCount = topCount === 'all' ? filtered.length : parseInt(topCount, 10);
            // Sort
            const sorted = [...filtered].sort((a, b) => {
                if (sortBy === 'culture_score') return (b.culture_score ?? 0) - (a.culture_score ?? 0);
                if (sortBy === 'response_count') return (b.response_count ?? 0) - (a.response_count ?? 0);
                if (sortBy === 'score_std') return (a.score_std ?? 0) - (b.score_std ?? 0);
                if (sortBy === 'improvement') return (b.improvement ?? 0) - (a.improvement ?? 0);
                return (b.culture_score ?? 0) - (a.culture_score ?? 0);
            });
            const topOrgs = sorted.slice(0, Math.max(1, topCount));
            const orgNames = topOrgs.map(org => org.name.length > 15 ? org.name.substring(0, 12) + '...' : org.name);
            const scores = topOrgs.map(org => org.culture_score);

            this.charts.topOrgsChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: orgNames,
                    datasets: [{
                        label: 'Culture Score',
                        data: scores,
                        backgroundColor: scores.map(score =>
                            score < 2.0 ? '#22c55e' : score < 2.5 ? '#f59e0b' : '#ef4444'
                        ),
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    indexAxis: 'y',
                    scales: {
                        x: {
                            title: { display: true, text: 'Culture Score' },
                            min: 1,
                            max: 4
                        }
                    },
                    plugins: {
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    const org = topOrgs[context.dataIndex];
                                    return [`${org.name}`, `Score: ${org.culture_score}`, `Domain: ${org.domain}`, `Responses: ${org.response_count}`];
                                }
                            }
                        }
                    }
                }
            });
        } catch (error) {
            console.error('Failed to setup top orgs chart:', error);
        }
    }

    async setupOrgSizeChart() {
        try {
            const organizationsData = await this.fetchWithCache('/api/organizations?limit=100');
            const ctx = document.getElementById('orgSizeChart');

            if (!ctx || !organizationsData) return;

            if (this.charts.orgSizeChart) {
                this.charts.orgSizeChart.destroy();
            }

            // Group by domain and employee size ranges
            const sizeRanges = [
                { min: 0, max: 1000, label: '<1K' },
                { min: 1000, max: 5000, label: '1K-5K' },
                { min: 5000, max: 20000, label: '5K-20K' },
                { min: 20000, max: 100000, label: '20K-100K' },
                { min: 100000, max: Infinity, label: '100K+' }
            ];

            const domainColors = {
                'Healthcare': '#ff6b6b',
                'University': '#4ecdc4',
                'Business': '#45b7d1'
            };

            const filtered = this.filterOrganizations(organizationsData);
            const datasets = [];
            const domains = [...new Set(filtered.map(org => org.domain))];

            domains.forEach(domain => {
                const sizeCounts = sizeRanges.map(range => {
                    return filtered.filter(org =>
                        org.domain === domain &&
                        org.employee_count >= range.min &&
                        org.employee_count < range.max
                    ).length;
                });

                datasets.push({
                    label: domain,
                    data: sizeCounts,
                    backgroundColor: domainColors[domain] || '#999'
                });
            });

            this.charts.orgSizeChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: sizeRanges.map(r => r.label),
                    datasets: datasets
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            title: { display: true, text: 'Organization Size' }
                        },
                        y: {
                            title: { display: true, text: 'Number of Organizations' }
                        }
                    }
                }
            });
        } catch (error) {
            console.error('Failed to setup org size chart:', error);
        }
    }

    async setupDeptPerformanceChart() {
        try {
            const ctx = document.getElementById('deptPerformanceChart');
            if (!ctx) return;

            await this.updateDeptPerformanceChart('all');

            // Add event listener for organization selection
            const orgSelector = document.getElementById('selectedOrganization');
            if (orgSelector) {
                orgSelector.addEventListener('change', async (e) => {
                    await this.updateDeptPerformanceChart(e.target.value);
                    await this.renderOrgSummaryCharts(e.target.value);
                });
                // Initial summary charts
                await this.renderOrgSummaryCharts(orgSelector.value || 'all');
            }
        } catch (error) {
            console.error('Failed to setup dept performance chart:', error);
        }
    }

    async updateDeptPerformanceChart(selectedOrg = 'all') {
        try {
            const ctx = document.getElementById('deptPerformanceChart');
            if (!ctx) return;

            const sectionsData = selectedOrg === 'all'
                ? await this.fetchWithCache('/api/sections')
                : await this.fetchWithCache(`/api/organizations/sections?organization=${selectedOrg}`);

            if (this.charts.deptPerformanceChart) {
                this.charts.deptPerformanceChart.destroy();
            }

            // Show section performance as radar chart
            const sectionNames = Object.keys(sectionsData);
            const sectionScores = sectionNames.map(section => {
                const sectionData = sectionsData[section];
                return sectionData.score || sectionData.overall_score || 0;
            });

            this.charts.deptPerformanceChart = new Chart(ctx, {
                type: 'radar',
                data: {
                    labels: sectionNames.map(s => s.replace(' & ', '\n& ')),
                    datasets: [{
                        label: selectedOrg === 'all' ? 'Overall Performance' : `${selectedOrg} Performance`,
                        data: sectionScores,
                        backgroundColor: 'rgba(54, 162, 235, 0.2)',
                        borderColor: 'rgba(54, 162, 235, 1)',
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: {
                        duration: 0 // Disable animations to prevent looping issues
                    },
                    scales: {
                        r: {
                            min: 1,
                            max: 4,
                            ticks: {
                                stepSize: 0.5
                            },
                            animate: false // Disable scale animations
                        }
                    }
                }
            });
        } catch (error) {
            console.error('Failed to update dept performance chart:', error);
        }
    }

    async setupOrgRadarChart() {
        try {
            const ctx = document.getElementById('orgRadarChart');
            if (!ctx) return;

            if (this.charts.orgRadarChart) {
                this.charts.orgRadarChart.destroy();
            }

            // Initialize empty radar chart
            this.charts.orgRadarChart = new Chart(ctx, {
                type: 'radar',
                data: {
                    labels: ['Power Abuse', 'Discrimination', 'Manipulative Culture', 'Failed Accountability', 'Mental Health Harm', 'Voice Erosion'],
                    datasets: []
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: {
                        duration: 0 // Disable animations to prevent looping issues
                    },
                    scales: {
                        r: {
                            min: 1,
                            max: 4,
                            ticks: {
                                stepSize: 0.5
                            },
                            animate: false // Disable scale animations
                        }
                    }
                }
            });
        } catch (error) {
            console.error('Failed to setup org radar chart:', error);
        }
    }

    async setupOrgControls() {
        try {
            const organizationsData = await this.fetchWithCache('/api/organizations?limit=100');

            // Populate organization selectors
            const orgSelectors = ['selectedOrganization', 'radarOrg1', 'radarOrg2', 'radarOrg3'];

            orgSelectors.forEach(selectorId => {
                const selector = document.getElementById(selectorId);
                if (selector && organizationsData) {
                    const currentOptions = Array.from(selector.options).map(opt => opt.value);
                    organizationsData.forEach(org => {
                        if (!currentOptions.includes(org.name)) {
                            const option = document.createElement('option');
                            option.value = org.name;
                            option.textContent = org.name;
                            selector.appendChild(option);
                        }
                    });
                }
            });

            // Add event listeners for interactive controls
            const minResponsesSelect = document.getElementById('orgMinResponses');
            if (minResponsesSelect) {
                minResponsesSelect.addEventListener('change', async (e) => {
                    await this.setupOrgBenchmarkChart();
                    await this.setupOrgScatterChart();
                    await this.setupTopOrgsChart();
                    await this.setupOrgSizeChart();
                });
            }

            const sortBySelect = document.getElementById('orgSortBy');
            if (sortBySelect && !sortBySelect.dataset.bound) {
                sortBySelect.addEventListener('change', async () => { await this.setupTopOrgsChart(); });
                sortBySelect.dataset.bound = '1';
            }

            const topCountSelect = document.getElementById('topOrgCount');
            if (topCountSelect && !topCountSelect.dataset.bound) {
                topCountSelect.addEventListener('change', async () => { await this.setupTopOrgsChart(); });
                topCountSelect.dataset.bound = '1';
            }

            // Populate domain filter
            try {
                const domains = await this.fetchWithCache('/api/domains/list');
                const domainSelect = document.getElementById('orgDomainFilter');
                if (domainSelect && Array.isArray(domains)) {
                    const current = new Set(Array.from(domainSelect.options).map(o => o.value));
                    domains.forEach(d => {
                        if (!current.has(d.value)) {
                            const opt = document.createElement('option');
                            opt.value = d.value;
                            opt.textContent = d.label;
                            domainSelect.appendChild(opt);
                        }
                    });
                }
            } catch (e) {
                console.warn('Failed to populate domain filter:', e);
            }

            // React to domain/score/size filters
            ['orgDomainFilter','scoreRange','orgSizeFilter'].forEach(id => {
                const el = document.getElementById(id);
                if (el) {
                    el.addEventListener('change', async () => {
                        await this.setupOrgBenchmarkChart();
                        await this.setupOrgScatterChart();
                        await this.setupTopOrgsChart();
                        await this.setupOrgSizeChart();
                    });
                }
            });

        } catch (error) {
            console.error('Failed to setup org controls:', error);
        }
    }

    async loadCorrelationsSection() {
        try {
            const correlationsData = await this.fetchWithCache('/api/correlations');
            this.renderCorrelationTable(correlationsData);
            this.renderCorrelationHeatmap(correlationsData);
        } catch (error) {
            console.error('Failed to load correlations section:', error);
        }
    }

    renderCorrelationHeatmap(correlationsData) {
        const ctx = document.getElementById('correlationHeatmap');
        if (!ctx || !Array.isArray(correlationsData)) return;

        // Build question label list Q1..Q22
        const labels = Array.from({ length: 22 }, (_, i) => `Q${i + 1}`.toUpperCase());
        const index = (q) => labels.indexOf(q.toUpperCase());

        // Initialize full matrix with zeros and diagonal of 1
        const size = labels.length;
        const matrix = Array.from({ length: size }, () => Array(size).fill(0));
        for (let i = 0; i < size; i++) matrix[i][i] = 1;

        // Fill from pair list
        correlationsData.forEach(p => {
            const i = index(p['Question 1']);
            const j = index(p['Question 2']);
            if (i >= 0 && j >= 0) {
                matrix[i][j] = p.Correlation;
                matrix[j][i] = p.Correlation;
            }
        });

        // Build scatter-like heatmap dataset
        const points = [];
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const v = matrix[y][x];
                points.push({ x, y, v });
            }
        }

        if (this.charts.correlationHeatmap) this.charts.correlationHeatmap.destroy();

        const colorFor = (v) => {
            const a = Math.min(1, Math.max(0.05, Math.abs(v)));
            return v >= 0 ? `rgba(37,99,235,${a})` : `rgba(220,38,38,${a})`;
        };

        const rect = ctx.getBoundingClientRect();
        const radius = Math.max(6, Math.floor(Math.min((rect.width||800)/size, (rect.height||500)/size)/2) - 2);
        this.charts.correlationHeatmap = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'Correlation',
                    data: points,
                    backgroundColor: points.map(p => colorFor(p.v)),
                    pointRadius: radius,
                    pointStyle: 'rectRounded',
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                plugins: { legend: { display: false }, tooltip: { enabled: false } },
                scales: {
                    x: {
                        type: 'linear',
                        min: -0.5,
                        max: size - 0.5,
                        ticks: {
                            stepSize: 1,
                            callback: (v) => labels[v] || ''
                        },
                        grid: { display: false }
                    },
                    y: {
                        reverse: true,
                        type: 'linear',
                        min: -0.5,
                        max: size - 0.5,
                        ticks: {
                            stepSize: 1,
                            callback: (v) => labels[v] || ''
                        },
                        grid: { display: false }
                    }
                }
            }
        });
    }

    renderCorrelationTable(correlationsData) {
        const tableBody = document.querySelector('#correlationTable tbody');
        if (!tableBody || !correlationsData) return;

        const topCorrelations = correlationsData.sort((a, b) => Math.abs(b.Correlation) - Math.abs(a.Correlation)).slice(0, 20);

        tableBody.innerHTML = topCorrelations.map(corr => `
            <tr>
                <td>${corr['Question 1']}</td>
                <td>${corr['Question 2']}</td>
                <td><span class="badge ${corr.Correlation > 0.5 ? 'bg-success' : corr.Correlation > 0.3 ? 'bg-warning' : 'bg-secondary'}">${corr.Correlation.toFixed(4)}</span></td>
                <td>${this.getCorrelationStrength(Math.abs(corr.Correlation))}</td>
            </tr>
        `).join('');
    }

    getCorrelationStrength(abs_corr) {
        if (abs_corr > 0.7) return 'Very Strong';
        if (abs_corr > 0.5) return 'Strong';
        if (abs_corr > 0.3) return 'Moderate';
        if (abs_corr > 0.1) return 'Weak';
        return 'Very Weak';
    }

    async refreshData() {
        this.showLoadingOverlay('Refreshing data...');

        try {
            // Clear cache
            this.cache.clear();

            // Refresh data from server
            const response = await fetch('/api/refresh', { method: 'POST' });
            const result = await response.json();

            if (result.success) {
                // Reload current section
                await this.loadQuickStats();
                await this.loadSectionData(this.currentSection);
                this.showToast('Data refreshed successfully', 'success');
            } else {
                this.showToast('Failed to refresh data: ' + result.error, 'error');
            }
        } catch (error) {
            console.error('Failed to refresh data:', error);
            this.showToast('Failed to refresh data', 'error');
        } finally {
            this.hideLoadingOverlay();
        }
    }

    exportData() {
        // Create export functionality
        const exportData = {
            timestamp: new Date().toISOString(),
            section: this.currentSection,
            data: this.data[this.currentSection] || {}
        };

        const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `hseg-dashboard-${this.currentSection}-${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        this.showToast('Data exported successfully', 'success');
    }

    formatNumber(num) {
        if (!num) return '0';
        if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
        if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
        return num.toString();
    }

    showToast(message, type = 'info') {
        // Simple toast notification
        const toast = document.createElement('div');
        toast.className = `alert alert-${type === 'success' ? 'success' : type === 'error' ? 'danger' : 'info'} position-fixed`;
        toast.style.cssText = 'top: 20px; right: 20px; z-index: 10000; min-width: 300px;';
        toast.textContent = message;

        document.body.appendChild(toast);

        setTimeout(() => {
            toast.remove();
        }, 3000);
    }

    showError(message) {
        this.showToast(message, 'error');
    }

    // Advanced Analytics Section
    async loadAdvancedSection() {
        try {
            // Load PCA data
            await this.setupPCAChart();
            await this.setupVarianceChart();
            await this.setupPCALoadingsChart();
            await this.setupSectionCorrelationMatrix();
            await this.setupTreemapChart();
            await this.setupDemographicChart();
            // Prime hierarchical tab filters with domain options
            try {
                const domains = await this.fetchWithCache('/api/domains/list');
                const sel = document.getElementById('hierarchicalDomain');
                if (sel && Array.isArray(domains)) {
                    const existing = new Set(Array.from(sel.options).map(o=>o.value));
                    domains.forEach(d => { if (!existing.has(d.value)) { const o=document.createElement('option'); o.value=d.value; o.textContent=d.label; sel.appendChild(o);} });
                }
            } catch {}
        } catch (error) {
            console.error('Failed to load advanced section:', error);
        }
    }

    async setupPCALoadingsChart() {
        try {
            const clusteringData = await this.fetchWithCache('/api/advanced/clustering');
            const ctx = document.getElementById('pcaLoadingsChart');
            if (!ctx || !clusteringData || !clusteringData.loadings) return;

            if (this.charts.pcaLoadingsChart) {
                this.charts.pcaLoadingsChart.destroy();
            }

            const featureSel = document.getElementById('pcaFeatures');
            const mode = featureSel ? featureSel.value : 'all';
            const q = clusteringData.loadings.questions.map(s => s.toUpperCase());
            const pc1 = clusteringData.loadings.pc1.map(Math.abs);
            const pc2 = clusteringData.loadings.pc2.map(Math.abs);

            let labels, d1, d2;
            if (mode === 'sections') {
                const map = {
                    'POWER ABUSE & SUPPRESSION': ['Q1','Q2','Q3','Q4'],
                    'DISCRIMINATION & EXCLUSION': ['Q5','Q6','Q7'],
                    'MANIPULATIVE WORK CULTURE': ['Q8','Q9','Q10'],
                    'FAILURE OF ACCOUNTABILITY': ['Q11','Q12','Q13','Q14'],
                    'MENTAL HEALTH HARM': ['Q15','Q16','Q17','Q18'],
                    'EROSION OF VOICE & AUTONOMY': ['Q19','Q20','Q21','Q22']
                };
                labels = Object.keys(map);
                d1 = labels.map(sec => {
                    const idxs = map[sec].map(code => q.indexOf(code)).filter(i => i>=0);
                    const vals = idxs.map(i => pc1[i]);
                    return vals.length ? vals.reduce((a,b)=>a+b,0)/vals.length : 0;
                });
                d2 = labels.map(sec => {
                    const idxs = map[sec].map(code => q.indexOf(code)).filter(i => i>=0);
                    const vals = idxs.map(i => pc2[i]);
                    return vals.length ? vals.reduce((a,b)=>a+b,0)/vals.length : 0;
                });
            } else if (mode === 'top_variance') {
                const idx = pc1.map((v,i)=>[v,i]).sort((a,b)=>b[0]-a[0]).slice(0,10).map(t=>t[1]);
                labels = idx.map(i=>q[i]);
                d1 = idx.map(i=>pc1[i]);
                d2 = idx.map(i=>pc2[i]);
            } else {
                const idx = pc1.map((v,i)=>[v,i]).sort((a,b)=>b[0]-a[0]).slice(0,10).map(t=>t[1]);
                labels = idx.map(i=>q[i]);
                d1 = idx.map(i=>pc1[i]);
                d2 = idx.map(i=>pc2[i]);
            }

            this.charts.pcaLoadingsChart = new Chart(ctx, {
                type: 'bar',
                data: { labels, datasets: [
                    { label: 'PC1 |loading|', data: d1, backgroundColor: '#2563eb' },
                    { label: 'PC2 |loading|', data: d2, backgroundColor: '#dc2626' }
                ] },
                options: { responsive: true, maintainAspectRatio: false, indexAxis: 'y' }
            });
        } catch (error) {
            console.error('Failed to setup PCA loadings chart:', error);
        }
    }

    // Demographics Section (standalone)
    async loadDemographicsSection() {
        try {
            const container = document.getElementById('demographicsContainer');
            if (!container) return;

            // Inject UI if empty
            if (!container.dataset.initialized) {
                container.innerHTML = `
                    <div class="row g-3">
                        <div class="col-12">
                            <div class="filter-card">
                                <div class="filter-header"><h6><i class="fas fa-filter me-2"></i>Demographics Filters</h6></div>
                                <div class="filter-body">
                                    <div class="row g-3">
                                        <div class="col-md-4">
                                            <label class="form-label">Category</label>
                                            <select class="form-select" id="demographicsCategory">
                                                <option value="tenure" selected>Tenure</option>
                                                <option value="position_level">Position Level</option>
                                                <option value="age_range">Age Range</option>
                                                <option value="gender_identity">Gender Identity</option>
                                                <option value="race_ethnicity">Race / Ethnicity</option>
                                                <option value="education_level">Education Level</option>
                                            </select>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div class="col-12">
                            <div class="chart-card">
                                <div class="chart-header">
                                    <h5 class="chart-title">Demographics Overview</h5>
                                </div>
                                <div class="chart-body">
                                    <canvas id="demographicsOverviewChart" style="height: 460px;"></canvas>
                                </div>
                            </div>
                        </div>

                        <div class="col-12">
                            <div class="chart-card">
                                <div class="chart-header">
                                    <h5 class="chart-title">Domain vs Tenure Heatmap</h5>
                                </div>
                                <div class="chart-body">
                                    <canvas id="tenureHeatmap" style="height: 460px;"></canvas>
                                </div>
                            </div>
                        </div>
                    </div>`;

                container.dataset.initialized = 'true';

                const category = document.getElementById('demographicsCategory');
                if (category) {
                    category.addEventListener('change', async (e) => {
                        await this.renderDemographicsOverview(e.target.value);
                    });
                }
            }

            // Initial render
            const initial = (document.getElementById('demographicsCategory') || {}).value || 'tenure';
            await this.renderDemographicsOverview(initial);
            await this.renderTenureHeatmap();
        } catch (error) {
            console.error('Failed to load demographics section:', error);
        }
    }

    async renderDemographicsOverview(type) {
        try {
            const data = await this.fetchWithCache(`/api/demographics?type=${type}`);
            const ctx = document.getElementById('demographicsOverviewChart');
            if (!ctx || !data) return;

            if (this.charts.demographicsOverviewChart) {
                this.charts.demographicsOverviewChart.destroy();
            }

            const labels = Object.keys(data);
            const scores = labels.map(l => data[l]?.avg_culture_score || 0);
            const counts = labels.map(l => data[l]?.count || 0);

            this.charts.demographicsOverviewChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels,
                    datasets: [
                        { label: 'Avg Culture Score', data: scores, backgroundColor: '#2563eb', yAxisID: 'y' },
                        { label: 'Response Count', data: counts, type: 'line', borderColor: '#dc2626', backgroundColor: 'transparent', yAxisID: 'y1' }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: { type: 'linear', position: 'left', min: 0 },
                        y1: { type: 'linear', position: 'right', grid: { drawOnChartArea: false } }
                    },
                    plugins: { legend: { position: 'top' } }
                }
            });
        } catch (e) {
            console.error('Failed to render demographics overview:', e);
        }
    }

    async renderTenureHeatmap() {
        try {
            const data = await this.fetchWithCache('/api/tenure/matrix');
            const ctx = document.getElementById('tenureHeatmap');
            if (!ctx || !data || !data.matrix || data.matrix.length === 0) return;

            if (this.charts.tenureHeatmap) {
                this.charts.tenureHeatmap.destroy();
            }

            const domains = data.domains;
            const tenures = data.tenure;
            const points = [];
            for (let y = 0; y < domains.length; y++) {
                for (let x = 0; x < tenures.length; x++) {
                    const v = data.matrix[y][x];
                    if (v !== null && v !== undefined) points.push({ x, y, v });
                }
            }

            const color = (v) => {
                const t = Math.max(0, Math.min(1, (v - 1) / (4 - 1)));
                const b = Math.round(255 * (1 - t));
                const r = Math.round(255 * t);
                return `rgba(${r},80,${b},0.85)`;
            };

            this.charts.tenureHeatmap = new Chart(ctx, {
                type: 'scatter',
                data: { datasets: [{ data: points, backgroundColor: points.map(p => color(p.v)), pointRadius: 16, pointStyle: 'rectRounded' }] },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c)=> `${domains[c.raw.y]} / ${tenures[c.raw.x]}: ${c.raw.v.toFixed(2)}` } } },
                    scales: {
                        x: { type: 'linear', min: -0.5, max: tenures.length - 0.5, ticks: { stepSize: 1, callback: (v)=> tenures[v] || '' }, grid: { display: false } },
                        y: { type: 'linear', reverse: true, min: -0.5, max: domains.length - 0.5, ticks: { stepSize: 1, callback: (v)=> domains[v] || '' }, grid: { display: false } }
                    }
                }
            });
        } catch (error) {
            console.error('Failed to render tenure heatmap:', error);
        }
    }

    async setupPCAChart() {
        try {
            const clusteringData = await this.fetchWithCache('/api/advanced/clustering');
            const ctx = document.getElementById('pcaChart');

            if (!ctx || !clusteringData.pca_data) return;

            if (this.charts.pcaChart) {
                this.charts.pcaChart.destroy();
            }

            // Group data by domain
            const domainColors = {
                'Healthcare': '#ff6b6b',
                'University': '#4ecdc4',
                'Business': '#45b7d1'
            };

            const datasets = [];
            const domains = [...new Set(clusteringData.pca_data.map(d => d.domain))];

            domains.forEach(domain => {
                const domainData = clusteringData.pca_data.filter(d => d.domain === domain);
                datasets.push({
                    label: domain,
                    data: domainData.map(d => ({x: d.x, y: d.y})),
                    backgroundColor: domainColors[domain] || '#999',
                    pointRadius: 3,
                    pointHoverRadius: 5
                });
            });

            this.charts.pcaChart = new Chart(ctx, {
                type: 'scatter',
                data: { datasets },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { title: { display: true, text: 'PC1' }},
                        y: { title: { display: true, text: 'PC2' }}
                    },
                    plugins: {
                        title: { display: true, text: 'PCA - Cultural Dimensions' },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    const point = clusteringData.pca_data[context.dataIndex];
                                    return `${point.organization} (${point.domain})`;
                                }
                            }
                        }
                    }
                }
            });
        } catch (error) {
            console.error('Failed to setup PCA chart:', error);
        }
    }

    async setupVarianceChart() {
        try {
            const clusteringData = await this.fetchWithCache('/api/advanced/clustering');
            const ctx = document.getElementById('varianceChart');

            if (!ctx || !clusteringData.explained_variance) return;

            if (this.charts.varianceChart) {
                this.charts.varianceChart.destroy();
            }

            const explainedVar = clusteringData.explained_variance;
            const cumulative = explainedVar.reduce((acc, val, i) => {
                acc.push((acc[i-1] || 0) + val);
                return acc;
            }, []);

            this.charts.varianceChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: explainedVar.map((_, i) => `PC${i+1}`),
                    datasets: [{
                        label: 'Explained Variance',
                        data: explainedVar,
                        backgroundColor: '#2563eb',
                        yAxisID: 'y'
                    }, {
                        label: 'Cumulative',
                        data: cumulative,
                        type: 'line',
                        borderColor: '#dc2626',
                        backgroundColor: 'transparent',
                        yAxisID: 'y1'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: { type: 'linear', display: true, position: 'left' },
                        y1: { type: 'linear', display: true, position: 'right', grid: { drawOnChartArea: false }}
                    }
                }
            });
        } catch (error) {
            console.error('Failed to setup variance chart:', error);
        }
    }

    async setupSectionCorrelationMatrix() {
        try {
            const resp = await this.fetchWithCache('/api/sections/correlation');
            const ctx = document.getElementById('sectionCorrelationMatrix');
            if (!ctx || !resp || !resp.labels || !resp.matrix || resp.matrix.length === 0) return;

            const labels = resp.labels;
            const size = labels.length;
            const points = [];
            for (let y = 0; y < size; y++) {
                for (let x = 0; x < size; x++) {
                    points.push({ x, y, v: resp.matrix[y][x] });
                }
            }

            if (this.charts.sectionCorrelationMatrix) this.charts.sectionCorrelationMatrix.destroy();

            const colorFor = (v) => {
                const a = Math.min(1, Math.max(0.05, Math.abs(v)));
                return v >= 0 ? `rgba(37,99,235,${a})` : `rgba(220,38,38,${a})`;
            };

            const rect = ctx.getBoundingClientRect();
            const radius = Math.max(6, Math.floor(Math.min((rect.width||800)/size, (rect.height||500)/size)/2) - 2);
            this.charts.sectionCorrelationMatrix = new Chart(ctx, {
                type: 'scatter',
                data: {
                    datasets: [{
                        data: points,
                        backgroundColor: points.map(p => colorFor(p.v)),
                        pointRadius: radius,
                        pointStyle: 'rectRounded',
                        borderWidth: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: false,
                    plugins: { legend: { display: false }, tooltip: { enabled: false } },
                    scales: {
                        x: { type: 'linear', min: -0.5, max: size - 0.5, ticks: { stepSize: 1, callback: (v) => labels[v] || '' }, grid: { display: false } },
                        y: { type: 'linear', min: -0.5, max: size - 0.5, reverse: true, ticks: { stepSize: 1, callback: (v) => labels[v] || '' }, grid: { display: false } }
                    }
                }
            });
        } catch (error) {
            console.error('Failed to setup correlation matrix:', error);
        }
    }

    async setupTreemapChart() {
        try {
            const hierarchicalData = await this.fetchWithCache('/api/advanced/hierarchical');
            const container = document.getElementById('treemapChart');

            if (!container || !hierarchicalData) return;

            // Simple treemap implementation using divs
            container.innerHTML = '';
            container.style.position = 'relative';
            container.style.height = '300px';

            const maxCount = Math.max(...hierarchicalData.map(d => d.count));

            hierarchicalData.slice(0, 20).forEach((item, index) => {
                const div = document.createElement('div');
                const size = Math.sqrt(item.count / maxCount) * 100;

                div.style.position = 'absolute';
                div.style.width = `${size}px`;
                div.style.height = `${size}px`;
                div.style.backgroundColor = this.getColorByDomain(item.domain);
                div.style.border = '1px solid #fff';
                div.style.left = `${(index % 5) * 80}px`;
                div.style.top = `${Math.floor(index / 5) * 60}px`;
                div.style.display = 'flex';
                div.style.alignItems = 'center';
                div.style.justifyContent = 'center';
                div.style.fontSize = '10px';
                div.style.color = 'white';
                div.style.fontWeight = 'bold';
                div.style.textAlign = 'center';
                div.style.cursor = 'pointer';
                div.title = `${item.organization} - ${item.department}\nCount: ${item.count}\nScore: ${item.avg_culture_score}`;
                const label = (item.department || '').toString();
                const short = label.length > 12 ? label.slice(0, 9) + '…' : label;
                div.innerHTML = `${short}<br/><small>${item.count}</small>`;

                container.appendChild(div);
            });
        } catch (error) {
            console.error('Failed to setup treemap chart:', error);
        }
    }

    async setupDemographicChart() {
        try {
            const demographicsData = await this.fetchWithCache('/api/demographics?type=tenure');
            const ctx = document.getElementById('demographicChart');

            if (!ctx || !demographicsData) return;

            if (this.charts.demographicChart) {
                this.charts.demographicChart.destroy();
            }

            const labels = Object.keys(demographicsData);
            const scores = labels.map(label => demographicsData[label].avg_culture_score || 0);
            const counts = labels.map(label => demographicsData[label].count || 0);

            this.charts.demographicChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Average Culture Score',
                        data: scores,
                        backgroundColor: '#2563eb',
                        yAxisID: 'y'
                    }, {
                        label: 'Response Count',
                        data: counts,
                        type: 'line',
                        borderColor: '#dc2626',
                        backgroundColor: 'transparent',
                        yAxisID: 'y1'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: { type: 'linear', display: true, position: 'left' },
                        y1: { type: 'linear', display: true, position: 'right', grid: { drawOnChartArea: false }}
                    }
                }
            });

            // Setup demographic type selector
            const selector = document.getElementById('demographicType');
            if (selector) {
                selector.addEventListener('change', async (e) => {
                    await this.updateDemographicChart(e.target.value);
                });
            }
        } catch (error) {
            console.error('Failed to setup demographic chart:', error);
        }
    }

    async updateDemographicChart(type) {
        try {
            const demographicsData = await this.fetchWithCache(`/api/demographics?type=${type}`);

            if (this.charts.demographicChart && demographicsData) {
                const labels = Object.keys(demographicsData);
                const scores = labels.map(label => demographicsData[label].avg_culture_score || 0);
                const counts = labels.map(label => demographicsData[label].count || 0);

                this.charts.demographicChart.data.labels = labels;
                this.charts.demographicChart.data.datasets[0].data = scores;
                this.charts.demographicChart.data.datasets[1].data = counts;
                this.charts.demographicChart.update();
            }
        } catch (error) {
            console.error('Failed to update demographic chart:', error);
        }
    }

    // Network Analysis Section
    async loadNetworkSection() {
        try {
            await this.setupNetworkChart();
            await this.setupSankeyChart();
            await this.updateNetworkStats();
        } catch (error) {
            console.error('Failed to load network section:', error);
        }
    }

    async setupNetworkChart() {
        try {
            const networkData = await this.fetchWithCache('/api/advanced/network');
            const container = document.getElementById('networkChart');

            if (!container || !networkData.nodes) return;

            // Simple force-like network visualization using SVG
            container.innerHTML = '';
            const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
            svg.style.width = '100%';
            svg.style.height = '500px';
            container.appendChild(svg);

            const nodes = networkData.nodes;
            const links = networkData.links;

            // Initialize positions randomly around center
            const width = container.clientWidth || 800;
            const height = 500;
            nodes.forEach((n,i)=>{ n.x = (width/2) + (Math.random()-0.5)*200; n.y = (height/2) + (Math.random()-0.5)*200; });

            // Build adjacency list for quick degree / components
            const adj = new Map(nodes.map(n=>[n.id, new Set()]));
            links.forEach(l=>{ if (adj.has(l.source) && adj.has(l.target)) { adj.get(l.source).add(l.target); adj.get(l.target).add(l.source);} });

            // Compute simple connected components
            const compId = new Map();
            let cid = 0;
            nodes.forEach(n=>{ if (!compId.has(n.id)) { const q=[n.id]; compId.set(n.id,cid); while(q.length){ const u=q.pop(); (adj.get(u)||[]).forEach(v=>{ if(!compId.has(v)){ compId.set(v,cid); q.push(v);} }); } cid++; } });

            // Run a few force iterations (very light)
            for (let iter=0; iter<200; iter++) {
                // repulsion
                for (let i=0;i<nodes.length;i++){
                    for (let j=i+1;j<nodes.length;j++){
                        const dx = nodes[j].x - nodes[i].x; const dy = nodes[j].y - nodes[i].y; const d2 = dx*dx+dy*dy+0.01; const f = 2000/d2; const fx = f*dx; const fy = f*dy; nodes[j].x += fx; nodes[j].y += fy; nodes[i].x -= fx; nodes[i].y -= fy; }
                }
                // attraction on edges
                links.forEach(l=>{
                    const a = nodes.find(n=>n.id===l.source); const b = nodes.find(n=>n.id===l.target);
                    if (!a||!b) return;
                    const dx=b.x-a.x; const dy=b.y-a.y; const dist=Math.sqrt(dx*dx+dy*dy)+0.01; const k=0.02; const fx=k*dx; const fy=k*dy; a.x+=fx; a.y+=fy; b.x-=fx; b.y-=fy;
                });
                // bound
                nodes.forEach(n=>{ n.x=Math.max(20,Math.min(width-20,n.x)); n.y=Math.max(20,Math.min(height-20,n.y)); });
            }

            // Draw links
            links.forEach(link => {
                const source = nodes.find(n => n.id === link.source);
                const target = nodes.find(n => n.id === link.target);

                if (source && target) {
                    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                    line.setAttribute('x1', source.x);
                    line.setAttribute('y1', source.y);
                    line.setAttribute('x2', target.x);
                    line.setAttribute('y2', target.y);
                    line.setAttribute('stroke', '#93a3b8');
                    line.setAttribute('stroke-width', Math.max(1, link.weight * 3));
                    line.setAttribute('opacity', '0.6');
                    svg.appendChild(line);
                }
            });

            // Draw nodes
            nodes.forEach(node => {
                const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                circle.setAttribute('cx', node.x);
                circle.setAttribute('cy', node.y);
                const degree = (adj.get(node.id) || new Set()).size;
                const r = Math.max(6, Math.min(18, degree + node.size * 0.4));
                circle.setAttribute('r', r);
                // Color by component hint
                const comp = compId.get(node.id) || 0;
                const hue = (comp*65)%360;
                circle.setAttribute('fill', `hsl(${hue} 70% 50%)`);
                circle.setAttribute('stroke', '#fff');
                circle.setAttribute('stroke-width', '2');
                circle.style.cursor = 'pointer';

                const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
                title.textContent = `${node.id}\nDomain: ${node.domain}\nScore: ${node.culture_score}\nResponses: ${node.responses}\nDegree: ${degree}\nCluster: ${comp}`;
                circle.appendChild(title);

                svg.appendChild(circle);

                // Add text labels for larger nodes
                if (node.size > 15) {
                    const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                    text.setAttribute('x', node.x);
                    text.setAttribute('y', node.y + 4);
                    text.setAttribute('text-anchor', 'middle');
                    text.setAttribute('font-size', '10px');
                    text.setAttribute('fill', '#fff');
                    text.textContent = node.id.substring(0, 8);
                    svg.appendChild(text);
                }
            });
        } catch (error) {
            console.error('Failed to setup network chart:', error);
        }
    }

    async setupSankeyChart() {
        try {
            const domainSel = document.getElementById('flowDomain');
            const minSel = document.getElementById('flowMinCount');
            const domain = (domainSel && domainSel.value) || 'all';
            const minCount = parseInt((minSel && minSel.value) || '5', 10);

            // Ensure domains loaded
            try {
                if (domainSel && domainSel.options.length <= 1) {
                    const domains = await this.fetchWithCache('/api/domains/list');
                    const have = new Set(Array.from(domainSel.options).map(o=>o.value));
                    domains.forEach(d=>{ if(!have.has(d.value)){ const o=document.createElement('option'); o.value=d.value; o.textContent=d.label; domainSel.appendChild(o);} });
                }
            } catch {}

            // Fetch hierarchical data filtered
            const params = new URLSearchParams({ min_count: String(minCount), domain, _: String(Date.now()) });
            const hierarchical = await this.fetchWithCache(`/api/advanced/hierarchical?${params.toString()}`);
            if (!hierarchical) return;

            // Aggregate flows Domain->Org and Org->Dept
            const byOrg = new Map();
            const byDept = new Map();
            hierarchical.forEach(d => {
                byOrg.set(d.organization, (byOrg.get(d.organization)||0) + d.count);
                const key = `${d.organization}||${d.department}`;
                byDept.set(key, (byDept.get(key)||0) + d.count);
            });

            const topOrgs = Array.from(byOrg.entries()).sort((a,b)=>b[1]-a[1]).slice(0,12);
            const orgLabels = topOrgs.map(e=>e[0]);
            const orgCounts = topOrgs.map(e=>e[1]);

            const filteredByDept = Array.from(byDept.entries()).filter(e=>orgLabels.includes(e[0].split('||')[0]));
            const topDept = filteredByDept.sort((a,b)=>b[1]-a[1]).slice(0,12);
            const deptLabels = topDept.map(e=>e[0].split('||')[1]);
            const deptCounts = topDept.map(e=>e[1]);

            // Render bar charts
            const c1 = document.getElementById('domainOrgFlowChart');
            const c2 = document.getElementById('orgDeptFlowChart');
            if (c1) {
                if (this.charts.domainOrgFlowChart) this.charts.domainOrgFlowChart.destroy();
                this.charts.domainOrgFlowChart = new Chart(c1, {
                    type: 'bar',
                    data: { labels: orgLabels, datasets: [{ label: 'Responses', data: orgCounts, backgroundColor: '#2563eb' }] },
                    options: { responsive: true, maintainAspectRatio: false, indexAxis: 'y' }
                });
            }
            if (c2) {
                if (this.charts.orgDeptFlowChart) this.charts.orgDeptFlowChart.destroy();
                this.charts.orgDeptFlowChart = new Chart(c2, {
                    type: 'bar',
                    data: { labels: deptLabels, datasets: [{ label: 'Responses', data: deptCounts, backgroundColor: '#10b981' }] },
                    options: { responsive: true, maintainAspectRatio: false, indexAxis: 'y' }
                });
            }

            // Bind change handlers to refresh
            if (domainSel && !domainSel.dataset.bound) {
                domainSel.addEventListener('change', ()=> this.setupSankeyChart());
                domainSel.dataset.bound = '1';
            }
            if (minSel && !minSel.dataset.bound) {
                minSel.addEventListener('change', ()=> this.setupSankeyChart());
                minSel.dataset.bound = '1';
            }
        } catch (error) {
            console.error('Failed to setup sankey chart:', error);
        }
    }

    async updateNetworkStats() {
        try {
            const networkData = await this.fetchWithCache('/api/advanced/network');

            if (networkData.nodes && networkData.links) {
                document.getElementById('totalOrgs').textContent = networkData.nodes.length;
                document.getElementById('totalConnections').textContent = networkData.links.length;

                const avgSimilarity = networkData.links.length > 0
                    ? (networkData.links.reduce((sum, link) => sum + link.similarity, 0) / networkData.links.length).toFixed(3)
                    : '0.000';
                document.getElementById('avgSimilarity').textContent = avgSimilarity;

                const maxConnections = networkData.nodes.length * (networkData.nodes.length - 1) / 2;
                const density = maxConnections > 0 ? (networkData.links.length / maxConnections).toFixed(3) : '0.000';
                document.getElementById('networkDensity').textContent = density;

                // Top similar pairs for quick insight
                const topPairs = [...networkData.links]
                    .sort((a,b) => b.similarity - a.similarity)
                    .slice(0, 5)
                    .map(l => `${l.source} — ${l.target}: ${l.similarity.toFixed(2)}`)
                    .join('<br/>');
                const tpEl = document.getElementById('topPairs');
                if (tpEl) tpEl.innerHTML = topPairs || 'No strong pairs available.';
            }
        } catch (error) {
            console.error('Failed to update network stats:', error);
        }
    }

    getColorByDomain(domain) {
        const colors = {
            'Healthcare': '#ff6b6b',
            'University': '#4ecdc4',
            'Business': '#45b7d1'
        };
        return colors[domain] || '#999999';
    }

    // New Tab Functionality
    setupSectionTabs() {
        // Setup tab functionality for all section tabs
        document.querySelectorAll('.section-tabs .nav-link').forEach(tabButton => {
            tabButton.addEventListener('click', (e) => {
                e.preventDefault();
                const tabId = tabButton.getAttribute('data-bs-target');
                if (tabId) {
                    this.activateTab(tabButton, tabId);
                }
            });
        });

        // Initialize Bootstrap tabs
        document.querySelectorAll('[data-bs-toggle="tab"]').forEach(tab => {
            tab.addEventListener('shown.bs.tab', (e) => {
                const targetTab = e.target.getAttribute('data-bs-target');
                this.onTabShown(targetTab);
            });
        });
    }

    activateTab(tabButton, targetId) {
        // Remove active class from all tab buttons in this section
        const tabContainer = tabButton.closest('.section-tabs');
        tabContainer.querySelectorAll('.nav-link').forEach(btn => btn.classList.remove('active'));

        // Add active class to clicked tab
        tabButton.classList.add('active');

        // Hide all tab panes in this section
        const contentContainer = tabContainer.nextElementSibling;
        contentContainer.querySelectorAll('.tab-pane').forEach(pane => {
            pane.classList.remove('show', 'active');
        });

        // Show target tab pane
        const targetPane = document.querySelector(targetId);
        if (targetPane) {
            targetPane.classList.add('show', 'active');
            this.resizeChartsInTab(targetPane);
        }
    }

    onTabShown(targetTabId) {
        // Called when a tab is shown - resize charts and reload data if needed
        const targetPane = document.querySelector(targetTabId);
        if (targetPane) {
            setTimeout(() => {
                this.resizeChartsInTab(targetPane);
                this.loadTabData(targetPane);
            }, 100);
        }
    }

    resizeChartsInTab(tabPane) {
        // Avoid resize loops; perform a no-animation update instead
        const canvases = tabPane.querySelectorAll('canvas');
        canvases.forEach(canvas => {
            const chart = Chart.getChart(canvas);
            if (chart) {
                try { chart.update('none'); } catch {}
            }
        });
    }

    destroyChartsInTab(tabPane) {
        // Properly destroy all charts in the given tab pane to prevent animation conflicts
        const canvases = tabPane.querySelectorAll('canvas');
        canvases.forEach(canvas => {
            const chart = Chart.getChart(canvas);
            if (chart) {
                chart.destroy();
            }
        });
    }

    loadTabData(tabPane) {
        // Load data specific to the tab that was shown
        const tabId = tabPane.id;
        const sectionId = tabPane.closest('.content-section').id;

        // Route to appropriate data loading method based on section and tab
        if (sectionId === 'organizations-section') {
            this.loadOrganizationTabData(tabId);
        } else if (sectionId === 'analytics-section') {
            this.loadAnalyticsTabData(tabId);
        } else if (sectionId === 'advanced-section') {
            this.loadAdvancedTabData(tabId);
        }
    }

    async loadOrganizationTabData(tabId) {
        switch (tabId) {
            case 'benchmark':
                await this.setupOrgBenchmarkChart();
                break;
            case 'performance':
                await this.setupOrgScatterChart();
                await this.setupTopOrgsChart();
                break;
            case 'distribution':
                await this.setupOrgSizeChart();
                await this.setupDeptPerformanceChart();
                break;
            case 'comparison':
                await this.setupOrgRadarChart();
                break;
        }
    }

    async loadAnalyticsTabData(tabId) {
        switch (tabId) {
            case 'sections':
                await this.loadAnalyticsSection();
                break;
            case 'trends':
                await this.setupAnalyticsTrendChart();
                break;
            case 'distributions':
                await this.setupDistributionChart();
                break;
            case 'insights':
                await this.generateStatisticalInsights();
                break;
        }
    }

    async setupAnalyticsTrendChart() {
        try {
            const range = (document.getElementById('analyticsTrendTimeRange')||{}).value || '30d';
            await this.updateAnalyticsTrendChart(range);
        } catch (e) { console.error('Failed to setup analytics trend chart:', e); }
    }

    async updateAnalyticsTrendChart(range) {
        try {
            const days = range === '7d' ? 7 : range === '30d' ? 30 : 90;
            const metric = (document.getElementById('analyticsTrendMetric')||{}).value || 'culture_score';
            const granularity = (document.getElementById('analyticsTrendGranularity')||{}).value || 'weekly';
            const smoothing = (document.getElementById('analyticsTrendSmoothing')||{}).value || 'none';

            const data = await this.fetchWithCache(`/api/analytics/trend?days=${days}&metric=${metric}&granularity=${granularity}&smoothing=${smoothing}`);
            const ctx = document.getElementById('analyticsTrendChart');
            if (!ctx || !data.labels) return;
            if (this.charts.analyticsTrendChart) this.charts.analyticsTrendChart.destroy();

            // Adjust chart options based on metric type
            let chartOptions = {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: metric === 'culture_score' ? { min: 1.5, max: 3.0, ticks: { stepSize: 0.25 } } :
                        metric === 'section_scores' ? { min: 1.0, max: 4.0, ticks: { stepSize: 0.5 } } : {}
                },
                plugins: {
                    legend: {
                        display: metric === 'section_scores',
                        position: 'right'
                    }
                }
            };

            this.charts.analyticsTrendChart = new Chart(ctx, {
                type: 'line',
                data: { labels: data.labels, datasets: data.datasets },
                options: chartOptions
            });
        } catch (e) { console.error('Failed to update analytics trend chart:', e); }
    }

    async loadAdvancedTabData(tabId) {
        switch (tabId) {
            case 'pca':
                await this.loadAdvancedSection();
                break;
            case 'clustering':
                await this.setupClusteringChart();
                break;
            case 'correlations-advanced':
                await this.setupAdvancedCorrelations();
                break;
            case 'hierarchical':
                await this.setupHierarchicalChart();
                break;
        }
    }

    setupFilterEventListeners() {
        // Global filter change handlers
        const filterSelectors = [
            '#orgMinResponses', '#orgDomainFilter', '#scoreRange', '#orgSizeFilter',
            '#orgSortBy', '#performanceTimeRange', '#topOrgCount',
            '#distributionGroupBy', '#distributionMetric',
            '#sectionOrgFilter', '#sectionOrgMulti', '#sectionChartType', '#sectionComparisonMode',
            '#analyticsTrendTimeRange', '#analyticsTrendMetric', '#analyticsTrendGranularity', '#analyticsTrendSmoothing',
            '#distributionType', '#distributionBins',
            '#pcaComponents', '#pcaFeatures', '#pcaScaling',
            '#clusteringAlgorithm', '#clusterCount', '#clusterMetric', '#clusterBy',
            '#correlationType', '#minCorrelation', '#correlationLevel',
            '#hierarchicalType', '#demographicType', '#hierarchicalMinResponses', '#hierarchicalDomain'
        ];

        filterSelectors.forEach(selector => {
            const element = document.querySelector(selector);
            if (element) {
                element.addEventListener('change', (e) => {
                    this.onFilterChange(selector, e.target.value);
                });
            }
        });

        // Export button handlers
        const exportButtons = document.querySelectorAll('[id^="export"]');
        exportButtons.forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.exportChart(e.target.closest('.chart-card'));
            });
        });

        // Reset button handlers
        const resetButtons = document.querySelectorAll('[id^="reset"]');
        resetButtons.forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.resetFilters(e.target.closest('.tab-pane'));
            });
        });
    }

    async setupHierarchicalChart() {
        try {
            const type = (document.getElementById('hierarchicalType') || {}).value || 'treemap';
            const minResp = parseInt((document.getElementById('hierarchicalMinResponses') || {}).value || '5', 10);
            const dom = (document.getElementById('hierarchicalDomain') || {}).value || 'all';

            const treemapEl = document.getElementById('treemapChart');
            const sunburstWrap = document.getElementById('sunburstContainer');
            if (!treemapEl) return;

            if (type === 'treemap') {
                if (sunburstWrap) sunburstWrap.style.display = 'none';
                treemapEl.style.display = '';
                const params = new URLSearchParams({ min_count: String(minResp), domain: dom, _: String(Date.now()) });
                const data = await this.fetchWithCache(`/api/advanced/hierarchical?${params.toString()}`);
                treemapEl.innerHTML = '';
                // If Plotly available, render authentic treemap; else fallback
                if (window.Plotly && Array.isArray(data)) {
                    const labels = data.map(d=>`${d.domain} / ${d.organization} / ${d.department}`);
                    const parents = data.map(d=>`${d.domain} / ${d.organization}`);
                    const values = data.map(d=>d.count);
                    const colors = data.map(d=>d.avg_culture_score);
                    const trace = {
                        type: 'treemap',
                        labels,
                        parents,
                        values,
                        marker: { colors, colorscale: 'RdBu', reversescale: true },
                        hovertemplate: '%{label}<br>Responses: %{value}<br>Avg Score: %{marker.color:.2f}<extra></extra>'
                    };
                    const layout = { height: 420, margin:{t:10,l:0,r:0,b:0} };
                    Plotly.react(treemapEl, [trace], layout, {displayModeBar:false});
                } else {
                    // Simple fallback
                    treemapEl.style.position = 'relative'; treemapEl.style.height = '400px';
                    if (!data || data.length === 0) return;
                    const maxCount = Math.max(...data.map(d => d.count));
                    data.slice(0, 30).forEach((item, idx) => {
                        const div = document.createElement('div');
                        const size = Math.max(40, Math.sqrt(item.count / maxCount) * 120);
                        div.style.position = 'absolute';
                        div.style.width = `${size}px`; div.style.height = `${size}px`;
                        div.style.left = `${(idx % 6) * 130}px`; div.style.top = `${Math.floor(idx / 6) * 130}px`;
                        div.style.backgroundColor = this.getColorByDomain(item.domain);
                        div.style.color = '#fff'; div.style.display='flex'; div.style.alignItems='center'; div.style.justifyContent='center';
                        div.style.borderRadius='6px'; div.style.boxShadow='0 1px 4px rgba(0,0,0,0.2)';
                        div.title = `${item.domain} • ${item.organization} • ${item.department}\nResponses: ${item.count}\nAvg Score: ${item.avg_culture_score}`;
                        const label = (item.department || '').toString();
                        const short = label.length > 14 ? label.slice(0, 11) + '…' : label;
                        div.innerHTML = `${short}<br/><small>${item.count}</small>`;
                        treemapEl.appendChild(div);
                    });
                }
            } else if (type === 'sunburst') {
                treemapEl.style.display = 'none';
                if (sunburstWrap) sunburstWrap.style.display = '';
                const params = new URLSearchParams({ min_count: String(minResp), domain: dom, _: String(Date.now()) });
                const data = await this.fetchWithCache(`/api/advanced/sunburst?${params.toString()}`);
                const sb = document.getElementById('sunburstPlot');
                if (window.Plotly && sb && Array.isArray(data)) {
                    const labels = data.map(d=>`${d.domain} / ${d.position_level} / ${d.department}`);
                    const parents = data.map(d=>`${d.domain} / ${d.position_level}`);
                    const values = data.map(d=>d.count);
                    const colors = data.map(d=>d.avg_culture_score);
                    const trace = { type:'sunburst', labels, parents, values, marker:{colors, colorscale:'RdBu', reversescale:true}, branchvalues:'total', hovertemplate:'%{label}<br>Responses: %{value}<br>Avg Score: %{marker.color:.2f}<extra></extra>' };
                    const layout = { height: 420, margin:{t:10,l:0,r:0,b:0} };
                    Plotly.react(sb, [trace], layout, {displayModeBar:false});
                } else {
                    sb.innerHTML = '<div class="text-muted">Sunburst view requires Plotly. Please connect to internet.</div>';
                }
            } else {
                // dendrogram not implemented; show info prominently
                if (sunburstWrap) sunburstWrap.style.display = 'none';
                treemapEl.style.display = '';
                treemapEl.innerHTML = '<div class="d-flex w-100 h-100 align-items-center justify-content-center text-muted" style="min-height:360px">Dendrogram view not available in this build</div>';
            }

            // Always update ridge-like chart for selected domain
            await this.setupRidgeChart(dom);
        } catch (error) {
            console.error('Failed to setup hierarchical chart:', error);
        }
    }

    async setupRidgeChart(domain) {
        try {
            const ctx = document.getElementById('ridgeChart');
            if (!ctx) return;
            const data = await this.fetchWithCache(`/api/advanced/ridge?domain=${encodeURIComponent(domain || 'all')}&bins=30`);
            if (!data || !data.sections) return;

            if (this.charts.ridgeChart) this.charts.ridgeChart.destroy();

            const labels = data.x.map(v => Number(v).toFixed(2));
            const color = (i)=>`hsl(${(i*55)%360} 70% 50%)`;
            const sections = Object.keys(data.sections);
            const maxY = Math.max(...sections.flatMap(s => data.sections[s].y));
            const offset = maxY * 1.2;
            const datasets = sections.map((name, i) => ({
                label: name,
                data: data.sections[name].y.map(v => v + i*offset),
                borderColor: color(i),
                backgroundColor: color(i)+'33',
                tension: 0.25,
                fill: true
            }));

            this.charts.ridgeChart = new Chart(ctx, {
                type: 'line',
                data: { labels, datasets },
                options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'bottom', labels: { boxWidth: 12 } } }, scales: { y: { display: false } } }
            });
        } catch (error) {
            console.error('Failed to setup ridge chart:', error);
        }
    }

    async setupDistributionChart() {
        try {
            const type = (document.getElementById('distributionType') || {}).value || 'scores';
            const bins = parseInt((document.getElementById('distributionBins') || {}).value || '20', 10);
            const ctx = document.getElementById('distributionChart');
            if (!ctx) return;

            // Clear distribution cache to ensure fresh data with new bin size
            this.clearCacheForPattern('/api/distributions/');

            if (this.charts.distributionChart) this.charts.distributionChart.destroy();

            if (type === 'scores') {
                const dist = await this.fetchWithCache(`/api/distributions/overall?bins=${bins}`);
                const labels = dist.bins.slice(0, -1).map((b,i)=>`${dist.bins[i].toFixed(2)}-${dist.bins[i+1].toFixed(2)}`);
                this.charts.distributionChart = new Chart(ctx, {
                    type: 'bar',
                    data: { labels, datasets: [{ label: 'All Domains', data: dist.overall, backgroundColor: '#2563eb' }] },
                    options: { responsive: true, maintainAspectRatio: false }
                });
            } else if (type === 'responses') {
                const dist = await this.fetchWithCache(`/api/distributions/responses?bins=${bins}`);
                const labels = dist.bins.slice(0, -1).map((b,i)=>`${Math.round(dist.bins[i])}-${Math.round(dist.bins[i+1])}`);
                this.charts.distributionChart = new Chart(ctx, {
                    type: 'bar',
                    data: { labels, datasets: [{ label: 'Organizations by Response Count', data: dist.counts, backgroundColor: '#10b981' }] },
                    options: { responsive: true, maintainAspectRatio: false }
                });
            } else if (type === 'sections') {
                // Get domain filter for sections analysis
                const domainFilter = (document.getElementById('orgDomainFilter') || {}).value || 'all';
                const domainParam = domainFilter && domainFilter !== 'all' ? `?domain=${encodeURIComponent(domainFilter)}` : '';

                const sections = await this.fetchWithCache(`/api/sections${domainParam}`);
                const labels = Object.keys(sections).map(label => label.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()));
                const values = Object.keys(sections).map(k => sections[k]?.overall_score || 0);

                // Color-code based on HSEG risk zones
                const colors = values.map(score => {
                    if (score <= 1.5) return '#dc2626'; // Crisis Zone - Red
                    if (score <= 2.0) return '#ea580c'; // At Risk Zone - Orange
                    if (score <= 2.5) return '#ca8a04'; // Mixed Zone - Yellow
                    if (score <= 3.0) return '#16a34a'; // Safe Zone - Green
                    return '#059669'; // Thriving Zone - Emerald
                });

                this.charts.distributionChart = new Chart(ctx, {
                    type: 'bar',
                    data: {
                        labels,
                        datasets: [{
                            label: `Cultural Risk Assessment ${domainFilter !== 'all' ? `(${domainFilter})` : '(All Domains)'}`,
                            data: values,
                            backgroundColor: colors,
                            borderColor: colors.map(c => c.replace('#', '#').replace(/(..)(..)(..)/, '#$1$2$3dd')),
                            borderWidth: 2
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: {
                                min: 1,
                                max: 4,
                                ticks: { stepSize: 0.5 },
                                title: { display: true, text: 'Cultural Health Score (1=Crisis, 4=Thriving)' }
                            },
                            x: {
                                title: { display: true, text: 'Cultural Assessment Dimensions' }
                            }
                        },
                        plugins: {
                            tooltip: {
                                callbacks: {
                                    afterLabel: function(context) {
                                        const score = context.parsed.y;
                                        if (score <= 1.5) return 'Zone: Crisis - Immediate intervention required';
                                        if (score <= 2.0) return 'Zone: At Risk - Early warning signs present';
                                        if (score <= 2.5) return 'Zone: Mixed - Inconsistent cultural experiences';
                                        if (score <= 3.0) return 'Zone: Safe - Generally healthy environment';
                                        return 'Zone: Thriving - Exemplary cultural practices';
                                    }
                                }
                            }
                        }
                    }
                });
            }
        } catch (error) {
            console.error('Failed to setup distribution chart:', error);
        }
    }

    onFilterChange(filterId, value) {
        // Handle filter changes and refresh appropriate charts
        console.log(`Filter ${filterId} changed to:`, value);

        // Debounce filter changes to avoid too many API calls
        clearTimeout(this.filterTimeout);
        this.filterTimeout = setTimeout(() => {
            this.applyFilters();
        }, 300);
    }

    applyFilters() {
        // Apply all current filter values and refresh charts
        const activeTab = document.querySelector('.tab-pane.show.active');
        if (activeTab) {
            this.loadTabData(activeTab);
        }
    }

    exportChart(chartCard) {
        if (!chartCard) return;

        const canvas = chartCard.querySelector('canvas');
        if (canvas) {
            const chart = Chart.getChart(canvas);
            if (chart) {
                const url = chart.toBase64Image();
                const link = document.createElement('a');
                link.download = `chart-${Date.now()}.png`;
                link.href = url;
                link.click();
            }
        }
    }

    resetFilters(tabPane) {
        if (!tabPane) return;

        // Reset all select elements in the tab pane to their default values
        const selects = tabPane.querySelectorAll('select');
        selects.forEach(select => {
            const defaultOption = select.querySelector('option[selected]');
            if (defaultOption) {
                select.value = defaultOption.value;
            } else {
                select.selectedIndex = 0;
            }
        });

        // Refresh the tab data
        this.loadTabData(tabPane);
    }

    async generateStatisticalInsights() {
        // Generate AI-powered insights for the Insights tab
        const insightsContainer = document.getElementById('statisticalInsights');
        if (!insightsContainer) return;

        insightsContainer.innerHTML = `
            <div class="tab-loading">
                <div class="spinner-border" role="status"></div>
                <p>Generating statistical insights...</p>
            </div>
        `;

        try {
            const stats = await this.fetchWithCache('/api/stats/quick');
            const sections = await this.fetchWithCache('/api/sections');

            const insights = this.calculateInsights(stats, sections);

            insightsContainer.innerHTML = `
                <div class="row g-3">
                    ${insights.map(insight => `
                        <div class="col-md-6">
                            <div class="insight-card">
                                <div class="insight-header">
                                    <i class="${insight.icon} insight-icon"></i>
                                    <h6>${insight.title}</h6>
                                </div>
                                <div class="insight-body">
                                    <p>${insight.description}</p>
                                    <div class="insight-metric">
                                        <span class="metric-value">${insight.value}</span>
                                        <span class="metric-label">${insight.label}</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    `).join('')}
                </div>
            `;
        } catch (error) {
            console.error('Failed to generate insights:', error);
            insightsContainer.innerHTML = `
                <div class="alert alert-warning">
                    <i class="fas fa-exclamation-triangle me-2"></i>
                    Failed to generate insights. Please try again later.
                </div>
            `;
        }
    }

    async setupClusteringChart() {
        try {
            const data = await this.fetchWithCache('/api/advanced/clustering');
            const ctx = document.getElementById('clusteringChart');
            if (!ctx || !data) return;

            if (this.charts.clusteringChart) this.charts.clusteringChart.destroy();

            const labels = (data.elbow || []).map(e => e.k);
            const inertias = (data.elbow || []).map(e => e.inertia);
            this.charts.clusteringChart = new Chart(ctx, {
                type: 'line',
                data: { labels, datasets: [{ label: 'Inertia', data: inertias, borderColor: '#2563eb', backgroundColor: 'transparent' }] },
                options: { responsive: true, maintainAspectRatio: false }
            });
        } catch (error) {
            console.error('Failed to setup clustering chart:', error);
        }
    }

    async renderOrgSummaryCharts(selectedOrg) {
        try {
            const summary = await this.fetchWithCache(`/api/organizations/summary?organization=${encodeURIComponent(selectedOrg)}`);
            const deptCtx = document.getElementById('deptPieChart');
            const posCtx = document.getElementById('positionPieChart');
            if (deptCtx) {
                if (this.charts.deptPieChart) this.charts.deptPieChart.destroy();
                const labels = Object.keys(summary.departments || {});
                const values = labels.map(k => summary.departments[k]);
                this.charts.deptPieChart = new Chart(deptCtx, {
                    type: 'doughnut',
                    data: { labels, datasets: [{ data: values, backgroundColor: labels.map((_,i)=>`hsl(${(i*50)%360} 70% 55%)`) }] },
                    options: { responsive: true, maintainAspectRatio: false }
                });
            }
            if (posCtx) {
                if (this.charts.positionPieChart) this.charts.positionPieChart.destroy();
                const labels = Object.keys(summary.positions || {});
                const values = labels.map(k => summary.positions[k]);
                this.charts.positionPieChart = new Chart(posCtx, {
                    type: 'doughnut',
                    data: { labels, datasets: [{ data: values, backgroundColor: labels.map((_,i)=>`hsl(${(i*65)%360} 70% 55%)`) }] },
                    options: { responsive: true, maintainAspectRatio: false }
                });
            }
        } catch (error) {
            console.error('Failed to render org summary charts:', error);
        }
    }

    calculateInsights(stats, sections) {
        const insights = [];

        // Calculate key insights from the data
        const scores = Object.keys(sections).map(k => {
            const v = sections[k];
            return typeof v === 'number' ? v : (v?.overall_score ?? v?.score ?? 0);
        }).filter(v => !isNaN(v));
        const avgScore = scores.length ? (scores.reduce((a,b)=>a+b,0) / scores.length) : 0;
        const minScore = scores.length ? Math.min(...scores) : 0;
        const maxScore = scores.length ? Math.max(...scores) : 0;

        insights.push({
            icon: 'fas fa-chart-line',
            title: 'Overall Performance',
            description: `The average culture score across all sections is ${avgScore.toFixed(2)}, indicating ${avgScore > 2.5 ? 'above-average' : 'below-average'} workplace culture.`,
            value: avgScore.toFixed(2),
            label: 'Average Score'
        });

        insights.push({
            icon: 'fas fa-exclamation-triangle',
            title: 'Priority Area',
            description: `The lowest performing section has a score of ${minScore.toFixed(2)}, requiring immediate attention and improvement initiatives.`,
            value: minScore.toFixed(2),
            label: 'Lowest Score'
        });

        insights.push({
            icon: 'fas fa-star',
            title: 'Best Performing Area',
            description: `The highest performing section scores ${maxScore.toFixed(2)}, demonstrating strong practices that could be replicated across other areas.`,
            value: maxScore.toFixed(2),
            label: 'Highest Score'
        });

        insights.push({
            icon: 'fas fa-users',
            title: 'Response Coverage',
            description: `With ${stats.total_responses} total responses from ${stats.num_organizations} organizations, we have robust data coverage for analysis.`,
            value: this.formatNumber(stats.total_responses || 0),
            label: 'Total Responses'
        });

        return insights;
    }
}

// Virtual Table Class for handling large datasets
class VirtualTable {
    constructor(container, options) {
        this.container = container;
        this.options = options;
        this.currentPage = 1;
        this.pageSize = 50;
        this.totalPages = 1;
        this.totalItems = 0;
        this.searchTerm = '';
        this.filterValue = 'all';
        this.data = [];
    }

    async initialize() {
        this.render();
        await this.loadData();
    }

    render() {
        this.container.innerHTML = `
            <div class="virtual-table">
                <div class="table-header-row" id="tableHeader">
                    ${this.options.columns.map(col => `
                        <div class="table-cell" style="width: ${col.width}">${col.title}</div>
                    `).join('')}
                </div>
                <div id="tableBody" class="table-body-virtual">
                    <!-- Data rows will be rendered here -->
                </div>
            </div>
            <div class="table-pagination">
                <div class="pagination-info">
                    <span id="paginationInfo">Loading...</span>
                </div>
                <div class="pagination-controls">
                    <button class="btn btn-sm btn-outline-secondary" id="prevPage" disabled>
                        <i class="fas fa-chevron-left"></i>
                    </button>
                    <button class="btn btn-sm btn-outline-secondary" id="nextPage" disabled>
                        <i class="fas fa-chevron-right"></i>
                    </button>
                </div>
            </div>
        `;

        this.setupPaginationEvents();
    }

    setupPaginationEvents() {
        const prevBtn = this.container.querySelector('#prevPage');
        const nextBtn = this.container.querySelector('#nextPage');

        if (prevBtn) {
            prevBtn.addEventListener('click', () => this.previousPage());
        }

        if (nextBtn) {
            nextBtn.addEventListener('click', () => this.nextPage());
        }
    }

    async loadData() {
        try {
            const result = await this.options.onLoadData(
                this.currentPage,
                this.pageSize,
                this.searchTerm,
                this.filterValue
            );

            this.data = result.data || [];
            this.totalPages = result.pages || 1;
            this.totalItems = result.total || 0;

            this.renderRows();
            this.updatePagination();
        } catch (error) {
            console.error('Failed to load table data:', error);
        }
    }

    renderRows() {
        const tableBody = this.container.querySelector('#tableBody');
        if (!tableBody) return;

        tableBody.innerHTML = this.data.map(row => `
            <div class="table-row">
                ${this.options.columns.map(col => `
                    <div class="table-cell" style="width: ${col.width}">
                        ${this.formatCellValue(row[col.key], col.key)}
                    </div>
                `).join('')}
            </div>
        `).join('');
    }

    formatCellValue(value, key) {
        if (value === null || value === undefined) return '-';

        if (key === 'culture_score') {
            return `<span class="badge bg-primary">${value}</span>`;
        }

        if (key === 'submission_date') {
            return new Date(value).toLocaleDateString();
        }

        return value.toString();
    }

    updatePagination() {
        const paginationInfo = this.container.querySelector('#paginationInfo');
        const prevBtn = this.container.querySelector('#prevPage');
        const nextBtn = this.container.querySelector('#nextPage');

        if (paginationInfo) {
            const start = (this.currentPage - 1) * this.pageSize + 1;
            const end = Math.min(this.currentPage * this.pageSize, this.totalItems);
            paginationInfo.textContent = `${start}-${end} of ${this.totalItems} records`;
        }

        if (prevBtn) {
            prevBtn.disabled = this.currentPage <= 1;
        }

        if (nextBtn) {
            nextBtn.disabled = this.currentPage >= this.totalPages;
        }
    }

    async previousPage() {
        if (this.currentPage > 1) {
            this.currentPage--;
            await this.loadData();
        }
    }

    async nextPage() {
        if (this.currentPage < this.totalPages) {
            this.currentPage++;
            await this.loadData();
        }
    }

    async search(searchTerm) {
        this.searchTerm = searchTerm;
        this.currentPage = 1;
        await this.loadData();
    }

    async filter(filterValue) {
        this.filterValue = filterValue;
        this.currentPage = 1;
        await this.loadData();
    }
}

// Initialize the application when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.dashboardApp = new DashboardApp();
});

// Attach advanced correlations handler to prototype (outside class body)
DashboardApp.prototype.setupAdvancedCorrelations = async function() {
    const level = (document.getElementById('correlationLevel')||{}).value || 'sections';
    try {
        if (level === 'questions') {
            const data = await this.fetchWithCache('/api/correlations');
            this.renderCorrelationHeatmap(data);
        } else {
            await this.setupSectionCorrelationMatrix();
        }
    } catch (e) {
        console.error('Failed to setup advanced correlations:', e);
    }
};
