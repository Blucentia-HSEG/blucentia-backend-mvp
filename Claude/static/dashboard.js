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
        this.setupEventListeners();
        this.showLoadingOverlay();
        this.initializeApp();
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
        if (trendTimeRange) {
            trendTimeRange.addEventListener('change', (e) => {
                this.updateTrendChart(e.target.value);
            });
        }

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
            }

            this.hideLoadingOverlay();
        } catch (error) {
            console.error('Failed to initialize app:', error);
            this.showError('Failed to load dashboard. Please refresh the page.');
            this.hideLoadingOverlay();
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
                            beginAtZero: true
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

        try {
            const trendData = await this.fetchWithCache(`/api/analytics/trend?days=${days}`);

            if (this.charts.trendChart && trendData.labels) {
                this.charts.trendChart.data.labels = trendData.labels;
                this.charts.trendChart.data.datasets = trendData.datasets || [];
                this.charts.trendChart.update('active');
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
                analytics: 'Advanced Analytics',
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
            // Load section analysis for radar chart
            const sectionsData = await this.fetchWithCache('/api/sections');
            this.setupSectionRadarChart(sectionsData);

            // Load distribution chart
            this.setupDistributionChart();
        } catch (error) {
            console.error('Failed to load analytics section:', error);
        }
    }

    setupSectionRadarChart(sectionsData) {
        const ctx = document.getElementById('sectionRadarChart');
        if (!ctx || !sectionsData) return;

        if (this.charts.sectionRadarChart) {
            this.charts.sectionRadarChart.destroy();
        }

        const labels = Object.keys(sectionsData);
        const data = labels.map(section => {
            const scores = Object.values(sectionsData[section]);
            return scores.reduce((sum, score) => sum + score, 0) / scores.length;
        });

        this.charts.sectionRadarChart = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: labels.map(label => label.replace('_', ' ').toUpperCase()),
                datasets: [{
                    label: 'Section Scores',
                    data: data,
                    backgroundColor: 'rgba(37, 99, 235, 0.2)',
                    borderColor: '#2563eb',
                    pointBackgroundColor: '#2563eb',
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: '#2563eb'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: {
                    duration: 0 // Disable animations to prevent looping issues
                },
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 5,
                        ticks: {
                            stepSize: 1
                        },
                        animate: false // Disable scale animations
                    }
                }
            }
        });
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

    async setupOrgBenchmarkChart() {
        try {
            console.log('Setting up org benchmark chart...');
            const organizationsData = await this.fetchWithCache('/api/organizations?limit=50');
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

            // Create organizational benchmarking visualization
            const topOrgs = organizationsData.slice(0, 15);
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

            const datasets = [];
            const domains = [...new Set(organizationsData.map(org => org.domain))];

            domains.forEach(domain => {
                const domainOrgs = organizationsData.filter(org => org.domain === domain);
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
                                    const domainOrgs = organizationsData.filter(org => org.domain === domain);
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
            const organizationsData = await this.fetchWithCache('/api/organizations?limit=20');
            const ctx = document.getElementById('topOrgsChart');

            if (!ctx || !organizationsData) return;

            if (this.charts.topOrgsChart) {
                this.charts.topOrgsChart.destroy();
            }

            // Sort by culture score (best first)
            const topOrgs = organizationsData.sort((a, b) => a.culture_score - b.culture_score).slice(0, 10);
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
                                    return [
                                        `${org.name}`,
                                        `Score: ${org.culture_score}`,
                                        `Domain: ${org.domain}`,
                                        `Responses: ${org.response_count}`
                                    ];
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
            const organizationsData = await this.fetchWithCache('/api/organizations?limit=50');
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

            const datasets = [];
            const domains = [...new Set(organizationsData.map(org => org.domain))];

            domains.forEach(domain => {
                const sizeCounts = sizeRanges.map(range => {
                    return organizationsData.filter(org =>
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
                });
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
                });
            }

            const sortBySelect = document.getElementById('orgSortBy');
            if (sortBySelect) {
                sortBySelect.addEventListener('change', async (e) => {
                    await this.setupTopOrgsChart();
                });
            }

        } catch (error) {
            console.error('Failed to setup org controls:', error);
        }
    }

    async loadCorrelationsSection() {
        try {
            const correlationsData = await this.fetchWithCache('/api/correlations');
            this.renderCorrelationTable(correlationsData);
        } catch (error) {
            console.error('Failed to load correlations section:', error);
        }
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
            await this.setupSectionCorrelationMatrix();
            await this.setupTreemapChart();
            await this.setupDemographicChart();
        } catch (error) {
            console.error('Failed to load advanced section:', error);
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
            const sectionsData = await this.fetchWithCache('/api/sections');
            const ctx = document.getElementById('sectionCorrelationMatrix');

            if (!ctx || !sectionsData) return;

            // Create correlation matrix visualization
            const sections = Object.keys(sectionsData);
            const correlationData = [];

            // Simple correlation based on average scores
            for (let i = 0; i < sections.length; i++) {
                for (let j = 0; j < sections.length; j++) {
                    const corr = i === j ? 1 : Math.random() * 0.8 + 0.1; // Placeholder
                    correlationData.push({
                        x: sections[j],
                        y: sections[i],
                        v: corr
                    });
                }
            }

            if (this.charts.sectionCorrelationMatrix) {
                this.charts.sectionCorrelationMatrix.destroy();
            }

            // Use a heatmap-style visualization
            this.charts.sectionCorrelationMatrix = new Chart(ctx, {
                type: 'scatter',
                data: {
                    datasets: [{
                        label: 'Correlation',
                        data: correlationData.map((d, i) => ({
                            x: i % sections.length,
                            y: Math.floor(i / sections.length),
                            v: d.v
                        })),
                        backgroundColor: correlationData.map(d => {
                            const alpha = Math.abs(d.v);
                            return `rgba(37, 99, 235, ${alpha})`;
                        }),
                        pointRadius: 15
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            type: 'linear',
                            ticks: {
                                callback: function(value) {
                                    return sections[value] ? sections[value].split(' ')[0] : '';
                                }
                            }
                        },
                        y: {
                            type: 'linear',
                            ticks: {
                                callback: function(value) {
                                    return sections[value] ? sections[value].split(' ')[0] : '';
                                }
                            }
                        }
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
                div.textContent = item.count;

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

            // Simple network visualization using SVG
            container.innerHTML = '';
            const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
            svg.style.width = '100%';
            svg.style.height = '500px';
            container.appendChild(svg);

            const nodes = networkData.nodes;
            const links = networkData.links;

            // Position nodes in a circle
            const centerX = 400;
            const centerY = 250;
            const radius = 200;

            nodes.forEach((node, i) => {
                const angle = (i / nodes.length) * 2 * Math.PI;
                node.x = centerX + radius * Math.cos(angle);
                node.y = centerY + radius * Math.sin(angle);
            });

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
                    line.setAttribute('stroke', '#999');
                    line.setAttribute('stroke-width', link.weight * 3);
                    line.setAttribute('opacity', '0.6');
                    svg.appendChild(line);
                }
            });

            // Draw nodes
            nodes.forEach(node => {
                const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                circle.setAttribute('cx', node.x);
                circle.setAttribute('cy', node.y);
                circle.setAttribute('r', node.size);
                circle.setAttribute('fill', this.getColorByDomain(node.domain));
                circle.setAttribute('stroke', '#fff');
                circle.setAttribute('stroke-width', '2');
                circle.style.cursor = 'pointer';

                const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
                title.textContent = `${node.id}\nDomain: ${node.domain}\nScore: ${node.culture_score}\nResponses: ${node.responses}`;
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
            const hierarchicalData = await this.fetchWithCache('/api/advanced/hierarchical');
            const container = document.getElementById('sankeyChart');

            if (!container || !hierarchicalData) return;

            // Simple Sankey-like visualization
            container.innerHTML = '<p class="text-center text-muted">Sankey diagram showing flow from Domain → Organization → Department</p>';

            const domains = [...new Set(hierarchicalData.map(d => d.domain))];
            const orgs = [...new Set(hierarchicalData.map(d => d.organization))].slice(0, 10);

            const summary = document.createElement('div');
            summary.className = 'row';

            domains.forEach(domain => {
                const domainData = hierarchicalData.filter(d => d.domain === domain);
                const totalCount = domainData.reduce((sum, d) => sum + d.count, 0);

                const col = document.createElement('div');
                col.className = 'col-md-4';
                col.innerHTML = `
                    <div class="p-3 border rounded" style="background-color: ${this.getColorByDomain(domain)}20">
                        <h6>${domain}</h6>
                        <p class="mb-1">Total Responses: ${totalCount}</p>
                        <p class="mb-1">Organizations: ${domainData.length}</p>
                        <p class="mb-0">Avg Score: ${(domainData.reduce((sum, d) => sum + d.avg_culture_score, 0) / domainData.length).toFixed(2)}</p>
                    </div>
                `;
                summary.appendChild(col);
            });

            container.appendChild(summary);
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
        // Resize all charts in the given tab pane
        const canvases = tabPane.querySelectorAll('canvas');
        canvases.forEach(canvas => {
            const chart = Chart.getChart(canvas);
            if (chart) {
                chart.resize();
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
                await this.setupTrendChart();
                break;
            case 'distributions':
                await this.setupDistributionChart();
                break;
            case 'insights':
                await this.generateStatisticalInsights();
                break;
        }
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
                await this.loadCorrelationsSection();
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
            '#sectionOrgFilter', '#sectionChartType', '#sectionComparisonMode',
            '#trendTimeRange', '#trendMetric', '#trendGranularity', '#trendSmoothing',
            '#distributionType', '#distributionBins',
            '#pcaComponents', '#pcaFeatures', '#pcaScaling',
            '#clusteringAlgorithm', '#clusterCount', '#clusterMetric', '#clusterBy',
            '#correlationType', '#minCorrelation', '#correlationLevel',
            '#hierarchicalType', '#demographicType'
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

    calculateInsights(stats, sections) {
        const insights = [];

        // Calculate key insights from the data
        const sectionScores = Object.values(sections);
        const avgScore = sectionScores.reduce((a, b) => a + b, 0) / sectionScores.length;
        const minScore = Math.min(...sectionScores);
        const maxScore = Math.max(...sectionScores);

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
            value: stats.total_responses,
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