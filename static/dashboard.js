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

        // Register Chart.js zoom plugin when available
        this.registerZoomPlugin();

        this.init();
    }

    registerZoomPlugin() {
        // Wait for Chart.js and zoom plugin to load
        const checkAndRegister = () => {
            if (typeof Chart !== 'undefined' && typeof Chart.register === 'function' &&
                typeof zoomPlugin !== 'undefined') {
                try {
                    Chart.register(zoomPlugin);
                    console.log('Chart.js zoom plugin registered successfully');
                } catch (error) {
                    console.warn('Failed to register zoom plugin:', error);
                }
            } else {
                setTimeout(checkAndRegister, 100);
            }
        };
        checkAndRegister();
    }

    getZoomOptions(chartType = 'default') {
        // Default zoom configuration for Chart.js charts
        return {
            zoom: {
                wheel: {
                    enabled: true,
                    modifierKey: 'ctrl'
                },
                pinch: {
                    enabled: true
                },
                mode: chartType === 'scatter' || chartType === 'line' ? 'xy' : 'x',
                scaleMode: 'xy'
            },
            pan: {
                enabled: true,
                mode: chartType === 'scatter' || chartType === 'line' ? 'xy' : 'x',
                modifierKey: 'shift'
            }
        };
    }

    enhanceChartWithZoom(chartOptions, chartType = 'default') {
        // Add zoom functionality to any Chart.js chart options
        if (!chartOptions.plugins) chartOptions.plugins = {};
        if (!chartOptions.plugins.zoom) {
            chartOptions.plugins.zoom = this.getZoomOptions(chartType);
        }

        // Add zoom reset button to legend
        if (!chartOptions.plugins.legend) chartOptions.plugins.legend = {};
        if (!chartOptions.plugins.legend.onClick) {
            const originalOnClick = chartOptions.plugins.legend.onClick;
            chartOptions.plugins.legend.onClick = function(e, legendItem, legend) {
                if (e.ctrlKey) {
                    // Ctrl+click legend to reset zoom
                    legend.chart.resetZoom();
                    return;
                }
                if (originalOnClick) originalOnClick.call(this, e, legendItem, legend);
            };
        }

        return chartOptions;
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
        // Sidebar navigation (only bind to sidebar links that have data-section)
        const sidebarSectionLinks = document.querySelectorAll('.sidebar .nav-link[data-section]');
        sidebarSectionLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                const section = link.dataset.section;
                if (!section) return; // ignore non-section links (e.g., tab buttons)
                e.preventDefault();
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
            // COMPREHENSIVE ANIMATION DISABLING - NO ANIMATIONS ANYWHERE
            Chart.defaults.animation = false;
            Chart.defaults.animations = false;
            Chart.defaults.animation = { duration: 0 };
            Chart.defaults.responsiveAnimationDuration = 0;

            // Disable all transition animations
            Chart.defaults.transitions = {
                active: { animation: { duration: 0 } },
                show: { animation: { duration: 0 } },
                hide: { animation: { duration: 0 } },
                resize: { animation: { duration: 0 } }
            };

            // Disable hover animations completely
            Chart.defaults.interaction = {
                intersect: false,
                mode: 'index',
                animationDuration: 0
            };
            Chart.defaults.interaction.animateOnHover = false;

            // Disable tooltip animations
            Chart.defaults.plugins = Chart.defaults.plugins || {};
            Chart.defaults.plugins.tooltip = Chart.defaults.plugins.tooltip || {};
            Chart.defaults.plugins.tooltip.animation = { duration: 0 };

            // Disable legend animations
            Chart.defaults.plugins.legend = Chart.defaults.plugins.legend || {};
            Chart.defaults.plugins.legend.animation = { duration: 0 };

            // Disable ALL element animations
            Chart.defaults.elements = Chart.defaults.elements || {};
            Chart.defaults.elements.point = Chart.defaults.elements.point || {};
            Chart.defaults.elements.point.hoverRadius = Chart.defaults.elements.point.radius;
            Chart.defaults.elements.line = Chart.defaults.elements.line || {};
            Chart.defaults.elements.bar = Chart.defaults.elements.bar || {};

            // Disable scale animations
            Chart.defaults.scales = Chart.defaults.scales || {};
            Chart.defaults.scales.x = Chart.defaults.scales.x || {};
            Chart.defaults.scales.y = Chart.defaults.scales.y || {};
            Chart.defaults.scales.x.animation = { duration: 0 };
            Chart.defaults.scales.y.animation = { duration: 0 };

            Chart.defaults.responsive = true;
        } catch (e) {
            console.warn('Unable to set Chart.js global defaults:', e);
        }
    }

    // Helper function to ensure NO animations and NO zoom in any chart
    getNoAnimationConfig() {
        return {
            animation: false,
            animations: false,
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 0 },
            transitions: {
                active: { animation: { duration: 0 } },
                show: { animation: { duration: 0 } },
                hide: { animation: { duration: 0 } },
                resize: { animation: { duration: 0 } }
            },
            interaction: {
                intersect: false,
                mode: 'index',
                animationDuration: 0
            },
            plugins: {
                tooltip: {
                    animation: { duration: 0 }
                },
                legend: {
                    animation: { duration: 0 }
                },
                zoom: {
                    zoom: {
                        wheel: {
                            enabled: false
                        },
                        pinch: {
                            enabled: false
                        },
                        drag: {
                            enabled: false
                        },
                        mode: 'xy',
                        rangeMin: {
                            x: null,
                            y: null
                        },
                        rangeMax: {
                            x: null,
                            y: null
                        }
                    },
                    pan: {
                        enabled: false,
                        mode: 'xy',
                        rangeMin: {
                            x: null,
                            y: null
                        },
                        rangeMax: {
                            x: null,
                            y: null
                        }
                    }
                }
            },
            scales: {
                x: {
                    animation: { duration: 0 }
                },
                y: {
                    animation: { duration: 0 }
                }
            }
        };
    }

    // Force disable animations and zoom on any chart after creation
    disableChartAnimations(chart) {
        if (!chart) return;

        try {
            // Disable all possible animation properties
            chart.options.animation = false;
            chart.options.animations = false;
            chart.options.animation = { duration: 0 };
            chart.options.responsiveAnimationDuration = 0;

            if (chart.options.transitions) {
                chart.options.transitions.active = { animation: { duration: 0 } };
                chart.options.transitions.show = { animation: { duration: 0 } };
                chart.options.transitions.hide = { animation: { duration: 0 } };
                chart.options.transitions.resize = { animation: { duration: 0 } };
            }

            // Disable zoom functionality completely
            if (!chart.options.plugins) chart.options.plugins = {};
            chart.options.plugins.zoom = {
                zoom: { enabled: false },
                pan: { enabled: false }
            };

            // Update the chart to apply changes
            chart.update('none'); // 'none' mode = no animation
        } catch (e) {
            console.warn('Failed to disable animations/zoom on chart:', e);
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

            // Create chart with zoom functionality
            const chartOptions = {
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
                        borderWidth: 1,
                        callbacks: {
                            afterLabel: function(context) {
                                const score = context.parsed.y;
                                if (score >= 7 && score <= 12) return 'Tier: Crisis 🔴';
                                if (score >= 13 && score <= 16) return 'Tier: At Risk 🟠';
                                if (score >= 17 && score <= 20) return 'Tier: Mixed ⚫';
                                if (score >= 21 && score <= 24) return 'Tier: Safe 🔵';
                                if (score >= 25 && score <= 28) return 'Tier: Thriving 🟢';
                                return '';
                            }
                        }
                    },
                    // Add horizontal lines for tier boundaries
                    annotation: {
                        annotations: {
                            crisis: {
                                type: 'line',
                                yMin: 12,
                                yMax: 12,
                                borderColor: '#ef4444',
                                borderWidth: 1,
                                borderDash: [5, 5],
                                label: {
                                    display: true,
                                    content: 'Crisis',
                                    position: 'end'
                                }
                            },
                            atRisk: {
                                type: 'line',
                                yMin: 16,
                                yMax: 16,
                                borderColor: '#f97316',
                                borderWidth: 1,
                                borderDash: [5, 5],
                                label: {
                                    display: true,
                                    content: 'At Risk',
                                    position: 'end'
                                }
                            },
                            mixed: {
                                type: 'line',
                                yMin: 20,
                                yMax: 20,
                                borderColor: '#6b7280',
                                borderWidth: 1,
                                borderDash: [5, 5],
                                label: {
                                    display: true,
                                    content: 'Mixed',
                                    position: 'end'
                                }
                            },
                            safe: {
                                type: 'line',
                                yMin: 24,
                                yMax: 24,
                                borderColor: '#3b82f6',
                                borderWidth: 1,
                                borderDash: [5, 5],
                                label: {
                                    display: true,
                                    content: 'Safe',
                                    position: 'end'
                                }
                            }
                        }
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
                        min: 7,
                        max: 28,
                        ticks: {
                            stepSize: 3,
                            callback: function(value, index, ticks) {
                                // Add HSEG tier indicators
                                if (value === 12) return value + ' (Crisis)';
                                if (value === 16) return value + ' (At Risk)';
                                if (value === 20) return value + ' (Mixed)';
                                if (value === 24) return value + ' (Safe)';
                                if (value === 28) return value + ' (Thriving)';
                                return value;
                            }
                        }
                    }
                },
                elements: {
                    point: {
                        radius: 4,
                        hoverRadius: 6
                    }
                }
            };

            // Enhance with zoom functionality
            this.enhanceChartWithZoom(chartOptions, 'line');

            this.charts.trendChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: trendData.labels,
                    datasets: trendData.datasets || []
                },
                options: chartOptions
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
                    // Update for HSEG 28-point scale
                    this.charts.trendChart.options.scales.y.min = 7;
                    this.charts.trendChart.options.scales.y.max = 28;
                    this.charts.trendChart.options.scales.y.ticks = { stepSize: 3 };
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
        const activeNavLink = document.querySelector(`[data-section="${section}"]`);
        if (activeNavLink) {
            activeNavLink.classList.add('active');
        }

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
        const scoringMode = (document.getElementById('sectionScoringMode') || {}).value || 'raw';
        const selectedOrg = (document.getElementById('sectionOrgFilter') || {}).value || 'all';
        const multiSel = document.getElementById('sectionOrgMulti');

        // Clear cache for organization sections when switching comparison modes or organizations
        this.clearCacheForPattern('/api/organizations/sections');

        // Prepare labels with category weights for HSEG mode
        const labels = Object.keys(sectionsData).map(label => {
            const sectionData = sectionsData[label];
            if (scoringMode === 'weighted' && sectionData.category_weight) {
                return `${label.replace('_', ' ').toUpperCase()}\n(Weight: ${sectionData.category_weight}x)`;
            } else if (scoringMode === 'percentage') {
                return `${label.replace('_', ' ').toUpperCase()}\n(% of Max)`;
            }
            return label.replace('_', ' ').toUpperCase();
        });

        // Get values based on scoring mode
        const overallValues = Object.keys(sectionsData).map(k => {
            const sectionData = sectionsData[k];
            if (scoringMode === 'weighted') {
                return sectionData.weighted_score || 0;
            } else if (scoringMode === 'percentage') {
                return sectionData.weighted_percentage || 0;
            } else {
                return sectionData.overall_score || 0;
            }
        });

        // Helper function to get organization values based on scoring mode
        const getOrgValues = (orgSections) => {
            return Object.keys(sectionsData).map(k => {
                const orgData = orgSections[k];
                if (!orgData) return 0;

                if (scoringMode === 'weighted') {
                    return orgData.weighted_score || orgData.score || 0;
                } else if (scoringMode === 'percentage') {
                    // Calculate percentage for this org's data
                    const sectionInfo = sectionsData[k];
                    if (sectionInfo && sectionInfo.category_weight) {
                        const rawScore = orgData.score || orgData.overall_score || 0;
                        const maxWeightedForCategory = 4 * sectionInfo.category_weight;
                        const orgWeightedScore = rawScore * sectionInfo.category_weight;
                        return (orgWeightedScore / maxWeightedForCategory) * 100;
                    }
                    return 0;
                } else {
                    return orgData.score || orgData.overall_score || 0;
                }
            });
        };

        const datasets = [];
        if (comparison === 'single' && selectedOrg !== 'all') {
            const orgSec = await this.fetchWithCache(`/api/organizations/sections?organization=${encodeURIComponent(selectedOrg)}`);
            const orgVals = getOrgValues(orgSec);
            datasets.push({ label: selectedOrg, data: orgVals, backgroundColor: 'rgba(37,99,235,0.2)', borderColor: '#2563eb' });
        } else if (comparison === 'single' || (comparison !== 'multi' && selectedOrg === 'all')) {
            datasets.push({ label: 'Overall', data: overallValues, backgroundColor: 'rgba(37, 99, 235, 0.2)', borderColor: '#2563eb' });
        } else if (comparison === 'benchmark') {
            datasets.push({ label: 'Overall', data: overallValues, backgroundColor: 'rgba(148,163,184,0.2)', borderColor: '#94a3b8' });
        }

        // If benchmark mode and a specific org selected, overlay org values
        if (comparison === 'benchmark' && selectedOrg && selectedOrg !== 'all') {
            const orgSec = await this.fetchWithCache(`/api/organizations/sections?organization=${encodeURIComponent(selectedOrg)}`);
            const orgVals = getOrgValues(orgSec);
            datasets.push({ label: selectedOrg, data: orgVals, backgroundColor: 'rgba(16,185,129,0.2)', borderColor: '#10b981' });
        }

        // Multi-organization comparison
        if (comparison === 'multi' && multiSel) {
            const chosen = Array.from(multiSel.selectedOptions || []).map(o => o.value).slice(0, 5);
            const palette = ['#2563eb','#10b981','#f59e0b','#ef4444','#8b5cf6'];
            for (let i=0;i<chosen.length;i++) {
                const org = chosen[i];
                const orgSec = await this.fetchWithCache(`/api/organizations/sections?organization=${encodeURIComponent(org)}`);
                const orgVals = getOrgValues(orgSec);
                datasets.push({ label: org, data: orgVals, backgroundColor: palette[i]+'33', borderColor: palette[i] });
            }
        }

        const commonOptions = {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            plugins: {
                title: {
                    display: true,
                    text: scoringMode === 'raw' ? 'Raw Question Averages (1-4 scale)' :
                          scoringMode === 'weighted' ? 'HSEG Weighted Scores (Risk-Proportional)' :
                          'Category Performance (% of Maximum)',
                    font: { size: 14, weight: 'bold' }
                }
            }
        };

        // Determine scale based on scoring mode
        let scaleOptions;
        if (scoringMode === 'weighted') {
            const maxValues = Object.values(sectionsData).map(s => s.weighted_score || 0);
            const maxVal = Math.max(...maxValues);
            scaleOptions = {
                min: 0,
                max: Math.ceil(maxVal * 1.1),
                ticks: { stepSize: Math.ceil(maxVal / 8) }
            };
        } else if (scoringMode === 'percentage') {
            scaleOptions = {
                min: 0,
                max: 100,
                ticks: { stepSize: 10, callback: function(value) { return value + '%'; } }
            };
        } else {
            scaleOptions = { min: 1, max: 4, ticks: { stepSize: 0.5 } };
        }

        if (chartType === 'bar') {
            this.charts.sectionRadarChart = new Chart(ctx, {
                type: 'bar',
                data: { labels, datasets },
                options: { ...commonOptions, scales: { y: scaleOptions } }
            });
        } else if (chartType === 'horizontal') {
            this.charts.sectionRadarChart = new Chart(ctx, {
                type: 'bar',
                data: { labels, datasets },
                options: { ...commonOptions, indexAxis: 'y', scales: { x: scaleOptions } }
            });
        } else {
            this.charts.sectionRadarChart = new Chart(ctx, {
                type: 'radar',
                data: { labels, datasets },
                options: { ...commonOptions, scales: { r: { ...scaleOptions, animate: false } }, plugins: { ...commonOptions.plugins, legend: { display: true } } }
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

    // Performance tab specific filters
    getPerformanceFilters() {
        const sortBy = (document.getElementById('orgSortBy') || {}).value || 'culture_score';
        const timeRange = (document.getElementById('performanceTimeRange') || {}).value || 'all';
        const topCount = parseInt((document.getElementById('topOrgCount') || {}).value || '20', 10);
        return { sortBy, timeRange, topCount };
    }

    // Distribution tab specific filters
    getDistributionFilters() {
        const groupBy = (document.getElementById('distributionGroupBy') || {}).value || 'domain';
        const metric = (document.getElementById('distributionMetric') || {}).value || 'count';
        return { groupBy, metric };
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
                // Use HSEG tier boundaries: Crisis (7-12), At Risk (13-16), Mixed (17-20), Safe (21-24), Thriving (25-28)
                if (scoreRange === 'high' && !(org.culture_score >= 21)) return false; // Safe + Thriving
                if (scoreRange === 'medium' && !(org.culture_score >= 17 && org.culture_score < 21)) return false; // Mixed
                if (scoreRange === 'low' && !(org.culture_score < 17)) return false; // Crisis + At Risk
            }
            return true;
        });
    }

    async setupOrgBenchmarkChart() {
        try {
            console.log('Setting up org benchmark chart...');
            // Get current filter values
            const filters = this.getOrganizationFilters();
            const params = new URLSearchParams({
                limit: '100',
                min_responses: filters.minResponses.toString(),
                domain: filters.domain,
                score_range: filters.scoreRange,
                size_filter: filters.sizeFilter
            });

            const organizationsData = await this.fetchWithCache(`/api/organizations?${params.toString()}`);
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
                        backgroundColor: cultureScores.map(score => {
                            // HSEG tier-based color coding
                            if (score >= 25) return '#22c55e'; // Thriving - Green
                            if (score >= 21) return '#3b82f6'; // Safe - Blue
                            if (score >= 17) return '#6b7280'; // Mixed - Gray
                            if (score >= 13) return '#f59e0b'; // At Risk - Orange
                            return '#ef4444'; // Crisis - Red
                        }),
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
            // Check if we're in performance tab
            const perfFilters = this.getPerformanceFilters();
            const params = new URLSearchParams({
                limit: perfFilters.topCount.toString(),
                min_responses: '1'
            });

            const organizationsData = await this.fetchWithCache(`/api/organizations?${params.toString()}`);
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
                        r: Math.sqrt(org.employee_count || 1) / 50 + 5
                    })),
                    backgroundColor: domainColors[domain] || '#999',
                    borderColor: domainColors[domain] || '#999'
                });
            });

            // Create chart with enhanced zoom functionality
            const scatterOptions = {
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
            };

            // Enhance with zoom functionality for scatter charts
            this.enhanceChartWithZoom(scatterOptions, 'scatter');

            this.charts.orgScatterChart = new Chart(ctx, {
                type: 'bubble',
                data: { datasets },
                options: scatterOptions
            });
        } catch (error) {
            console.error('Failed to setup org scatter chart:', error);
        }
    }

    async setupTopOrgsChart() {
        try {
            // Get performance filter values
            const perfFilters = this.getPerformanceFilters();
            const params = new URLSearchParams({
                limit: perfFilters.topCount.toString(),
                min_responses: '1'
            });

            const organizationsData = await this.fetchWithCache(`/api/organizations?${params.toString()}`);
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
                        backgroundColor: scores.map(score => {
                            // HSEG tier-based color coding
                            if (score >= 25) return '#22c55e'; // Thriving - Green
                            if (score >= 21) return '#3b82f6'; // Safe - Blue
                            if (score >= 17) return '#6b7280'; // Mixed - Gray
                            if (score >= 13) return '#f59e0b'; // At Risk - Orange
                            return '#ef4444'; // Crisis - Red
                        }),
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
            // Get distribution filter values
            const distFilters = this.getDistributionFilters();
            const organizationsData = await this.fetchWithCache('/api/organizations?limit=100');
            const ctx = document.getElementById('orgSizeChart');

            if (!ctx || !organizationsData) return;

            if (this.charts.orgSizeChart) {
                this.charts.orgSizeChart.destroy();
            }

            // Group data based on filter
            let groupedData = {};
            organizationsData.forEach(org => {
                let groupKey;
                switch (distFilters.groupBy) {
                    case 'domain':
                        groupKey = org.domain || 'Unknown';
                        break;
                    case 'size':
                        const size = org.employee_count || 0;
                        if (size < 100) groupKey = 'Small (<100)';
                        else if (size < 1000) groupKey = 'Medium (100-1K)';
                        else if (size < 10000) groupKey = 'Large (1K-10K)';
                        else groupKey = 'Enterprise (10K+)';
                        break;
                    case 'location':
                        groupKey = 'United States'; // Placeholder since we don't have location data
                        break;
                    case 'industry':
                        groupKey = org.domain || 'Unknown'; // Use domain as industry proxy
                        break;
                    default:
                        groupKey = org.domain || 'Unknown';
                }

                if (!groupedData[groupKey]) {
                    groupedData[groupKey] = { count: 0, totalScore: 0, totalResponses: 0 };
                }
                groupedData[groupKey].count++;
                groupedData[groupKey].totalScore += org.culture_score || 0;
                groupedData[groupKey].totalResponses += org.response_count || 0;
            });

            // Extract data based on metric
            const labels = Object.keys(groupedData);
            const data = labels.map(label => {
                const group = groupedData[label];
                switch (distFilters.metric) {
                    case 'count':
                        return group.count;
                    case 'avg_score':
                        return group.count > 0 ? group.totalScore / group.count : 0;
                    case 'total_responses':
                        return group.totalResponses;
                    default:
                        return group.count;
                }
            });

            // Create pie/doughnut chart with filtered data
            const backgroundColors = [
                '#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57',
                '#ff9ff3', '#54a0ff', '#5f27cd', '#00d2d3', '#ff9f43'
            ];

            this.charts.orgSizeChart = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: labels,
                    datasets: [{
                        data: data,
                        backgroundColor: backgroundColors.slice(0, labels.length),
                        borderWidth: 2,
                        borderColor: '#ffffff'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: { duration: 0 },
                    plugins: {
                        title: {
                            display: true,
                            text: `Distribution by ${distFilters.groupBy} (${distFilters.metric})`,
                            font: { size: 14 }
                        },
                        legend: {
                            position: 'bottom'
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
                    plugins: {
                        tooltip: {
                            animation: { duration: 0 },
                            callbacks: {
                                label: function(context) {
                                    return context.dataset.label + ': ' + context.parsed.r.toFixed(1) + '%';
                                }
                            }
                        }
                    },
                    scales: {
                        r: {
                            min: 0,
                            max: 100,
                            ticks: {
                                stepSize: 20,
                                callback: function(value) {
                                    return value + '%';
                                }
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

    async updateOrgRadarChart() {
        try {
            const org1 = document.getElementById('radarOrg1')?.value;
            const org2 = document.getElementById('radarOrg2')?.value;
            const org3 = document.getElementById('radarOrg3')?.value;

            if (!this.charts.orgRadarChart) return;

            const datasets = [];
            const colors = ['rgba(255, 99, 132, 0.2)', 'rgba(54, 162, 235, 0.2)', 'rgba(255, 205, 86, 0.2)'];
            const borderColors = ['rgba(255, 99, 132, 1)', 'rgba(54, 162, 235, 1)', 'rgba(255, 205, 86, 1)'];

            const organizations = [org1, org2, org3].filter(org => org && org !== 'all');

            for (let i = 0; i < organizations.length; i++) {
                const orgName = organizations[i];
                try {
                    const orgData = await this.fetchWithCache(`/api/organizations/sections?organization=${encodeURIComponent(orgName)}`);

                    if (orgData) {
                        const sectionOrder = ['Power Abuse Suppression', 'Discrimination Exclusion', 'Manipulative Work Culture', 'Failure of Accountability', 'Mental Health Harm', 'Erosion of Voice Autonomy'];
                        const scores = sectionOrder.map(section => {
                            return orgData[section]?.weighted_percentage || 0;
                        });

                        datasets.push({
                            label: orgName,
                            data: scores,
                            backgroundColor: colors[i],
                            borderColor: borderColors[i],
                            borderWidth: 2
                        });
                    }
                } catch (error) {
                    console.error(`Failed to load data for ${orgName}:`, error);
                }
            }

            this.charts.orgRadarChart.data.datasets = datasets;
            this.charts.orgRadarChart.update();
        } catch (error) {
            console.error('Failed to update org radar chart:', error);
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
                sortBySelect.addEventListener('change', async () => {
                    await this.setupTopOrgsChart();
                    await this.setupOrgScatterChart();
                });
                sortBySelect.dataset.bound = '1';
            }

            const topCountSelect = document.getElementById('topOrgCount');
            if (topCountSelect && !topCountSelect.dataset.bound) {
                topCountSelect.addEventListener('change', async () => {
                    await this.setupTopOrgsChart();
                    await this.setupOrgScatterChart();
                });
                topCountSelect.dataset.bound = '1';
            }

            const timeRangeSelect = document.getElementById('performanceTimeRange');
            if (timeRangeSelect && !timeRangeSelect.dataset.bound) {
                timeRangeSelect.addEventListener('change', async () => {
                    await this.setupTopOrgsChart();
                    await this.setupOrgScatterChart();
                });
                timeRangeSelect.dataset.bound = '1';
            }

            // Distribution filters
            const distributionGroupBy = document.getElementById('distributionGroupBy');
            if (distributionGroupBy && !distributionGroupBy.dataset.bound) {
                distributionGroupBy.addEventListener('change', async () => {
                    await this.setupOrgSizeChart();
                });
                distributionGroupBy.dataset.bound = '1';
            }

            const distributionMetric = document.getElementById('distributionMetric');
            if (distributionMetric && !distributionMetric.dataset.bound) {
                distributionMetric.addEventListener('change', async () => {
                    await this.setupOrgSizeChart();
                });
                distributionMetric.dataset.bound = '1';
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

            // Add event listeners for radar org selectors
            ['radarOrg1', 'radarOrg2', 'radarOrg3'].forEach(selectorId => {
                const selector = document.getElementById(selectorId);
                if (selector && !selector.dataset.radarBound) {
                    selector.addEventListener('change', async () => {
                        await this.updateOrgRadarChart();
                    });
                    selector.dataset.radarBound = '1';
                }
            });

        } catch (error) {
            console.error('Failed to setup org controls:', error);
        }
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

    getScoreColor(score) {
        // Return color based on culture score (0-4 scale typically)
        if (score >= 3.5) return '#16a34a'; // Green - Good
        if (score >= 2.5) return '#eab308'; // Yellow - Average
        if (score >= 1.5) return '#ea580c'; // Orange - Below Average
        return '#dc2626'; // Red - Poor
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
                options: { responsive: true, maintainAspectRatio: false, indexAxis: 'y', animation: { duration: 0 } }
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
                // Use HSEG scale (7-28) for color mapping
                const t = Math.max(0, Math.min(1, (v - 7) / (28 - 7)));
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
            // Get filter values
            const pcaComponents = (document.getElementById('pcaComponents') || {}).value || '2';
            const pcaFeatures = (document.getElementById('pcaFeatures') || {}).value || 'all';
            const pcaScaling = (document.getElementById('pcaScaling') || {}).value || 'standard';

            const clusteringData = await this.fetchWithCache(`/api/advanced/clustering?components=${pcaComponents}&features=${pcaFeatures}&scaling=${pcaScaling}`);
            const ctx = document.getElementById('pcaChart');

            if (!ctx || !clusteringData.pca_data) return;

            if (this.charts.pcaChart) {
                this.charts.pcaChart.destroy();
            }

            // Group data by HSEG tier for risk-based visualization
            const tierColors = {
                'Crisis': '#dc2626',      // Red
                'At Risk': '#ea580c',     // Orange
                'Mixed': '#6b7280',       // Gray
                'Safe': '#16a34a',        // Green
                'Thriving': '#059669'     // Emerald
            };

            const datasets = [];
            const tiers = [...new Set(clusteringData.pca_data.map(d => d.hseg_tier))];

            tiers.forEach(tier => {
                const tierData = clusteringData.pca_data.filter(d => d.hseg_tier === tier);
                datasets.push({
                    label: `${tier} (${tierData.length})`,
                    data: tierData.map(d => ({x: d.x, y: d.y})),
                    backgroundColor: tierColors[tier] || '#999',
                    pointRadius: 4,
                    pointHoverRadius: 6
                });
            });

            this.charts.pcaChart = new Chart(ctx, {
                type: 'scatter',
                data: { datasets },
                options: {
                    ...this.getNoAnimationConfig(),
                    scales: {
                        x: {
                            title: { display: true, text: 'PC1' },
                            animation: { duration: 0 }
                        },
                        y: {
                            title: { display: true, text: 'PC2' },
                            animation: { duration: 0 }
                        }
                    },
                    plugins: {
                        title: {
                            display: true,
                            text: 'PCA Analysis - HSEG Cultural Risk Clustering',
                            animation: { duration: 0 }
                        },
                        subtitle: {
                            display: true,
                            text: `${clusteringData.hseg_context?.total_samples || 0} samples | Avg Score: ${clusteringData.hseg_context?.score_stats?.mean || 'N/A'}`,
                            animation: { duration: 0 }
                        },
                        tooltip: {
                            animation: { duration: 0 },
                            callbacks: {
                                title: function(context) {
                                    const point = clusteringData.pca_data[context[0].dataIndex];
                                    return point.organization;
                                },
                                label: function(context) {
                                    const point = clusteringData.pca_data[context.dataIndex];
                                    return [
                                        `Domain: ${point.domain}`,
                                        `HSEG Score: ${point.hseg_score}`,
                                        `Tier: ${point.hseg_tier}`,
                                        `PC1: ${point.x.toFixed(2)}`,
                                        `PC2: ${point.y.toFixed(2)}`
                                    ];
                                }
                            }
                        },
                        zoom: {
                            zoom: {
                                wheel: { enabled: false },
                                pinch: { enabled: false },
                                drag: { enabled: false }
                            },
                            pan: { enabled: false }
                        }
                    }
                }
            });

            // Force disable any remaining animations
            this.disableChartAnimations(this.charts.pcaChart);
        } catch (error) {
            console.error('Failed to setup PCA chart:', error);
        }
    }

    async setupVarianceChart() {
        try {
            const clusteringData = await this.fetchWithCache('/api/advanced/clustering');
            const ctx = document.getElementById('varianceChart');

            if (!ctx || !clusteringData.hseg_context) return;

            if (this.charts.varianceChart) {
                this.charts.varianceChart.destroy();
            }

            const tierCounts = clusteringData.hseg_context.tier_counts;
            const scoreStats = clusteringData.hseg_context.score_stats;
            const tierColors = {
                'Crisis': '#dc2626',
                'At Risk': '#ea580c',
                'Mixed': '#6b7280',
                'Safe': '#16a34a',
                'Thriving': '#059669'
            };

            this.charts.varianceChart = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: Object.keys(tierCounts).filter(tier => tierCounts[tier] > 0),
                    datasets: [{
                        data: Object.keys(tierCounts).filter(tier => tierCounts[tier] > 0).map(tier => tierCounts[tier]),
                        backgroundColor: Object.keys(tierCounts).filter(tier => tierCounts[tier] > 0).map(tier => tierColors[tier]),
                        borderWidth: 2,
                        borderColor: '#ffffff'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: { duration: 0 },
                    plugins: {
                        title: {
                            display: true,
                            text: 'HSEG Tier Distribution',
                            font: { size: 14, weight: 'bold' }
                        },
                        legend: {
                            position: 'bottom',
                            labels: {
                                usePointStyle: true,
                                font: { size: 11 }
                            }
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    const tier = context.label;
                                    const count = context.parsed;
                                    const total = clusteringData.hseg_context.total_samples;
                                    const percentage = ((count / total) * 100).toFixed(1);
                                    return `${tier}: ${count} (${percentage}%)`;
                                }
                            }
                        }
                    }
                }
            });
        } catch (error) {
            console.error('Failed to setup HSEG tier chart:', error);
        }
    }


    async setupTreemapChart() {
        try {
            // Get filter values
            const minCount = (document.getElementById('hierarchicalMinResponses') || {}).value || 5;
            const domain = (document.getElementById('hierarchicalDomain') || {}).value || 'all';

            const treemapData = await this.fetchWithCache(`/api/advanced/hierarchical?min_count=${minCount}&domain=${domain}`);
            const container = document.getElementById('treemapChart');

            if (!container) return;

            // Clear container
            container.innerHTML = '';

            if (!treemapData || treemapData.length === 0) {
                container.innerHTML = '<div class="d-flex align-items-center justify-content-center h-100 text-muted" style="min-height:300px;">No organizational data available with current filters</div>';
                return;
            }

            // Debug logging
            console.log('Treemap data received:', treemapData.length, 'items');

            // Check if Plotly is available
            if (typeof Plotly !== 'undefined') {
                // Create proper hierarchical structure for treemap
                const hierarchyMap = {};

                // Build hierarchy: Domain -> Organization -> Department
                treemapData.forEach(item => {
                    const domain = item.domain || 'Unknown Domain';
                    const org = item.organization || 'Unknown Organization';
                    const dept = item.department || 'Unknown Department';

                    if (!hierarchyMap[domain]) hierarchyMap[domain] = {};
                    if (!hierarchyMap[domain][org]) hierarchyMap[domain][org] = {};

                    hierarchyMap[domain][org][dept] = {
                        count: item.count || 1,
                        score: item.avg_culture_score || 0
                    };
                });

                // Prepare data for Plotly treemap
                const labels = [];
                const parents = [];
                const values = [];
                const colors = [];

                // Add root domains
                Object.keys(hierarchyMap).forEach(domain => {
                    labels.push(domain);
                    parents.push('');

                    let domainTotal = 0;
                    let domainScoreSum = 0;
                    let domainCount = 0;

                    Object.keys(hierarchyMap[domain]).forEach(org => {
                        Object.keys(hierarchyMap[domain][org]).forEach(dept => {
                            const data = hierarchyMap[domain][org][dept];
                            domainTotal += data.count;
                            domainScoreSum += data.score * data.count;
                            domainCount += data.count;
                        });
                    });

                    values.push(domainTotal);
                    colors.push(domainCount > 0 ? domainScoreSum / domainCount : 0);
                });

                // Add organizations
                Object.keys(hierarchyMap).forEach(domain => {
                    Object.keys(hierarchyMap[domain]).forEach(org => {
                        labels.push(`${org}`);
                        parents.push(domain);

                        let orgTotal = 0;
                        let orgScoreSum = 0;
                        let orgCount = 0;

                        Object.keys(hierarchyMap[domain][org]).forEach(dept => {
                            const data = hierarchyMap[domain][org][dept];
                            orgTotal += data.count;
                            orgScoreSum += data.score * data.count;
                            orgCount += data.count;
                        });

                        values.push(orgTotal);
                        colors.push(orgCount > 0 ? orgScoreSum / orgCount : 0);

                        // Add departments
                        Object.keys(hierarchyMap[domain][org]).forEach(dept => {
                            const data = hierarchyMap[domain][org][dept];
                            labels.push(`${dept}`);
                            parents.push(`${org}`);
                            values.push(data.count);
                            colors.push(data.score);
                        });
                    });
                });

                const trace = {
                    type: 'treemap',
                    labels: labels,
                    parents: parents,
                    values: values,
                    marker: {
                        colors: colors,
                        colorscale: 'RdBu',
                        reversescale: true,
                        colorbar: {
                            title: 'Avg Culture Score',
                            titleside: 'right'
                        },
                        line: { width: 2, color: 'white' }
                    },
                    textinfo: 'label+value',
                    hovertemplate: '%{label}<br>Responses: %{value}<br>Avg Score: %{color:.2f}<extra></extra>'
                };

                const layout = {
                    title: 'Organizational Structure Overview',
                    font: { size: 12 },
                    margin: { t: 50, l: 10, r: 10, b: 10 },
                    height: 400
                };

                const config = {
                    displayModeBar: false,
                    responsive: true
                };

                Plotly.newPlot(container, [trace], layout, config);

            } else {
                // Fallback without Plotly
                container.style.position = 'relative';
                container.style.height = '400px';
                container.style.backgroundColor = '#f8f9fa';
                container.style.border = '1px solid #dee2e6';
                container.style.borderRadius = '6px';

                const maxCount = Math.max(...treemapData.map(d => d.count || 1), 1);

                treemapData.slice(0, 24).forEach((item, idx) => {
                    const div = document.createElement('div');
                    const count = item.count || 1;
                    const ratio = count / maxCount;
                    const size = Math.max(50, Math.sqrt(ratio) * 120);

                    div.style.position = 'absolute';
                    div.style.width = `${size}px`;
                    div.style.height = `${size}px`;
                    div.style.left = `${(idx % 6) * 140 + 10}px`;
                    div.style.top = `${Math.floor(idx / 6) * 140 + 10}px`;
                    div.style.backgroundColor = this.getScoreColor(item.avg_culture_score || 0);
                    div.style.color = '#fff';
                    div.style.display = 'flex';
                    div.style.flexDirection = 'column';
                    div.style.alignItems = 'center';
                    div.style.justifyContent = 'center';
                    div.style.borderRadius = '8px';
                    div.style.boxShadow = '0 2px 4px rgba(0,0,0,0.1)';
                    div.style.cursor = 'pointer';
                    div.style.fontSize = '11px';
                    div.style.textAlign = 'center';
                    div.style.padding = '4px';

                    div.title = `${item.domain || 'Unknown'} → ${item.organization || 'Unknown'} → ${item.department || 'Unknown'}\nResponses: ${item.count || 0}\nAvg Score: ${(item.avg_culture_score || 0).toFixed(2)}`;

                    const orgName = (item.organization || 'Unknown').slice(0, 12);
                    const deptName = (item.department || 'Unknown').slice(0, 10);

                    div.innerHTML = `<div style="font-weight:bold;font-size:10px;">${orgName}</div><div style="font-size:9px;">${deptName}</div><div style="font-size:8px;margin-top:2px;">${item.count || 0}</div>`;

                    container.appendChild(div);
                });

                // Add title
                const titleDiv = document.createElement('div');
                titleDiv.style.position = 'absolute';
                titleDiv.style.top = '5px';
                titleDiv.style.left = '10px';
                titleDiv.style.fontWeight = 'bold';
                titleDiv.style.color = '#495057';
                titleDiv.innerHTML = 'Organizational Structure Overview';
                container.appendChild(titleDiv);
            }

        } catch (error) {
            console.error('Failed to setup treemap chart:', error);
            const container = document.getElementById('treemapChart');
            if (container) container.innerHTML = '<div class="alert alert-danger">Error loading treemap visualization</div>';
        }
    }

    async setupRidgePlot() {
        try {
            console.log('📊 setupRidgePlot called');

            // Get filter values
            const domain = (document.getElementById('hierarchicalDomain') || {}).value || 'all';
            const bins = (document.getElementById('ridgeBins') || {}).value || 30;

            console.log('📊 Ridge plot filters:', { domain, bins });

            const ridgeData = await this.fetchWithCache(`/api/advanced/ridge?domain=${domain}&bins=${bins}`);
            const container = document.getElementById('ridgePlot');

            console.log('📊 Container found:', !!container);
            console.log('📊 Ridge data received:', ridgeData ? 'yes' : 'no');
            console.log('📊 Domains count:', ridgeData?.domains?.length || 0);

            if (!container) {
                console.error('📊 Container ridgePlot not found!');
                return;
            }

            if (!ridgeData || !ridgeData.domains || ridgeData.domains.length === 0) {
                console.log('📊 No ridge data available');
                container.innerHTML = '<div class="alert alert-info">No ridge plot data available</div>';
                return;
            }

            // Check if Plotly is available
            console.log('📊 Plotly available:', typeof Plotly !== 'undefined');
            if (typeof Plotly === 'undefined') {
                container.innerHTML = '<div class="alert alert-warning">Ridge plot requires Plotly.js. Please check your internet connection.</div>';
                return;
            }

            // Clear container
            container.innerHTML = '';

            const traces = [];
            const colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'];

            ridgeData.domains.forEach((domainData, domainIndex) => {
                domainData.distributions.forEach((distribution, sectionIndex) => {
                    if (distribution.count > 0) {
                        const y_offset = domainIndex * ridgeData.sections.length + sectionIndex;

                        // Normalize density for better visualization
                        const normalizedDensity = distribution.density.map(d => d * 0.8 + y_offset);

                        traces.push({
                            type: 'scatter',
                            mode: 'lines',
                            fill: 'tonexty',
                            x: distribution.x,
                            y: normalizedDensity,
                            name: `${domainData.domain} - ${distribution.section}`,
                            fillcolor: colors[sectionIndex % colors.length] + '40',
                            line: {
                                color: colors[sectionIndex % colors.length],
                                width: 2
                            },
                            hovertemplate: `<b>${domainData.domain}</b><br><b>${distribution.section}</b><br>Score: %{x:.2f}<br>Count: ${distribution.count}<br>Mean: ${distribution.mean.toFixed(2)}<extra></extra>`,
                            showlegend: domainIndex === 0
                        });

                        // Add baseline
                        traces.push({
                            type: 'scatter',
                            mode: 'lines',
                            x: distribution.x,
                            y: Array(distribution.x.length).fill(y_offset),
                            line: { color: colors[sectionIndex % colors.length], width: 1 },
                            showlegend: false,
                            hoverinfo: 'skip'
                        });
                    }
                });
            });

            const layout = {
                title: {
                    text: 'Section Score Distributions<br><sub>Cultural Assessment by Domain and Section</sub>',
                    font: { size: 16 }
                },
                xaxis: {
                    title: 'Culture Score',
                    range: [1, 4],
                    showgrid: true
                },
                yaxis: {
                    title: 'Domains and Sections',
                    tickmode: 'array',
                    tickvals: ridgeData.domains.flatMap((domain, dIdx) =>
                        ridgeData.sections.map((section, sIdx) => dIdx * ridgeData.sections.length + sIdx)
                    ),
                    ticktext: ridgeData.domains.flatMap(domain =>
                        ridgeData.sections.map(section => `${domain.domain}<br>${section.replace(' & ', '<br>&')}`)
                    ),
                    showgrid: false
                },
                margin: { t: 80, l: 200, r: 50, b: 60 },
                height: Math.max(400, ridgeData.domains.length * ridgeData.sections.length * 40)
            };

            const config = {
                responsive: true,
                displayModeBar: true,
                modeBarButtonsToAdd: ['zoom2d', 'pan2d', 'zoomIn2d', 'zoomOut2d', 'resetScale2d'],
                modeBarButtonsToRemove: ['lasso2d', 'select2d']
            };

            console.log('📊 About to create Plotly ridge plot with', traces.length, 'traces');

            try {
                await Plotly.newPlot(container, traces, layout, config);
                console.log('📊 Ridge plot created successfully!');
            } catch (plotlyError) {
                console.error('📊 Plotly ridge plot error:', plotlyError);
                container.innerHTML = `<div class="alert alert-danger">Plotly ridge plot error: ${plotlyError.message}</div>`;
            }

        } catch (error) {
            console.error('📊 Failed to setup ridge plot:', error);
            const container = document.getElementById('ridgePlot');
            if (container) container.innerHTML = `<div class="alert alert-danger">Error loading ridge plot visualization: ${error.message}</div>`;
        }
    }

    async setupSunburstChart() { return;
        try {
            console.log('🌻 setupSunburstChart called');

            // Get filter values
            const minCount = (document.getElementById('hierarchicalMinResponses') || {}).value || 3;
            const domain = (document.getElementById('hierarchicalDomain') || {}).value || 'all';

            console.log('🌻 Filter values:', { minCount, domain });

            const sunburstData = await this.fetchWithCache(`/api/advanced/sunburst?min_count=${minCount}&domain=${domain}`);
            const container = document.getElementById('sunburstChart');

            console.log('🌻 Container found:', !!container);
            console.log('🌻 Sunburst data received:', sunburstData?.length || 0, 'items');

            if (sunburstData && sunburstData.length > 0) {
                console.log('🌻 Sample sunburst data:', sunburstData[0]);
            }

            if (!container) {
                console.error('🌻 Container not found: sunburstChart');
                return;
            }

            if (!sunburstData || sunburstData.length === 0) {
                console.log('🌻 No data available');
                container.innerHTML = '<div class="alert alert-info">No sunburst data available with current filters</div>';
                return;
            }

            // Clear container
            container.innerHTML = '';

            // Check if Plotly is available
            console.log('🌻 Plotly available:', typeof Plotly !== 'undefined');
            if (typeof Plotly === 'undefined') {
                container.innerHTML = '<div class="alert alert-warning">Sunburst visualization requires Plotly.js. Please check your internet connection or install Plotly.js.</div>';
                return;
            }

            // Build proper hierarchical structure for sunburst
            const hierarchyNodes = new Set();
            const labels = [];
            const parents = [];
            const values = [];
            const colors = [];

            // First, collect all unique domains and position levels
            const domains = [...new Set(sunburstData.map(d => d.domain))];
            const domainPositions = {};

            sunburstData.forEach(d => {
                if (!domainPositions[d.domain]) {
                    domainPositions[d.domain] = new Set();
                }
                domainPositions[d.domain].add(d.position_level);
            });

            // Add root domains
            domains.forEach(domain => {
                if (!hierarchyNodes.has(domain)) {
                    labels.push(domain);
                    parents.push('');
                    values.push(0); // Will be calculated by Plotly
                    colors.push(0);
                    hierarchyNodes.add(domain);
                }
            });

            // Add domain/position level combinations
            Object.entries(domainPositions).forEach(([domain, positions]) => {
                positions.forEach(position => {
                    const positionKey = `${domain} - ${position}`;
                    if (!hierarchyNodes.has(positionKey)) {
                        labels.push(positionKey);
                        parents.push(domain);
                        values.push(0);
                        colors.push(0);
                        hierarchyNodes.add(positionKey);
                    }
                });
            });

            // Add department leaf nodes
            sunburstData.forEach(d => {
                const parentKey = `${d.domain} - ${d.position_level}`;
                const leafKey = `${d.domain} - ${d.position_level} - ${d.department}`;

                labels.push(leafKey);
                parents.push(parentKey);
                values.push(d.count);
                colors.push(d.avg_culture_score);
            });

            const trace = {
                type: 'sunburst',
                labels: labels,
                parents: parents,
                values: values,
                marker: {
                    colors: colors,
                    colorscale: 'RdBu',
                    reversescale: true,
                    colorbar: {
                        title: 'Culture Score',
                        titleside: 'right'
                    },
                    line: { width: 2, color: 'white' }
                },
                branchvalues: 'total',
                hovertemplate: '<b>%{label}</b><br>Responses: %{value}<br>Culture Score: %{marker.color:.2f}<extra></extra>',
                leaf: { opacity: 0.8 }
            };

            const layout = {
                title: {
                    text: 'Position Hierarchy Sunburst<br><sub>Domain → Position Level → Department</sub>',
                    font: { size: 16 }
                },
                margin: { t: 80, l: 10, r: 10, b: 10 },
                height: 420,
                font: { size: 12 }
            };

            const config = {
                responsive: true,
                displayModeBar: true,
                modeBarButtonsToAdd: ['zoom2d', 'pan2d', 'zoomIn2d', 'zoomOut2d', 'resetScale2d'],
                modeBarButtonsToRemove: ['lasso2d', 'select2d']
            };

            console.log('🌻 About to create Plotly sunburst with:', {
                labels: labels.length,
                parents: parents.length,
                values: values.length,
                colors: colors.length
            });

            console.log('🌻 Sample labels:', labels.slice(0, 5));
            console.log('🌻 Sample parents:', parents.slice(0, 5));

            try {
                await Plotly.newPlot(container, [trace], layout, config);
                console.log('🌻 Plotly sunburst created successfully!');
            } catch (plotlyError) {
                console.error('🌻 Plotly error:', plotlyError);
                container.innerHTML = `<div class="alert alert-danger">Plotly rendering error: ${plotlyError.message}</div>`;
            }

        } catch (error) {
            console.error('🌻 Failed to setup sunburst chart:', error);
            const container = document.getElementById('sunburstChart');
            if (container) container.innerHTML = `<div class="alert alert-danger">Error loading sunburst visualization: ${error.message}</div>`;
        }
    }

    async setupHierarchicalChart() {
        try {
            // This is the original hierarchical chart from the existing code
            const type = (document.getElementById('hierarchicalType') || {}).value || 'treemap';
            const minResp = parseInt((document.getElementById('hierarchicalMinResponses') || {}).value || '5', 10);
            const dom = (document.getElementById('hierarchicalDomain') || {}).value || 'all';

            // Fill domain dropdown if empty
            const domainSel = document.getElementById('hierarchicalDomain');
            if (domainSel && domainSel.options.length <= 1) {
                const domains = await this.fetchWithCache('/api/domains/list');
                const have = new Set(Array.from(domainSel.options).map(o => o.value));
                domains.forEach(domain => {
                    if (!have.has(domain)) {
                        const opt = new Option(domain, domain);
                        domainSel.appendChild(opt);
                    }
                });
            }

            // Fetch data and update original chart
            const params = new URLSearchParams({ min_count: String(minResp), domain: dom, _: String(Date.now()) });
            const hierarchical = await this.fetchWithCache(`/api/advanced/hierarchical?${params.toString()}`);
            if (!hierarchical) return;

            // Update original hierarchical visualization
            const treemapEl = document.getElementById('treemapChart');
            if (treemapEl && hierarchical.length > 0) {
                hierarchical.forEach(d => {
                    if (!d.color) d.color = this.getColorByDomain(d.domain);
                });
            }

        } catch (error) {
            console.error('Failed to setup hierarchical chart:', error);
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

            // Adjust chart options based on metric type using HSEG framework
            let chartOptions = {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: (metric === 'culture_score' || metric === 'section_scores') ? {
                        min: 7,
                        max: 28,
                        ticks: {
                            stepSize: 3,
                            callback: function(value, index, ticks) {
                                // Add HSEG tier indicators
                                if (value === 12) return value + ' (Crisis)';
                                if (value === 16) return value + ' (At Risk)';
                                if (value === 20) return value + ' (Mixed)';
                                if (value === 24) return value + ' (Safe)';
                                if (value === 28) return value + ' (Thriving)';
                                return value;
                            }
                        }
                    } : {}
                },
                plugins: {
                    legend: {
                        display: metric === 'section_scores',
                        position: 'right'
                    },
                    tooltip: {
                        callbacks: {
                            afterLabel: function(context) {
                                if (metric === 'culture_score' || metric === 'section_scores') {
                                    const score = context.parsed.y;
                                    if (score >= 7 && score <= 12) return 'Tier: Crisis 🔴';
                                    if (score >= 13 && score <= 16) return 'Tier: At Risk 🟠';
                                    if (score >= 17 && score <= 20) return 'Tier: Mixed ⚫';
                                    if (score >= 21 && score <= 24) return 'Tier: Safe 🔵';
                                    if (score >= 25 && score <= 28) return 'Tier: Thriving 🟢';
                                }
                                return '';
                            }
                        }
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
                // Only load PCA-specific charts, not all advanced charts
                await this.setupPCAChart();
                await this.setupVarianceChart();
                await this.setupPCALoadingsChart();
                break;
            case 'clustering':
                await this.setupClusteringChart();
                break;
            case 'correlations-advanced':
                await this.setupAdvancedCorrelations();
                break;
            case 'hierarchical':
                await this.setupHierarchicalChart();
                await this.setupTreemapChart();
                await this.setupRidgePlot();
                break;
        }
    }

    setupFilterEventListeners() {
        // Global filter change handlers
        const filterSelectors = [
            '#orgMinResponses', '#orgDomainFilter', '#scoreRange', '#orgSizeFilter',
            '#orgSortBy', '#performanceTimeRange', '#topOrgCount',
            '#distributionGroupBy', '#distributionMetric',
            '#sectionOrgFilter', '#sectionOrgMulti', '#sectionChartType', '#sectionComparisonMode', '#sectionScoringMode',
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
            if (!treemapEl) return;

            if (type === 'treemap') {
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
                    const maxCount = Math.max(...data.map(d => d.count || 1), 1);
                    data.slice(0, 30).forEach((item, idx) => {
                        const div = document.createElement('div');
                        const count = item.count || 1;
                        const ratio = count / maxCount;
                        const size = Math.max(40, Math.sqrt(ratio) * 120);
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
            }

            // Always update ridge-like chart for selected domain
            await this.setupRidgeChart(dom);
        } catch (error) {
            console.error('Failed to setup hierarchical chart:', error);
        }
    }

    async setupDendrogramChart(minCount = 5, domain = 'all') {
        try {
            console.log('🌳 setupDendrogramChart called');

            const container = document.getElementById('treemapChart');
            if (!container) return;

            const data = await this.fetchWithCache(`/api/advanced/hierarchical?min_count=${minCount}&domain=${domain}`);

            if (!data || data.length === 0) {
                container.innerHTML = '<div class="d-flex align-items-center justify-content-center h-100 text-muted" style="min-height:360px;">No data available for dendrogram</div>';
                return;
            }

            container.innerHTML = '';

            if (typeof Plotly !== 'undefined') {
                // Create a simple dendrogram using domain->org->dept hierarchy
                const domains = [...new Set(data.map(d => d.domain))];
                const nodes = [];
                const links = [];
                let nodeId = 0;

                // Create root node
                nodes.push({ id: nodeId++, name: 'Organizations', level: 0, x: 300, y: 50 });

                domains.forEach((domain, dIdx) => {
                    const domainId = nodeId++;
                    nodes.push({
                        id: domainId,
                        name: domain,
                        level: 1,
                        x: 150 + dIdx * 300,
                        y: 150
                    });
                    links.push({ source: 0, target: domainId });

                    const domainOrgs = [...new Set(data.filter(d => d.domain === domain).map(d => d.organization))];
                    domainOrgs.forEach((org, oIdx) => {
                        const orgId = nodeId++;
                        nodes.push({
                            id: orgId,
                            name: org.length > 15 ? org.substring(0, 15) + '...' : org,
                            level: 2,
                            x: 100 + dIdx * 300 + oIdx * 100,
                            y: 250
                        });
                        links.push({ source: domainId, target: orgId });
                    });
                });

                // Create a simple tree visualization
                const trace = {
                    x: nodes.map(n => n.x),
                    y: nodes.map(n => n.y),
                    text: nodes.map(n => n.name),
                    mode: 'markers+text',
                    type: 'scatter',
                    textposition: 'bottom center',
                    marker: {
                        size: nodes.map(n => n.level === 0 ? 15 : n.level === 1 ? 12 : 8),
                        color: nodes.map(n => n.level === 0 ? '#1f77b4' : n.level === 1 ? '#ff7f0e' : '#2ca02c'),
                    }
                };

                // Add connection lines
                const lineTraces = links.map(link => {
                    const source = nodes[link.source];
                    const target = nodes[link.target];
                    return {
                        x: [source.x, target.x],
                        y: [source.y, target.y],
                        mode: 'lines',
                        type: 'scatter',
                        line: { color: '#999', width: 1 },
                        showlegend: false,
                        hoverinfo: 'skip'
                    };
                });

                const layout = {
                    title: 'Organization Hierarchy Dendrogram',
                    showlegend: false,
                    height: 400,
                    xaxis: { showgrid: false, showticklabels: false, zeroline: false },
                    yaxis: { showgrid: false, showticklabels: false, zeroline: false },
                    margin: { t: 40, l: 20, r: 20, b: 20 }
                };

                await Plotly.newPlot(container, [trace, ...lineTraces], layout, { displayModeBar: false });
                console.log('🌳 Dendrogram created successfully!');
            } else {
                container.innerHTML = `
                    <div class="p-4">
                        <h6>Organization Hierarchy Structure</h6>
                        <div class="simple-tree">
                            ${domains.slice(0, 5).map(domain => `
                                <div class="domain-node mb-3">
                                    <strong>${domain}</strong>
                                    ${[...new Set(data.filter(d => d.domain === domain).map(d => d.organization))].slice(0, 3).map(org => `
                                        <div class="org-node ms-3">
                                            • ${org.length > 30 ? org.substring(0, 30) + '...' : org}
                                        </div>
                                    `).join('')}
                                </div>
                            `).join('')}
                        </div>
                    </div>
                `;
            }
        } catch (error) {
            console.error('🌳 Failed to setup dendrogram chart:', error);
            const container = document.getElementById('treemapChart');
            if (container) container.innerHTML = '<div class="alert alert-danger">Error loading dendrogram visualization</div>';
        }
    }

    async setupRidgeChart(domain) {
        try {
            const ctx = document.getElementById('ridgeChart');
            if (!ctx) return;
            const data = await this.fetchWithCache(`/api/advanced/ridge?domain=${encodeURIComponent(domain || 'all')}&bins=30`);
            if (!data || !data.domains || data.domains.length === 0) return;

            if (this.charts.ridgeChart) this.charts.ridgeChart.destroy();

            // Get the first domain's distributions (or combine all domains)
            const allDistributions = data.domains.flatMap(d => d.distributions);
            if (allDistributions.length === 0) return;

            // Use x values from the first distribution (they should be the same for all)
            const labels = allDistributions[0].x.map(v => Number(v).toFixed(2));
            const color = (i) => `hsl(${(i*55)%360} 70% 50%)`;

            // Calculate max density for offset calculation
            const maxDensity = Math.max(...allDistributions.flatMap(d => d.density));
            const offset = maxDensity * 1.2;

            const datasets = allDistributions.map((distribution, i) => ({
                label: distribution.section,
                data: distribution.density.map(v => v + i * offset),
                borderColor: color(i),
                backgroundColor: color(i) + '33',
                tension: 0.25,
                fill: true
            }));

            this.charts.ridgeChart = new Chart(ctx, {
                type: 'line',
                data: { labels, datasets },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: { duration: 0 }, // Disable animations
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: { boxWidth: 12 }
                        }
                    },
                    scales: {
                        y: { display: false },
                        x: {
                            title: {
                                display: true,
                                text: 'Culture Score'
                            }
                        }
                    }
                }
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

                // Color-code based on HSEG tier boundaries (7-28 scale)
                const colors = values.map(score => {
                    if (score <= 12) return '#dc2626'; // Crisis (7-12) - Red
                    if (score <= 16) return '#ea580c'; // At Risk (13-16) - Orange
                    if (score <= 20) return '#ca8a04'; // Mixed (17-20) - Yellow
                    if (score <= 24) return '#16a34a'; // Safe (21-24) - Green
                    return '#059669'; // Thriving (25-28) - Emerald
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
                                min: 7,
                                max: 28,
                                ticks: { stepSize: 3 },
                                title: { display: true, text: 'HSEG Score (7=Crisis, 28=Thriving)' }
                            },
                            x: {
                                title: { display: true, text: 'HSEG Cultural Assessment Categories' }
                            }
                        },
                        plugins: {
                            tooltip: {
                                callbacks: {
                                    afterLabel: function(context) {
                                        const score = context.parsed.y;
                                        if (score <= 12) return 'Tier: Crisis - Immediate intervention required';
                                        if (score <= 16) return 'Tier: At Risk - Early warning signs present';
                                        if (score <= 20) return 'Tier: Mixed - Inconsistent cultural experiences';
                                        if (score <= 24) return 'Tier: Safe - Generally healthy environment';
                                        return 'Tier: Thriving - Exemplary cultural practices';
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
            this.applyFilters(filterId);
        }, 300);
    }

    applyFilters(filterId) {
        // Apply all current filter values and refresh appropriate charts
        const activeTab = document.querySelector('.tab-pane.show.active');

        if (activeTab) {
            const tabId = activeTab.id;

            // Handle specific filter updates for advanced analytics
            if (filterId && filterId.includes('pca')) {
                // PCA filter changed - update PCA related charts
                this.setupPCAChart();
                this.setupPCALoadingsChart();
                this.setupVarianceChart();
            } else if (filterId && filterId.includes('clustering')) {
                // Clustering filter changed - update clustering chart
                this.setupClusteringChart();
            } else if (filterId && filterId.includes('hierarchical')) {
                // Hierarchical filter changed - update hierarchical charts
                this.setupHierarchicalChart();
                this.setupTreemapChart();
                this.setupRidgePlot();
            } else {
                // General filter change - reload tab data
                this.loadTabData(activeTab);
            }
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

    getFilterValue(filterType) {
        // Get filter values for insights generation
        switch (filterType) {
            case 'domain':
                // Try to get domain filter from org section, default to 'all'
                const domainElement = document.getElementById('orgDomainFilter');
                return domainElement ? domainElement.value : 'all';
            case 'organization':
                // Try to get selected organization, default to 'all'
                const orgElement = document.getElementById('selectedOrganization');
                return orgElement ? orgElement.value : 'all';
            default:
                return 'all';
        }
    }

    async generateStatisticalInsights() {
        // Generate HSEG-framework insights for investors and stakeholders
        const insightsContainer = document.getElementById('statisticalInsights');
        if (!insightsContainer) return;

        insightsContainer.innerHTML = `
            <div class="tab-loading">
                <div class="spinner-border" role="status"></div>
                <p>Analyzing HSEG cultural risk assessment framework...</p>
            </div>
        `;

        try {
            // Get current filter values
            const domainFilter = this.getFilterValue('domain');
            const orgFilter = this.getFilterValue('organization');

            // Build query parameters
            const params = new URLSearchParams();
            if (domainFilter && domainFilter !== 'all') params.append('domain', domainFilter);
            if (orgFilter && orgFilter !== 'all') params.append('organization', orgFilter);

            // Fetch comprehensive HSEG insights
            const hsegInsights = await this.fetchWithCache(`/api/insights/hseg?${params.toString()}`);

            if (hsegInsights.error) {
                throw new Error(hsegInsights.error);
            }

            // Render HSEG insights
            this.renderHSEGInsights(hsegInsights, insightsContainer);

        } catch (error) {
            console.error('Failed to generate HSEG insights:', error);
            insightsContainer.innerHTML = `
                <div class="alert alert-danger">
                    <i class="fas fa-exclamation-triangle me-2"></i>
                    <strong>Risk Assessment Error:</strong> Unable to generate cultural risk insights.
                    This may indicate data integrity issues requiring immediate attention.
                </div>
            `;
        }
    }

    renderHSEGInsights(hsegInsights, container) {
        const { overall_assessment, tier_distribution, category_analysis, risk_assessment, key_insights, recommendations, methodology } = hsegInsights;

        container.innerHTML = `
            <div class="insights-header mb-4">
                <h4><i class="fas fa-shield-alt me-2"></i>HSEG Cultural Risk Assessment - Executive Summary</h4>
                <p class="text-muted">Comprehensive risk intelligence based on the five-tier HSEG framework with 28-point weighted scoring</p>
            </div>

            <!-- Overall Assessment Card -->
            <div class="row g-3 mb-4">
                <div class="col-12">
                    <div class="card border-0 shadow-sm hseg-overview-card">
                        <div class="card-body">
                            <div class="row align-items-center">
                                <div class="col-md-3 text-center">
                                    <div class="hseg-tier-display ${overall_assessment.tier.toLowerCase()}-tier">
                                        <div class="tier-icon">${overall_assessment.tier_icon}</div>
                                        <div class="tier-score">${overall_assessment.score}</div>
                                        <div class="tier-name">${overall_assessment.tier}</div>
                                    </div>
                                </div>
                                <div class="col-md-9">
                                    <h5 class="mb-2">Overall Organizational Health: ${overall_assessment.tier}</h5>
                                    <p class="text-muted mb-3">${overall_assessment.tier_description}</p>
                                    <div class="row g-2">
                                        <div class="col-sm-4">
                                            <small class="text-muted">Score Range</small>
                                            <div class="fw-bold">${overall_assessment.score_range}</div>
                                        </div>
                                        <div class="col-sm-4">
                                            <small class="text-muted">Standard Deviation</small>
                                            <div class="fw-bold">±${overall_assessment.standard_deviation}</div>
                                        </div>
                                        <div class="col-sm-4">
                                            <small class="text-muted">Assessment Scale</small>
                                            <div class="fw-bold">7-28 Points</div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Risk Assessment Dashboard -->
            <div class="row g-3 mb-4">
                <div class="col-md-8">
                    <div class="card border-0 shadow-sm">
                        <div class="card-header bg-transparent">
                            <h6 class="mb-0"><i class="fas fa-chart-pie me-2"></i>Tier Distribution Analysis</h6>
                        </div>
                        <div class="card-body">
                            <div class="tier-distribution-grid">
                                ${Object.entries(tier_distribution.percentages).map(([tier, percentage]) => `
                                    <div class="tier-item ${tier.toLowerCase()}-tier">
                                        <div class="tier-percentage">${percentage}%</div>
                                        <div class="tier-label">${tier}</div>
                                        <div class="tier-count">${tier_distribution.counts[tier]} responses</div>
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card border-0 shadow-sm">
                        <div class="card-header bg-transparent">
                            <h6 class="mb-0"><i class="fas fa-exclamation-triangle me-2"></i>Risk Profile</h6>
                        </div>
                        <div class="card-body">
                            <div class="risk-gauge ${risk_assessment.risk_level.toLowerCase()}-risk">
                                <div class="risk-percentage">${risk_assessment.at_risk_percentage}%</div>
                                <div class="risk-label">${risk_assessment.risk_level} Risk</div>
                            </div>
                            <div class="mt-3">
                                <div class="small text-muted">At-Risk Population</div>
                                <div class="fw-bold">${risk_assessment.at_risk_count} / ${risk_assessment.total_assessed} responses</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Category Analysis -->
            <div class="row g-3 mb-4">
                <div class="col-12">
                    <div class="card border-0 shadow-sm">
                        <div class="card-header bg-transparent">
                            <h6 class="mb-0"><i class="fas fa-layer-group me-2"></i>HSEG Category Performance Analysis</h6>
                        </div>
                        <div class="card-body">
                            <div class="category-analysis-grid">
                                ${Object.entries(category_analysis).map(([category, data]) => `
                                    <div class="category-item ${data.risk_level.toLowerCase()}-risk">
                                        <div class="category-header">
                                            <h6 class="category-name">${category}</h6>
                                            <span class="risk-badge ${data.risk_level.toLowerCase()}">${data.risk_level}</span>
                                        </div>
                                        <div class="category-metrics">
                                            <div class="metric">
                                                <span class="metric-value">${data.average_score}</span>
                                                <span class="metric-label">Avg Score</span>
                                            </div>
                                            <div class="metric">
                                                <span class="metric-value">${data.weighted_score}</span>
                                                <span class="metric-label">Weighted</span>
                                            </div>
                                            <div class="metric">
                                                <span class="metric-value">${data.weight}x</span>
                                                <span class="metric-label">Weight</span>
                                            </div>
                                        </div>
                                        <div class="category-stats">
                                            <small class="text-muted">Range: ${data.score_range} | σ: ${data.standard_deviation}</small>
                                        </div>
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Key Insights and Recommendations -->
            <div class="row g-3 mb-4">
                <div class="col-md-6">
                    <div class="card border-0 shadow-sm">
                        <div class="card-header bg-transparent">
                            <h6 class="mb-0"><i class="fas fa-lightbulb me-2"></i>Key Strategic Insights</h6>
                        </div>
                        <div class="card-body">
                            ${key_insights.map(insight => `
                                <div class="insight-item mb-3">
                                    <div class="insight-text">${insight}</div>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card border-0 shadow-sm">
                        <div class="card-header bg-transparent">
                            <h6 class="mb-0"><i class="fas fa-tasks me-2"></i>Strategic Recommendations</h6>
                        </div>
                        <div class="card-body">
                            ${recommendations.map(rec => `
                                <div class="recommendation-item mb-3">
                                    <i class="fas fa-arrow-right me-2 text-primary"></i>
                                    <span>${rec}</span>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                </div>
            </div>

            <!-- Methodology Information -->
            <div class="row g-3">
                <div class="col-12">
                    <div class="card border-0 shadow-sm methodology-card">
                        <div class="card-header bg-transparent">
                            <h6 class="mb-0"><i class="fas fa-info-circle me-2"></i>Assessment Methodology</h6>
                        </div>
                        <div class="card-body">
                            <div class="row g-3">
                                <div class="col-md-3">
                                    <div class="methodology-metric">
                                        <div class="metric-value">${methodology.framework}</div>
                                        <div class="metric-label">Framework</div>
                                    </div>
                                </div>
                                <div class="col-md-2">
                                    <div class="methodology-metric">
                                        <div class="metric-value">${methodology.categories}</div>
                                        <div class="metric-label">Categories</div>
                                    </div>
                                </div>
                                <div class="col-md-2">
                                    <div class="methodology-metric">
                                        <div class="metric-value">${methodology.total_questions}</div>
                                        <div class="metric-label">Questions</div>
                                    </div>
                                </div>
                                <div class="col-md-3">
                                    <div class="methodology-metric">
                                        <div class="metric-value">${methodology.score_range}</div>
                                        <div class="metric-label">Score Range</div>
                                    </div>
                                </div>
                                <div class="col-md-2">
                                    <div class="methodology-metric">
                                        <div class="metric-value">Risk-Based</div>
                                        <div class="metric-label">Weighting</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    calculateHSEGInsights(stats, sections, organizations) {
        // HSEG Five-Tier Assessment Framework
        const riskZones = [
            { name: 'Crisis', range: [1.0, 1.5], icon: '🔴', cssClass: 'crisis-zone', description: 'Immediate intervention required' },
            { name: 'At Risk', range: [1.5, 2.0], icon: '🟠', cssClass: 'at-risk-zone', description: 'Early warning signs present' },
            { name: 'Mixed', range: [2.0, 2.5], icon: '⚪', cssClass: 'mixed-zone', description: 'Inconsistent experiences' },
            { name: 'Safe', range: [2.5, 3.0], icon: '🔵', cssClass: 'safe-zone', description: 'Generally healthy environment' },
            { name: 'Thriving', range: [3.0, 4.0], icon: '🟢', cssClass: 'thriving-zone', description: 'Exemplary cultural practices' }
        ];

        // Calculate organization distribution across risk zones
        const zoneDistribution = riskZones.map(zone => {
            const count = organizations.filter(org =>
                org.culture_score >= zone.range[0] && org.culture_score < zone.range[1]
            ).length;
            return {
                ...zone,
                count,
                percentage: Math.round((count / organizations.length) * 100)
            };
        });

        // Generate key insights for investors
        const keyInsights = [];

        // Crisis zone analysis (HSEG scores 7-12)
        const crisisOrgs = organizations.filter(org => org.culture_score <= 12);
        if (crisisOrgs.length > 0) {
            keyInsights.push({
                title: 'Critical Risk Exposure',
                priority: 'critical',
                priorityLabel: 'CRITICAL',
                icon: 'fas fa-exclamation-triangle',
                description: `${crisisOrgs.length} organizations are in crisis zones with severely compromised cultural environments.`,
                value: crisisOrgs.length,
                label: 'Organizations in Crisis',
                secondaryMetric: `${Math.round((crisisOrgs.length / organizations.length) * 100)}% of portfolio`,
                businessImpact: 'High liability risk, potential regulatory action, employee exodus, reputation damage',
                recommendedAction: 'Immediate leadership intervention, external cultural audit, crisis management plan'
            });
        }

        // High-performing organizations (HSEG scores 25-28)
        const thrivingOrgs = organizations.filter(org => org.culture_score >= 25);
        keyInsights.push({
            title: 'Cultural Excellence Leaders',
            priority: 'opportunity',
            priorityLabel: 'OPPORTUNITY',
            icon: 'fas fa-trophy',
            description: `${thrivingOrgs.length} organizations demonstrate exemplary cultural practices and can serve as benchmarks.`,
            value: thrivingOrgs.length,
            label: 'Thriving Organizations',
            secondaryMetric: `${Math.round((thrivingOrgs.length / organizations.length) * 100)}% of portfolio`,
            businessImpact: 'Talent retention, productivity gains, positive brand value, competitive advantage',
            recommendedAction: 'Extract best practices, expand successful models, use as cultural mentors'
        });

        // Section-based risk analysis (HSEG scores below 17 = Crisis + At Risk)
        const criticalSections = Object.entries(sections)
            .filter(([_, data]) => data.overall_score < 17)
            .sort((a, b) => a[1].overall_score - b[1].overall_score);

        if (criticalSections.length > 0) {
            const worstSection = criticalSections[0];
            keyInsights.push({
                title: 'Systemic Cultural Weakness',
                priority: 'warning',
                priorityLabel: 'WARNING',
                icon: 'fas fa-exclamation',
                description: `"${worstSection[0]}" shows the lowest scores across the portfolio, indicating systemic issues.`,
                value: worstSection[1].overall_score.toFixed(2),
                label: 'Risk Score',
                secondaryMetric: `Affects ${worstSection[1].count || 'multiple'} organizations`,
                businessImpact: 'Operational inefficiency, legal vulnerabilities, stakeholder confidence erosion',
                recommendedAction: 'Targeted intervention programs, policy review, leadership accountability measures'
            });
        }

        // Response volume and data quality
        const totalResponses = stats.total_responses || organizations.reduce((sum, org) => sum + org.response_count, 0);
        const avgResponsesPerOrg = Math.round(totalResponses / organizations.length);

        keyInsights.push({
            title: 'Assessment Coverage & Data Quality',
            priority: avgResponsesPerOrg < 10 ? 'warning' : 'info',
            priorityLabel: avgResponsesPerOrg < 10 ? 'CONCERN' : 'STABLE',
            icon: 'fas fa-chart-bar',
            description: `Assessment based on ${totalResponses} employee responses across ${organizations.length} organizations.`,
            value: avgResponsesPerOrg,
            label: 'Avg Responses/Org',
            secondaryMetric: `${totalResponses} total responses`,
            businessImpact: avgResponsesPerOrg < 10 ? 'Limited statistical reliability, potential blind spots' : 'Robust data foundation for decision-making',
            recommendedAction: avgResponsesPerOrg < 10 ? 'Increase participation rates, expand survey reach' : 'Maintain current assessment frequency'
        });

        // Domain analysis
        const domains = [...new Set(organizations.map(org => org.domain))];
        const domainAnalysis = domains.map(domain => {
            const domainOrgs = organizations.filter(org => org.domain === domain);
            const avgScore = domainOrgs.reduce((sum, org) => sum + org.culture_score, 0) / domainOrgs.length;
            const riskIndex = this.calculateRiskIndex(domainOrgs);

            let riskLevel, riskLabel, recommendation;
            if (avgScore < 2.0) {
                riskLevel = 'high-risk';
                riskLabel = 'HIGH RISK';
                recommendation = 'Immediate sector-wide intervention required';
            } else if (avgScore < 2.5) {
                riskLevel = 'medium-risk';
                riskLabel = 'MEDIUM RISK';
                recommendation = 'Enhanced monitoring and targeted improvements';
            } else {
                riskLevel = 'low-risk';
                riskLabel = 'STABLE';
                recommendation = 'Maintain current practices, monitor for degradation';
            }

            return {
                name: domain,
                orgCount: domainOrgs.length,
                avgScore: avgScore.toFixed(2),
                riskIndex: riskIndex.toFixed(1),
                riskLevel,
                riskLabel,
                recommendation
            };
        });

        return {
            riskZones: zoneDistribution,
            keyInsights,
            domainAnalysis
        };
    }

    calculateRiskIndex(organizations) {
        // Calculate a composite risk index based on score variance, response coverage, and trend
        if (!organizations.length) return 0;

        const scores = organizations.map(org => org.culture_score);
        const avgScore = scores.reduce((a, b) => a + b, 0) / scores.length;
        const variance = scores.reduce((sum, score) => sum + Math.pow(score - avgScore, 2), 0) / scores.length;
        const avgResponses = organizations.reduce((sum, org) => sum + org.response_count, 0) / organizations.length;

        // Risk increases with lower scores, higher variance, and lower response counts
        const scoreRisk = (4 - avgScore) * 25; // 0-75 points
        const varianceRisk = Math.min(variance * 20, 20); // 0-20 points
        const coverageRisk = Math.max(0, (10 - avgResponses) * 0.5); // 0-5 points

        return Math.min(100, scoreRisk + varianceRisk + coverageRisk);
    }

    async setupClusteringChart() {
        try {
            // Get filter values
            const algorithm = (document.getElementById('clusteringAlgorithm') || {}).value || 'kmeans';
            const clusterCount = (document.getElementById('clusterCount') || {}).value || '4';
            const metric = (document.getElementById('clusterMetric') || {}).value || 'euclidean';
            const clusterBy = (document.getElementById('clusterBy') || {}).value || 'organization';

            const data = await this.fetchWithCache(`/api/advanced/clustering?algorithm=${algorithm}&clusters=${clusterCount}&metric=${metric}&cluster_by=${clusterBy}`);
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
            description: `The average HSEG score across all categories is ${avgScore.toFixed(2)}, indicating ${avgScore >= 20 ? 'healthy (Mixed to Thriving)' : avgScore >= 17 ? 'mixed' : 'concerning (Crisis to At Risk)'} organizational culture.`,
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
        }
    } catch (e) {
        console.error('Failed to setup advanced correlations:', e);
    }
};
