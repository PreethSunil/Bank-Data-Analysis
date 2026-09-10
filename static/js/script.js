/* ==========================================================================
   Universal Bank Analytics - Frontend Engine (ES6 Vanilla JS)
   ========================================================================== */

// Global App State
const AppState = {
    theme: localStorage.getItem('ub_theme') || 'dark',
    analysisData: null,
    modelMetrics: null,
    customerPage: 1,
    customerSortBy: 'ID',
    customerSortDir: 'asc',
    customerSearch: '',
    chartInstances: {}
};

// Global Chart.js Colors & Dark Theme Defaults
const ChartColors = {
    gold: '#f59e0b',
    goldGlow: 'rgba(245, 158, 11, 0.4)',
    blue: '#3b82f6',
    blueGlow: 'rgba(59, 130, 246, 0.4)',
    green: '#10b981',
    purple: '#8b5cf6',
    cyan: '#06b6d4',
    red: '#ef4444',
    orange: '#f97316',
    grid: 'rgba(255, 255, 255, 0.08)',
    text: '#94a3b8'
};

document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    initClock();
    initNavigation();
    routePageInitializer();
});

/* ==========================================================================
   1. Theme & UI Controls
   ========================================================================== */
function initTheme() {
    const htmlEl = document.documentElement;
    htmlEl.setAttribute('data-theme', AppState.theme);
    updateThemeIcon();

    const themeBtn = document.getElementById('theme-toggle-btn');
    if (themeBtn) {
        themeBtn.addEventListener('click', () => {
            AppState.theme = AppState.theme === 'dark' ? 'light' : 'dark';
            htmlEl.setAttribute('data-theme', AppState.theme);
            localStorage.setItem('ub_theme', AppState.theme);
            updateThemeIcon();
            showToast(`Switched to ${AppState.theme.toUpperCase()} theme mode`, 'info');
        });
    }
}

function updateThemeIcon() {
    const icon = document.querySelector('#theme-toggle-btn i');
    if (icon) {
        icon.className = AppState.theme === 'dark' ? 'fa-solid fa-sun' : 'fa-solid fa-moon';
    }
}

function initClock() {
    const timeEl = document.getElementById('live-time');
    const dateEl = document.getElementById('live-date');

    function update() {
        const now = new Date();
        if (timeEl) timeEl.textContent = now.toLocaleTimeString();
        if (dateEl) dateEl.textContent = now.toLocaleDateString(undefined, { weekday: 'short', month: 'short', day: 'numeric', year: 'numeric' });
    }
    update();
    setInterval(update, 1000);
}

function initNavigation() {
    const sidebar = document.getElementById('sidebar');
    const sidebarToggle = document.getElementById('sidebar-toggle');
    const mobileMenuBtn = document.getElementById('mobile-menu-btn');

    if (sidebarToggle && sidebar) {
        sidebarToggle.addEventListener('click', () => {
            sidebar.classList.toggle('collapsed');
            const icon = sidebarToggle.querySelector('i');
            if (icon) icon.className = sidebar.classList.contains('collapsed') ? 'fa-solid fa-chevron-right' : 'fa-solid fa-chevron-left';

            // Dispatch resize event so Chart.js & Grid re-calculate dimensions smoothly
            setTimeout(() => {
                window.dispatchEvent(new Event('resize'));
            }, 300);
        });
    }

    if (mobileMenuBtn && sidebar) {
        mobileMenuBtn.addEventListener('click', () => {
            sidebar.classList.toggle('mobile-open');
            setTimeout(() => {
                window.dispatchEvent(new Event('resize'));
            }, 300);
        });
    }
}

function showToast(message, type = 'gold') {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `<i class="fa-solid fa-circle-info text-gold"></i> <span>${message}</span>`;
    container.appendChild(toast);

    setTimeout(() => {
        toast.remove();
    }, 4000);
}

function toggleLoading(show) {
    const loader = document.getElementById('loading-screen');
    if (loader) {
        if (show) loader.classList.remove('hidden');
        else loader.classList.add('hidden');
    }
}

/* ==========================================================================
   2. Page Specific Initializers
   ========================================================================== */
function routePageInitializer() {
    const path = window.location.pathname;

    if (path.includes('/dashboard') || path === '/') {
        loadDashboardData();
    } else if (path.includes('/analysis')) {
        loadAnalysisData();
    } else if (path.includes('/customers')) {
        loadCustomerTable();
        initTableSearch();
    } else if (path.includes('/models')) {
        loadModelMetrics();
    } else if (path.includes('/prediction')) {
        // Prediction page ready
    } else if (path.includes('/insights')) {
        loadInsights();
    } else if (path.includes('/churn')) {
        loadChurnData();
    }
}

/* ==========================================================================
   3. Dashboard Data & Quick Predict
   ========================================================================== */
async function loadDashboardData() {
    try {
        // Fetch KPIs
        const kpiRes = await fetch('/api/dashboard-kpis');
        const kpiData = await kpiRes.json();

        animateValue('kpi-total-customers', 0, kpiData.total_customers, 1000, '');
        animateValue('kpi-avg-age', 0, kpiData.avg_age, 1000, ' yrs');
        animateValue('kpi-avg-income', 0, kpiData.avg_income, 1000, 'k', '$');
        animateValue('kpi-avg-mortgage', 0, kpiData.avg_mortgage, 1000, 'k', '$');
        animateValue('kpi-loan-rate', 0, kpiData.loan_acceptance_rate, 1000, '%');
        animateValue('kpi-online-pct', 0, kpiData.online_banking_pct, 1000, '%');
        animateValue('kpi-cc-pct', 0, kpiData.credit_card_pct, 1000, '%');

        // Fetch Analysis Summary for Charts
        const res = await fetch('/api/analysis-data');
        const data = await res.json();

        renderDashLoanChart(data.loan_dist);
        renderDashIncomeChart(data.income_dist);

        // Fetch Model Snapshot
        const mRes = await fetch('/api/model-performance');
        const mData = await mRes.json();
        const bestName = mData.best_model || 'Random Forest';
        const bestMetrics = mData.metrics[bestName] || {};

        document.getElementById('dash-best-model-name').textContent = bestName;
        document.getElementById('dash-best-acc').textContent = `${bestMetrics.accuracy_pct}%`;
        document.getElementById('dash-best-prec').textContent = `${(bestMetrics.precision * 100).toFixed(1)}%`;
        document.getElementById('dash-best-rec').textContent = `${(bestMetrics.recall * 100).toFixed(1)}%`;
        document.getElementById('dash-best-f1').textContent = `${bestMetrics.f1_pct}%`;

    } catch (e) {
        console.error('Dashboard load error:', e);
    }
}

function animateValue(id, start, end, duration, suffix = '', prefix = '') {
    const obj = document.getElementById(id);
    if (!obj) return;
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        const val = (progress * (end - start) + start).toFixed(end % 1 === 0 ? 0 : 1);
        obj.innerHTML = `${prefix}${Number(val).toLocaleString()}<span class="unit">${suffix}</span>`;
        if (progress < 1) {
            window.requestAnimationFrame(step);
        }
    };
    window.requestAnimationFrame(step);
}

function renderDashLoanChart(loanDist) {
    const ctx = document.getElementById('dashLoanChart');
    if (!ctx) return;
    if (AppState.chartInstances['dashLoanChart']) AppState.chartInstances['dashLoanChart'].destroy();

    AppState.chartInstances['dashLoanChart'] = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: Object.keys(loanDist),
            datasets: [{
                data: Object.values(loanDist),
                backgroundColor: [ChartColors.gold, ChartColors.blue],
                borderColor: 'transparent',
                hoverOffset: 10
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'bottom', labels: { color: ChartColors.text, font: { family: 'Plus Jakarta Sans', size: 12 } } }
            },
            cutout: '70%'
        }
    });
}

function renderDashIncomeChart(incomeDist) {
    const ctx = document.getElementById('dashIncomeChart');
    if (!ctx) return;
    if (AppState.chartInstances['dashIncomeChart']) AppState.chartInstances['dashIncomeChart'].destroy();

    AppState.chartInstances['dashIncomeChart'] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: Object.keys(incomeDist),
            datasets: [{
                label: 'Customers Count',
                data: Object.values(incomeDist),
                backgroundColor: ChartColors.blueGlow,
                borderColor: ChartColors.blue,
                borderWidth: 2,
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { grid: { color: ChartColors.grid }, ticks: { color: ChartColors.text } },
                y: { grid: { color: ChartColors.grid }, ticks: { color: ChartColors.text } }
            },
            plugins: { legend: { display: false } }
        }
    });
}

async function runQuickPrediction() {
    const income = document.getElementById('quick-income').value;
    const cc_avg = document.getElementById('quick-ccavg').value;
    const cd_account = document.getElementById('quick-cd').value;
    const education = document.getElementById('quick-edu').value;

    const payload = {
        age: 40,
        experience: 15,
        income: income,
        family: 2,
        cc_avg: cc_avg,
        education: education,
        mortgage: 0,
        securities: 0,
        cd_account: cd_account,
        online: 1,
        credit_card: 0
    };

    try {
        const res = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();

        const resultBox = document.getElementById('quick-result-box');
        const statusEl = document.getElementById('quick-result-status');
        const probEl = document.getElementById('quick-result-prob');

        if (resultBox && statusEl && probEl) {
            resultBox.classList.remove('hidden');
            statusEl.textContent = data.prediction_label === 'ACCEPTED' ? '✓ LOAN ACCEPTANCE LIKELY' : '✗ LOAN REJECTION LIKELY';
            statusEl.style.color = data.prediction_label === 'ACCEPTED' ? ChartColors.gold : ChartColors.red;
            probEl.textContent = `Calculated Probability: ${data.probability_pct}% (${data.confidence_pct}% Confidence)`;
        }
    } catch (e) {
        showToast('Error executing prediction', 'red');
    }
}

/* ==========================================================================
   4. EDA / Analysis Page (14 Visualizations)
   ========================================================================== */
async function loadAnalysisData(filters = {}) {
    toggleLoading(true);
    try {
        let url = '/api/analysis-data?';
        if (filters.min_age) url += `min_age=${filters.min_age}&`;
        if (filters.max_age) url += `max_age=${filters.max_age}&`;
        if (filters.education) url += `education=${filters.education}&`;
        if (filters.loan_status) url += `loan_status=${filters.loan_status}&`;

        const res = await fetch(url);
        const data = await res.json();
        AppState.analysisData = data;

        document.getElementById('filter-count-badge').textContent = data.filtered_total.toLocaleString();

        // Render All 14 Visualizations
        createBarChart('chartAge', data.age_dist, 'Age Groups', ChartColors.blue);
        createBarChart('chartIncome', data.income_dist, 'Income Brackets ($000)', ChartColors.gold);
        createBarChart('chartExp', data.exp_dist, 'Work Experience', ChartColors.green);
        createDoughnutChart('chartEdu', data.edu_dist, [ChartColors.blue, ChartColors.purple, ChartColors.gold]);
        createBarChart('chartFamily', data.family_dist, 'Family Size', ChartColors.cyan);
        createDoughnutChart('chartMortgage', data.mortgage_dist, [ChartColors.text, ChartColors.red]);
        createDoughnutChart('chartLoan', data.loan_dist, [ChartColors.gold, ChartColors.blue]);
        createDoughnutChart('chartOnline', { 'Online User': data.binary_summary['Online Banking'], 'Non-Online': data.filtered_total - data.binary_summary['Online Banking'] }, [ChartColors.blue, ChartColors.grid]);
        createDoughnutChart('chartSecurities', { 'Securities Holder': data.binary_summary['Securities Account'], 'No Securities': data.filtered_total - data.binary_summary['Securities Account'] }, [ChartColors.green, ChartColors.grid]);
        createDoughnutChart('chartCC', { 'Credit Card User': data.binary_summary['Credit Card'], 'No Card': data.filtered_total - data.binary_summary['Credit Card'] }, [ChartColors.purple, ChartColors.grid]);
        createDoughnutChart('chartCD', { 'CD Account Holder': data.binary_summary['CD Account'], 'No CD Account': data.filtered_total - data.binary_summary['CD Account'] }, [ChartColors.gold, ChartColors.grid]);
        createBarChart('chartEduLoan', data.edu_loan_rate, 'Acceptance Rate (%)', ChartColors.gold, true);
        renderScatterPlot('chartScatter', data.scatter_data);
        renderHeatmap(data.correlation);

        toggleLoading(false);
    } catch (e) {
        console.error(e);
        toggleLoading(false);
    }
}

function applyAnalysisFilters() {
    const min_age = document.getElementById('filter-min-age').value;
    const max_age = document.getElementById('filter-max-age').value;
    const education = document.getElementById('filter-education').value;
    const loan_status = document.getElementById('filter-loan').value;

    loadAnalysisData({ min_age, max_age, education, loan_status });
    showToast('Analysis filters applied successfully', 'gold');
}

function resetAnalysisFilters() {
    document.getElementById('filter-min-age').value = '';
    document.getElementById('filter-max-age').value = '';
    document.getElementById('filter-education').value = '';
    document.getElementById('filter-loan').value = '';
    loadAnalysisData({});
    showToast('Filters reset to full dataset', 'info');
}

function createBarChart(canvasId, dictData, label, color, isPct = false) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;
    if (AppState.chartInstances[canvasId]) AppState.chartInstances[canvasId].destroy();

    AppState.chartInstances[canvasId] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: Object.keys(dictData),
            datasets: [{
                label: label,
                data: Object.values(dictData),
                backgroundColor: color + '44',
                borderColor: color,
                borderWidth: 2,
                borderRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { grid: { color: ChartColors.grid }, ticks: { color: ChartColors.text } },
                y: { grid: { color: ChartColors.grid }, ticks: { color: ChartColors.text, callback: (v) => isPct ? v + '%' : v } }
            },
            plugins: { legend: { display: false } }
        }
    });
}

function createDoughnutChart(canvasId, dictData, colors) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;
    if (AppState.chartInstances[canvasId]) AppState.chartInstances[canvasId].destroy();

    AppState.chartInstances[canvasId] = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: Object.keys(dictData),
            datasets: [{
                data: Object.values(dictData),
                backgroundColor: colors,
                borderColor: 'transparent',
                hoverOffset: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'bottom', labels: { color: ChartColors.text, font: { size: 11 } } }
            },
            cutout: '65%'
        }
    });
}

function renderScatterPlot(canvasId, scatterData) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;
    if (AppState.chartInstances[canvasId]) AppState.chartInstances[canvasId].destroy();

    const accepted = scatterData.filter(d => d['Personal Loan'] === 1).map(d => ({ x: d.Income, y: d.CCAvg }));
    const rejected = scatterData.filter(d => d['Personal Loan'] === 0).map(d => ({ x: d.Income, y: d.CCAvg }));

    AppState.chartInstances[canvasId] = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [
                { label: 'Accepted Loan (1)', data: accepted, backgroundColor: ChartColors.gold, pointRadius: 5 },
                { label: 'Rejected Loan (0)', data: rejected, backgroundColor: ChartColors.blueGlow, pointRadius: 3 }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { title: { display: true, text: 'Annual Income ($000)', color: ChartColors.text }, grid: { color: ChartColors.grid }, ticks: { color: ChartColors.text } },
                y: { title: { display: true, text: 'Monthly Credit Card Spend ($000)', color: ChartColors.text }, grid: { color: ChartColors.grid }, ticks: { color: ChartColors.text } }
            },
            plugins: { legend: { position: 'top', labels: { color: ChartColors.text } } }
        }
    });
}

function renderHeatmap(corrData) {
    const table = document.getElementById('heatmap-table');
    if (!table) return;

    let html = '<thead><tr><th>Feature</th>';
    corrData.features.forEach(f => {
        html += `<th>${f}</th>`;
    });
    html += '</tr></thead><tbody>';

    corrData.matrix.forEach((row, i) => {
        html += `<tr><th>${corrData.features[i]}</th>`;
        row.forEach(val => {
            const opacity = Math.abs(val);
            const bgColor = val >= 0 ? `rgba(245, 158, 11, ${opacity * 0.8})` : `rgba(59, 130, 246, ${opacity * 0.8})`;
            const textColor = opacity > 0.4 ? '#ffffff' : ChartColors.text;
            html += `<td style="background-color: ${bgColor}; color: ${textColor}">${val}</td>`;
        });
        html += '</tr>';
    });
    html += '</tbody>';
    table.innerHTML = html;
}

/* ==========================================================================
   5. Customer Table (Search, Sort, Pagination, Export)
   ========================================================================== */
async function loadCustomerTable() {
    try {
        const url = `/api/customer-data?page=${AppState.customerPage}&limit=15&search=${encodeURIComponent(AppState.customerSearch)}&sort_by=${AppState.customerSortBy}&sort_dir=${AppState.customerSortDir}`;
        const res = await fetch(url);
        const data = await res.json();

        const tbody = document.getElementById('customer-table-body');
        if (!tbody) return;

        let html = '';
        data.data.forEach(c => {
            const loanBadge = c['Personal Loan'] === 1
                ? '<span class="badge badge-gold">Accepted (1)</span>'
                : '<span class="badge badge-blue">Rejected (0)</span>';

            const eduLabel = c.Education === 1 ? 'Undergrad' : (c.Education === 2 ? 'Graduate' : 'Professional');

            html += `<tr>
                <td><strong>#${c.ID}</strong></td>
                <td>${c.Age} yrs</td>
                <td>${c.Experience} yrs</td>
                <td><strong>$${c.Income}k</strong></td>
                <td>${c.Family}</td>
                <td>$${c.CCAvg}k/mo</td>
                <td>${eduLabel}</td>
                <td>$${c.Mortgage}k</td>
                <td>${loanBadge}</td>
                <td>${c['CD Account'] === 1 ? 'Yes' : 'No'}</td>
                <td>${c.Online === 1 ? 'Yes' : 'No'}</td>
                <td>${c.CreditCard === 1 ? 'Yes' : 'No'}</td>
            </tr>`;
        });
        tbody.innerHTML = html;

        document.getElementById('current-page-num').textContent = data.page;
        document.getElementById('total-pages-num').textContent = data.total_pages;
        document.getElementById('table-total-num').textContent = data.total_records.toLocaleString();
        document.getElementById('table-start-num').textContent = ((data.page - 1) * data.limit) + 1;
        document.getElementById('table-end-num').textContent = Math.min(data.page * data.limit, data.total_records);

    } catch (e) {
        console.error(e);
    }
}

function initTableSearch() {
    const input = document.getElementById('table-search-input');
    if (!input) return;
    let timer = null;
    input.addEventListener('input', (e) => {
        clearTimeout(timer);
        timer = setTimeout(() => {
            AppState.customerSearch = e.target.value;
            AppState.customerPage = 1;
            loadCustomerTable();
        }, 300);
    });
}

function changePage(delta) {
    const totalPages = parseInt(document.getElementById('total-pages-num').textContent);
    const newPage = AppState.customerPage + delta;
    if (newPage >= 1 && newPage <= totalPages) {
        AppState.customerPage = newPage;
        loadCustomerTable();
    }
}

function sortTable(colName) {
    if (AppState.customerSortBy === colName) {
        AppState.customerSortDir = AppState.customerSortDir === 'asc' ? 'desc' : 'asc';
    } else {
        AppState.customerSortBy = colName;
        AppState.customerSortDir = 'asc';
    }
    loadCustomerTable();
}

function printTablePDF() {
    window.print();
}

/* ==========================================================================
   6. ML Models Performance Comparison Page
   ========================================================================== */
async function loadModelMetrics() {
    toggleLoading(true);
    try {
        const res = await fetch('/api/model-performance');
        const data = await res.json();
        AppState.modelMetrics = data;

        const rf = data.metrics['Random Forest'];
        if (!rf) { toggleLoading(false); return; }

        // Hero banner
        const heroAcc = document.getElementById('hero-best-acc');
        const heroF1  = document.getElementById('hero-best-f1');
        if (heroAcc) heroAcc.textContent = `${rf.accuracy_pct}%`;
        if (heroF1)  heroF1.textContent  = `${rf.f1_pct}%`;

        // 4 metric cards
        const set = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
        set('rf-acc',  `${rf.accuracy_pct}%`);
        set('rf-prec', `${(rf.precision * 100).toFixed(1)}%`);
        set('rf-rec',  `${(rf.recall    * 100).toFixed(1)}%`);
        set('rf-f1',   `${rf.f1_pct}%`);

        // Feature importance chart (Chart.js)
        renderFeatureImportanceChart(data.feature_importance);

        toggleLoading(false);
    } catch (e) {
        console.error(e);
        toggleLoading(false);
    }
}

function renderModelComparisonChart(metrics) {
    const ctx = document.getElementById('modelCompChart');
    if (!ctx) return;

    const names = Object.keys(metrics);
    const accs = names.map(n => metrics[n].accuracy_pct);
    const precs = names.map(n => (metrics[n].precision * 100).toFixed(1));
    const recs = names.map(n => (metrics[n].recall * 100).toFixed(1));
    const f1s = names.map(n => metrics[n].f1_pct);

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: names,
            datasets: [
                { label: 'Accuracy (%)', data: accs, backgroundColor: ChartColors.blue },
                { label: 'Precision (%)', data: precs, backgroundColor: ChartColors.green },
                { label: 'Recall (%)', data: recs, backgroundColor: ChartColors.purple },
                { label: 'F1 Score (%)', data: f1s, backgroundColor: ChartColors.gold }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { grid: { color: ChartColors.grid }, ticks: { color: ChartColors.text } },
                y: { min: 40, max: 100, grid: { color: ChartColors.grid }, ticks: { color: ChartColors.text } }
            },
            plugins: { legend: { labels: { color: ChartColors.text } } }
        }
    });
}

function renderFeatureImportanceChart(featImp) {
    const ctx = document.getElementById('featureImpChart');
    if (!ctx) return;

    const labels = featImp.map(f => f.feature);
    const values = featImp.map(f => (f.importance * 100).toFixed(1));

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Importance Weight (%)',
                data: values,
                backgroundColor: ChartColors.goldGlow,
                borderColor: ChartColors.gold,
                borderWidth: 2,
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { grid: { color: ChartColors.grid }, ticks: { color: ChartColors.text } },
                y: { grid: { color: ChartColors.grid }, ticks: { color: ChartColors.text } }
            },
            plugins: { legend: { display: false } }
        }
    });
}

function switchMatrixModel(modelName, btnEl = null) {
    if (!AppState.modelMetrics) return;

    if (btnEl) {
        document.querySelectorAll('.model-select-tabs .tab-btn').forEach(b => b.classList.remove('active'));
        btnEl.classList.add('active');
    }

    const cm = AppState.modelMetrics.metrics[modelName]?.confusion_matrix || [[0, 0], [0, 0]];
    document.getElementById('cm-tn').innerHTML = `${cm[0][0]} <span class="cm-sub">(True Negatives)</span>`;
    document.getElementById('cm-fp').innerHTML = `${cm[0][1]} <span class="cm-sub">(False Positives)</span>`;
    document.getElementById('cm-fn').innerHTML = `${cm[1][0]} <span class="cm-sub">(False Negatives)</span>`;
    document.getElementById('cm-tp').innerHTML = `${cm[1][1]} <span class="cm-sub">(True Positives)</span>`;
}

/* ==========================================================================
   7. Live Prediction Page Handler
   ========================================================================== */
function fillSampleProfile() {
    document.getElementById('p-age').value = 42;
    document.getElementById('p-exp').value = 18;
    document.getElementById('p-income').value = 165;
    document.getElementById('p-family').value = 3;
    document.getElementById('p-ccavg').value = 5.4;
    document.getElementById('p-edu').value = 3;
    document.getElementById('p-mortgage').value = 210;
    document.getElementById('p-securities').value = 1;
    document.getElementById('p-cd').value = 1;
    document.getElementById('p-online').value = 1;
    document.getElementById('p-cc').value = 1;
    showToast('Loaded high probability sample profile!', 'gold');
}

async function handlePrediction(e) {
    e.preventDefault();

    const submitBtn = document.getElementById('predict-submit-btn');
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Running Machine Learning Model...';

    const payload = {
        age: document.getElementById('p-age').value,
        experience: document.getElementById('p-exp').value,
        income: document.getElementById('p-income').value,
        family: document.getElementById('p-family').value,
        cc_avg: document.getElementById('p-ccavg').value,
        education: document.getElementById('p-edu').value,
        mortgage: document.getElementById('p-mortgage').value,
        securities: document.getElementById('p-securities').value,
        cd_account: document.getElementById('p-cd').value,
        online: document.getElementById('p-online').value,
        credit_card: document.getElementById('p-cc').value
    };

    try {
        const res = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();

        submitBtn.disabled = false;
        submitBtn.innerHTML = '<i class="fa-solid fa-brain"></i> Calculate Loan Acceptance Probability';

        if (data.error) {
            showToast(`Prediction Error: ${data.error}`, 'red');
            return;
        }

        // Render Output
        document.getElementById('result-placeholder').classList.add('hidden');
        const resultContent = document.getElementById('result-content');
        resultContent.classList.remove('hidden');

        const banner = document.getElementById('status-banner');
        const icon = document.getElementById('status-icon');
        const title = document.getElementById('status-title');
        const subtitle = document.getElementById('status-subtitle');

        if (data.prediction === 1) {
            banner.className = 'status-banner accepted';
            icon.innerHTML = '<i class="fa-solid fa-circle-check"></i>';
            title.textContent = 'PERSONAL LOAN ACCEPTANCE LIKELY';
            subtitle.textContent = 'Model Classification: Class 1 (Accepted)';
        } else {
            banner.className = 'status-banner rejected';
            icon.innerHTML = '<i class="fa-solid fa-circle-xmark"></i>';
            title.textContent = 'PERSONAL LOAN REJECTION LIKELY';
            subtitle.textContent = 'Model Classification: Class 0 (Rejected)';
        }

        document.getElementById('prob-pct-num').textContent = `${data.probability_pct}%`;
        document.getElementById('prob-bar-fill').style.width = `${data.probability_pct}%`;
        document.getElementById('confidence-pct').textContent = `${data.confidence_pct}%`;
        document.getElementById('risk-tier').textContent = data.risk_level;
        document.getElementById('rec-text').textContent = data.recommendation;

        showToast('Prediction calculated successfully', 'gold');

    } catch (err) {
        submitBtn.disabled = false;
        submitBtn.innerHTML = '<i class="fa-solid fa-brain"></i> Calculate Loan Acceptance Probability';
        showToast('Server connection error during prediction', 'red');
    }
}

/* ==========================================================================
   8. Automated Insights Page
   ========================================================================== */
async function loadInsights() {
    try {
        const res = await fetch('/api/insights');
        const data = await res.json();

        document.getElementById('ins-income-grp').textContent = 'Income > $100k';
        document.getElementById('ins-edu-lvl').textContent = data.top_edu.split(' ')[0] + ' Degree';
        document.getElementById('ins-avg-mortgage').textContent = data.avg_mortgage_val;
        document.getElementById('ins-loan-pct').textContent = data.overall_acceptance;
        document.getElementById('ins-online-pct').textContent = data.online_adoption;
        document.getElementById('ins-cc-pct').textContent = data.cc_ownership;

        const container = document.getElementById('insights-container');
        if (!container) return;

        let html = '';
        data.insights.forEach(item => {
            html += `<div class="glass-card insight-card-item">
                <div class="insight-meta">
                    <span class="insight-cat"><i class="fa-solid fa-tag"></i> ${item.category}</span>
                    <span class="badge badge-gold">${item.impact}</span>
                </div>
                <h3 class="insight-item-title">${item.title}</h3>
                <p class="insight-item-desc">${item.description}</p>
            </div>`;
        });
        container.innerHTML = html;

    } catch (e) {
        console.error(e);
    }
}

/* ==========================================================================
   9. Churn Analysis Page
   ========================================================================== */
async function loadChurnData() {
    try {
        const res = await fetch('/api/churn-data');
        const data = await res.json();
        
        animateValue('churn-total', 0, data.total_customers, 1000, '');
        animateValue('churn-rate', 0, data.churn_rate, 1000, '%');
        animateValue('churn-avg-credit', 0, data.avg_credit_score, 1000, '');
        animateValue('churn-avg-balance', 0, data.avg_balance, 1000, '', '$');
        
        // churnGeoChart (bar)
        const geoDict = {};
        for (const [k, v] of Object.entries(data.churn_by_geography)) { geoDict[k] = v.rate; }
        createBarChart('churnGeoChart', geoDict, 'Churn Rate (%)', ChartColors.gold, true);
        
        // churnGenderChart (doughnut)
        const genderDict = {};
        for (const [k, v] of Object.entries(data.churn_by_gender)) { genderDict[k] = v.churned; }
        createDoughnutChart('churnGenderChart', genderDict, [ChartColors.blue, ChartColors.purple]);
        
        // churnAgeChart (bar)
        const ageDict = {};
        for (const [k, v] of Object.entries(data.churn_by_age_group)) { ageDict[k] = v.churned; }
        createBarChart('churnAgeChart', ageDict, 'Churned Count', ChartColors.cyan);
        
        // churnProductsChart (bar)
        const prodDict = {};
        for (const [k, v] of Object.entries(data.churn_by_products)) { prodDict[k] = v.rate; }
        createBarChart('churnProductsChart', prodDict, 'Churn Rate (%)', ChartColors.green, true);
        
        // churnTenureChart (line)
        const ctxTenure = document.getElementById('churnTenureChart');
        if (ctxTenure) {
            if (AppState.chartInstances['churnTenureChart']) AppState.chartInstances['churnTenureChart'].destroy();
            const tenLabels = Object.keys(data.churn_by_tenure).sort((a,b) => parseInt(a)-parseInt(b));
            const tenRates = tenLabels.map(l => data.churn_by_tenure[l]);
            
            AppState.chartInstances['churnTenureChart'] = new Chart(ctxTenure, {
                type: 'line',
                data: {
                    labels: tenLabels,
                    datasets: [{
                        label: 'Churn Rate (%)',
                        data: tenRates,
                        borderColor: ChartColors.orange,
                        backgroundColor: ChartColors.orange + '22',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { grid: { color: ChartColors.grid }, ticks: { color: ChartColors.text } },
                        y: { grid: { color: ChartColors.grid }, ticks: { color: ChartColors.text, callback: function(value) { return value + "%" } } }
                    },
                    plugins: { legend: { display: false } }
                }
            });
        }
        
        // churnActiveChart (doughnut)
        const actDict = {
            'Active': data.churn_by_activity.active.churned,
            'Inactive': data.churn_by_activity.inactive.churned
        };
        createDoughnutChart('churnActiveChart', actDict, [ChartColors.gold, ChartColors.red]);
        
    } catch(e) {
        console.error('Error loading churn data', e);
    }
}

/* Helper to Export Chart as PNG */
function exportChartPNG(canvasId, filename) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const link = document.createElement('a');
    link.download = `${filename}.png`;
    link.href = canvas.toDataURL('image/png');
    link.click();
    showToast(`Exported ${filename}.png`, 'gold');
}
