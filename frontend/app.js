// FinTech Risk Analyzer Frontend
// Configuration
const API_BASE_URL = 'http://localhost:5002/api';

// Global variables
let riskDistributionChart = null;
let riskTrendsChart = null;

// Pagination variables
let currentPage = 1;
let perPage = 50;
let totalTransactions = 0;
let totalPages = 0;

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 FinTech Risk Analyzer Frontend Initialized');
    
    // Initialize tabs
    initializeTabs();
    
    // Load initial dashboard data
    loadDashboardData();
    
    // Set up form submission
    setupFormHandlers();
    
    // Initialize charts
    initializeCharts();
    
    // Load transactions for the transactions tab
    loadTransactions();
    
    // Load insights for the insights tab
    loadInsights();
});

/**
 * Initialize Bootstrap tabs functionality
 */
function initializeTabs() {
    const tabElements = document.querySelectorAll('[data-bs-toggle="tab"]');
    tabElements.forEach(tab => {
        tab.addEventListener('shown.bs.tab', function(event) {
            const targetTab = event.target.getAttribute('data-bs-target');
            console.log('📑 Tab switched to:', targetTab);
            
            // Load data for specific tabs when they become active
            switch(targetTab) {
                case '#dashboard':
                    loadDashboardData();
                    break;
                case '#transactions':
                    loadTransactions();
                    break;
                case '#insights':
                    loadInsights();
                    break;
            }
        });
    });
    
    // Also load data when page first loads (for dashboard tab)
    // Add a small delay to ensure charts are initialized
    setTimeout(() => {
        loadDashboardData();
    }, 100);
}

/**
 * Set up form handlers
 */
function setupFormHandlers() {
    const form = document.getElementById('riskAnalysisForm');
    if (form) {
        form.addEventListener('submit', handleTransactionSubmit);
    }
}

/**
 * Handle transaction form submission
 */
async function handleTransactionSubmit(event) {
    event.preventDefault();
    
    const formData = new FormData(event.target);
    const transactionData = {
        amount: parseFloat(formData.get('amount')),
        merchant: formData.get('merchant'),
        device_type: formData.get('deviceType'),
        latitude: parseFloat(formData.get('latitude')),
        longitude: parseFloat(formData.get('longitude'))
    };
    
    // Add timestamp if provided
    const timestamp = formData.get('timestamp');
    if (timestamp) {
        transactionData.timestamp = new Date(timestamp).toISOString();
    }
    
    try {
        showLoading(true);
        showAlert('Analyzing transaction...', 'info');
        
        const response = await fetch(`${API_BASE_URL}/analyze`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(transactionData)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        console.log('✅ Analysis result:', result);
        console.log('💾 Transaction should be saved with ID:', result.database_id);
        console.log('💾 Transaction ID:', result.transaction_id);
        
        displayRiskResults(result);
        showAlert(`Transaction analyzed and saved successfully! Transaction ID: ${result.transaction_id}. You can now view it in the Transactions tab.`, 'success');
        
        // Switch to analyze tab to show results
        const analyzeTab = document.getElementById('analyze-tab');
        if (analyzeTab) {
            const tab = new bootstrap.Tab(analyzeTab);
            tab.show();
        }
        
        // Refresh dashboard data and transactions
        loadDashboardData();
        
        // Refresh transactions tab to show the new entry immediately
        loadTransactions();
        
        // Also refresh after a short delay to ensure the API has processed the save
        setTimeout(() => {
            loadTransactions();
        }, 2000);
        
    } catch (error) {
        console.error('Error analyzing transaction:', error);
        showAlert('Error analyzing transaction: ' + error.message, 'danger');
    } finally {
        showLoading(false);
    }
}

/**
 * Display risk analysis results
 */
function displayRiskResults(result) {
    console.log('🎯 displayRiskResults called with:', result);
    
    const resultsSection = document.getElementById('resultsSection');
    const riskScore = document.getElementById('riskScore');
    const riskLevel = document.getElementById('riskLevel');
    const mlPredictionBar = document.getElementById('mlPredictionBar');
    const mlPredictionText = document.getElementById('mlPredictionText');
    const ruleBasedBar = document.getElementById('ruleBasedBar');
    const ruleBasedText = document.getElementById('ruleBasedText');
    const riskFactors = document.getElementById('riskFactors');
    
    if (!result.risk_analysis) {
        showAlert('Invalid response format from API', 'warning');
        return;
    }
    
    const analysis = result.risk_analysis;
    console.log('📊 Analysis object:', analysis);
    
    // Update risk score display
    const finalScore = Math.round(analysis.final_score * 100);
    riskScore.textContent = finalScore;
    
    // Update risk level and styling
    const level = analysis.risk_level;
    riskLevel.textContent = level.charAt(0).toUpperCase() + level.slice(1);
    riskLevel.className = `mt-2 risk-${level}`;
    
    // Update progress bars
    const mlScore = Math.round(analysis.ml_prediction * 100);
    const ruleScore = Math.round(analysis.rule_based_score * 100);
    
    mlPredictionBar.style.width = `${mlScore}%`;
    mlPredictionText.textContent = `${mlScore}%`;
    
    ruleBasedBar.style.width = `${ruleScore}%`;
    ruleBasedText.textContent = `${ruleScore}%`;
    
    // Update risk factors
    riskFactors.innerHTML = generateRiskFactorsHTML(analysis);
    
    // Update detailed explanations
    console.log('🔄 About to call updateDetailedExplanations...');
    updateDetailedExplanations(analysis);
    console.log('✅ updateDetailedExplanations completed');
    
    // Show results section
    resultsSection.style.display = 'block';
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

/**
 * Generate HTML for risk factors display
 */
function generateRiskFactorsHTML(analysis) {
    const factors = [];
    
    if (analysis.risk_level === 'high') {
        factors.push('<span class="badge bg-danger me-2">High Risk</span>');
    } else if (analysis.risk_level === 'medium') {
        factors.push('<span class="badge bg-warning me-2">Medium Risk</span>');
    } else {
        factors.push('<span class="badge bg-success me-2">Low Risk</span>');
    }
    
    // Add specific risk indicators based on the analysis
    if (analysis.ml_prediction > 0.7) {
        factors.push('<span class="badge bg-warning me-2">ML High Risk</span>');
    }
    
    if (analysis.rule_based_score > 0.7) {
        factors.push('<span class="badge bg-info me-2">Rule High Risk</span>');
    }
    
    // Add transaction details
    factors.push('<div class="mt-2"><strong>Transaction ID:</strong> ' + (analysis.transaction_id || 'N/A') + '</div>');
    factors.push('<div><strong>Analysis Time:</strong> ' + new Date().toLocaleString() + '</div>');
    
    return factors.join('');
}

/**
 * Update detailed explanations for risk analysis
 */
async function updateDetailedExplanations(analysis) {
    console.log('🔍 updateDetailedExplanations called with:', analysis);
    
    // Test if we can find any elements at all
    console.log('🔍 Testing element discovery...');
    const testElement = document.getElementById('mlReasons');
    console.log('🔍 Test element found:', testElement);
    console.log('🔍 Document ready state:', document.readyState);
    
    // Update ML reasoning
    const mlReasons = document.getElementById('mlReasons');
    console.log('📱 mlReasons element:', mlReasons);
    if (mlReasons && analysis.ml_reasons) {
        console.log('✅ Updating ML reasons:', analysis.ml_reasons);
        mlReasons.innerHTML = analysis.ml_reasons.map(reason => 
            `<div class="explanation-item"><i class="fas fa-chevron-right me-2"></i>${reason}</div>`
        ).join('');
        console.log('✅ ML reasons HTML updated');
    } else {
        console.log('❌ ML reasons not found or element missing');
        console.log('❌ mlReasons element exists:', !!mlReasons);
        console.log('❌ analysis.ml_reasons exists:', !!analysis.ml_reasons);
        if (analysis.ml_reasons) {
            console.log('❌ analysis.ml_reasons content:', analysis.ml_reasons);
        }
    }
    
    // Update rule-based reasoning
    const ruleReasons = document.getElementById('ruleReasons');
    console.log('⚙️ ruleReasons element:', ruleReasons);
    if (ruleReasons && analysis.rule_reasons) {
        console.log('✅ Updating rule reasons:', analysis.rule_reasons);
        ruleReasons.innerHTML = analysis.rule_reasons.map(reason => 
            `<div class="explanation-item"><i class="fas fa-chevron-right me-2"></i>${reason}</div>`
        ).join('');
    } else {
        console.log('❌ Rule reasons not found or element missing');
    }
    
        // Update final reasoning
    const finalReasoning = document.getElementById('finalReasoning');
    console.log('🎯 finalReasoning element:', finalReasoning);
    if (finalReasoning && analysis.final_reasoning) {
        console.log('✅ Updating final reasoning:', analysis.final_reasoning);
        finalReasoning.innerHTML = analysis.final_reasoning.map(reason =>
            `<div class="explanation-item"><i class="fas fa-info-circle me-2"></i>${reason}</div>`
        ).join('');
    } else {
        console.log('❌ Final reasoning not found or element missing');
    }

    // Generate and update AI explanation and prevention tips
    console.log('🤖 Generating AI explanation...');
    const aiExplanation = await generateAIExplanation(analysis);
    
    const aiExplanationElement = document.getElementById('aiExplanation');
    if (aiExplanationElement) {
        aiExplanationElement.innerHTML = `<div class="explanation-item"><i class="fas fa-lightbulb me-2"></i>${aiExplanation.explanation}</div>`;
        console.log('✅ AI explanation updated');
    }
    
    const preventionTipsElement = document.getElementById('preventionTips');
    if (preventionTipsElement) {
        preventionTipsElement.innerHTML = aiExplanation.preventionTips;
        console.log('✅ Prevention tips updated');
    }
}

/**
 * Load dashboard data from API
 */
async function loadDashboardData() {
    try {
        console.log('📊 Loading dashboard data...');
        
        // Load risk summary
        const summaryResponse = await fetch(`${API_BASE_URL}/risk-summary`);
        if (summaryResponse.ok) {
            const summary = await summaryResponse.json();
            updateDashboardStats(summary);
        }
        
        // Load recent transactions for charts
        const transactionsResponse = await fetch(`${API_BASE_URL}/transactions?per_page=1000`);
        if (transactionsResponse.ok) {
            const transactions = await transactionsResponse.json();
            console.log('📊 Loaded', transactions.transactions.length, 'transactions for charts');
            updateCharts(transactions.transactions);
        }
        
    } catch (error) {
        console.error('Error loading dashboard data:', error);
        showAlert('Error loading dashboard data', 'warning');
    }
}

/**
 * Update dashboard statistics
 */
function updateDashboardStats(summary) {
    console.log('📈 Updating dashboard stats:', summary);
    
    // Extract data from the nested structure
    const stats = summary.overall_stats || summary;
    
    const totalTransactions = document.getElementById('totalTransactions');
    const avgRiskScore = document.getElementById('avgRiskScore');
    const highRiskCount = document.getElementById('highRiskCount');
    const fraudRate = document.getElementById('fraudRate');
    
    if (totalTransactions) totalTransactions.textContent = stats.total_transactions?.toLocaleString() || '0';
    
    // Calculate average risk score from risk counts
    if (avgRiskScore && stats.total_transactions) {
        const totalRisk = (stats.high_risk_count * 0.8) + (stats.medium_risk_count * 0.5) + (stats.low_risk_count * 0.2);
        const avgRisk = totalRisk / stats.total_transactions;
        avgRiskScore.textContent = Math.round(avgRisk * 100) + '%';
    }
    
    if (highRiskCount) highRiskCount.textContent = stats.high_risk_count?.toLocaleString() || '0';
    if (fraudRate) fraudRate.textContent = (stats.fraud_rate || 0).toFixed(2) + '%';
    
    console.log('📈 Updated dashboard with:', {
        total: stats.total_transactions,
        avgRisk: 'calculated from risk counts',
        highRisk: stats.high_risk_count,
        fraudRate: stats.fraud_rate
    });
}

/**
 * Load transactions for the transactions tab
 */
async function loadTransactions() {
    try {
        console.log('📋 Loading transactions page', currentPage, 'with', perPage, 'per page...');
        
        const response = await fetch(`${API_BASE_URL}/transactions?per_page=${perPage}&page=${currentPage}`);
        if (response.ok) {
            const data = await response.json();
            console.log('📋 Transactions API response:', data);
            
            // Update pagination info
            totalTransactions = data.pagination.total;
            totalPages = data.pagination.pages;
            currentPage = data.pagination.page;
            
                    // Sort transactions by timestamp (newest first)
        const sortedTransactions = data.transactions.sort((a, b) => {
            const timeA = new Date(a.timestamp || 0);
            const timeB = new Date(b.timestamp || 0);
            return timeB - timeA;
        });
        
        console.log('📋 Loaded', sortedTransactions.length, 'transactions, sorted by newest first');
        console.log('📋 First transaction timestamp:', sortedTransactions[0]?.timestamp);
        console.log('📋 Last transaction timestamp:', sortedTransactions[sortedTransactions.length - 1]?.timestamp);
        
        updateTransactionsTable(sortedTransactions);
        updatePaginationControls();
        } else {
            console.error('❌ Transactions API error:', response.status);
        }
        
    } catch (error) {
        console.error('Error loading transactions:', error);
        showAlert('Error loading transactions', 'warning');
    }
}

/**
 * Update transactions table
 */
function updateTransactionsTable(transactions) {
    const tbody = document.getElementById('transactionsTableBody');
    if (!tbody) return;
    
    if (!transactions || transactions.length === 0) {
        tbody.innerHTML = '<tr><td colspan="7" class="text-center text-muted">No transactions found</td></tr>';
        return;
    }
    
    console.log('📋 Updating transactions table with:', transactions[0]); // Log first transaction for debugging
    console.log('📋 First transaction structure:', {
        id: transactions[0]?.id,
        transaction_id: transactions[0]?.transaction_id,
        amount: transactions[0]?.amount,
        risk_score: transactions[0]?.risk_score,
        final_score: transactions[0]?.final_score,
        risk_level: transactions[0]?.risk_level
    });
    
    console.log('📋 Rendering', transactions.length, 'transactions');
    
    tbody.innerHTML = transactions.map(tx => {
        // Handle different possible data structures
        let riskScore = 0;
        let riskLevel = 'low';
        
        if (tx.final_score !== undefined) {
            // New analysis format (0-1 scale)
            riskScore = Math.min(tx.final_score, 1.0);
            riskLevel = tx.risk_level || getRiskLevelFromScore(riskScore);
        } else if (tx.risk_score !== undefined) {
            // Database format (0-100 scale)
            riskScore = Math.min(tx.risk_score / 100, 1.0);
            riskLevel = getRiskLevelFromScore(riskScore);
        }
        
        const transactionId = tx.transaction_id || tx.id || 'N/A';
        const amount = tx.amount || 0;
        const merchant = tx.merchant || 'N/A';
        const deviceType = tx.device_type || 'N/A';
        const timestamp = tx.timestamp || 'N/A';
        
        return `
            <tr class="transaction-row" style="cursor: pointer;" onclick="showTransactionDetails(${JSON.stringify(tx).replace(/"/g, '&quot;')})">
                <td><code>${transactionId}</code></td>
                <td><strong>$${amount.toFixed(2)}</strong></td>
                <td>${merchant}</td>
                <td><span class="badge bg-secondary">${deviceType}</span></td>
                <td>
                    <div class="progress" style="height: 20px;">
                        <div class="progress-bar ${getRiskColor(riskLevel)}" 
                             style="width: ${Math.round(riskScore * 100)}%">
                            ${Math.round(riskScore * 100)}%
                        </div>
                    </div>
                </td>
                <td>
                    <span class="badge ${getRiskBadgeClass(riskLevel)}">
                        ${riskLevel.toUpperCase()}
                    </span>
                </td>
                <td>${timestamp !== 'N/A' ? new Date(timestamp).toLocaleString() : 'N/A'}</td>
            </tr>
        `;
    }).join('');
    
    // Update transaction count
    const countBadge = document.getElementById('transactionCount');
    if (countBadge) {
        countBadge.textContent = totalTransactions.toLocaleString();
    }
}

/**
 * Load insights for the insights tab
 */
async function loadInsights() {
    try {
        console.log('🧠 Loading insights...');
        
        // Load risk patterns
        const patternsResponse = await fetch(`${API_BASE_URL}/risk-summary`);
        if (patternsResponse.ok) {
            const patterns = await patternsResponse.json();
            updateRiskPatterns(patterns);
        }
        
        // Load time analysis
        const timeResponse = await fetch(`${API_BASE_URL}/transactions?per_page=1000`);
        if (timeResponse.ok) {
            const timeData = await timeResponse.json();
            updateTimeAnalysis(timeData.transactions);
        }
        
    } catch (error) {
        console.error('Error loading insights:', error);
        showAlert('Error loading insights', 'warning');
    }
}

/**
 * Update risk patterns in insights tab
 */
function updateRiskPatterns(patterns) {
    const container = document.getElementById('riskPatterns');
    if (!container) return;
    
    console.log('🧠 Updating risk patterns with:', patterns);
    
    // Handle both old and new API response structures
    const stats = patterns.overall_stats || patterns;
    
    container.innerHTML = `
        <div class="row">
            <div class="col-md-6">
                <h6>Risk Distribution</h6>
                <ul class="list-unstyled">
                    <li><strong>Low Risk:</strong> ${stats.low_risk_count || 0} transactions</li>
                    <li><strong>Medium Risk:</strong> ${stats.medium_risk_count || 0} transactions</li>
                    <li><strong>High Risk:</strong> ${stats.high_risk_count || 0} transactions</li>
                </ul>
            </div>
            <div class="col-md-6">
                <h6>Key Metrics</h6>
                <ul class="list-unstyled">
                    <li><strong>Fraud Rate:</strong> ${(stats.fraud_rate || 0).toFixed(2)}%</li>
                    <li><strong>Avg Risk Score:</strong> ${Math.round((stats.average_risk_score || 0) * 100)}%</li>
                    <li><strong>Total Transactions:</strong> ${stats.total_transactions?.toLocaleString() || 0}</li>
                </ul>
            </div>
        </div>
    `;
}

/**
 * Update time analysis in insights tab
 */
function updateTimeAnalysis(transactions) {
    const container = document.getElementById('timeAnalysis');
    if (!container) return;
    
    // Group transactions by hour
    const hourlyData = {};
    transactions.forEach(tx => {
        if (tx.timestamp) {
            const hour = new Date(tx.timestamp).getHours();
            hourlyData[hour] = (hourlyData[hour] || 0) + 1;
        }
    });
    
    const peakHour = Object.entries(hourlyData).reduce((a, b) => hourlyData[a] > hourlyData[b] ? a : b, 0);
    
    container.innerHTML = `
        <div class="row">
            <div class="col-md-6">
                <h6>Peak Activity</h6>
                <p><strong>Busiest Hour:</strong> ${peakHour}:00 (${hourlyData[peakHour] || 0} transactions)</p>
                <p><strong>Lowest Activity:</strong> ${Object.entries(hourlyData).reduce((a, b) => hourlyData[a] < hourlyData[b] ? a : b, 0)}:00</p>
            </div>
            <div class="col-md-6">
                <h6>Time Patterns</h6>
                <p><strong>Business Hours:</strong> Most transactions during 9 AM - 6 PM</p>
                <p><strong>Weekend vs Weekday:</strong> ${transactions.length > 0 ? 'Analyzing...' : 'No data'}</p>
            </div>
        </div>
    `;
}

/**
 * Initialize charts
 */
function initializeCharts() {
    console.log('📊 Initializing charts...');
    console.log('📊 Chart.js available:', typeof Chart !== 'undefined');
    
    // Risk Distribution Chart
    const riskCtx = document.getElementById('riskDistributionChart');
    console.log('📊 Risk distribution canvas:', riskCtx);
    if (riskCtx) {
        riskDistributionChart = new Chart(riskCtx, {
            type: 'doughnut',
            data: {
                labels: ['Low Risk', 'Medium Risk', 'High Risk'],
                datasets: [{
                    data: [0, 0, 0],
                    backgroundColor: ['#28a745', '#ffc107', '#dc3545'],
                    borderWidth: 2,
                    borderColor: '#fff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }
    
    // Risk Trends Chart
    const trendsCtx = document.getElementById('riskTrendsChart');
    console.log('📊 Risk trends canvas:', trendsCtx);
    if (trendsCtx) {
        riskTrendsChart = new Chart(trendsCtx, {
            type: 'line',
            data: {
                labels: ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00'],
                datasets: [{
                    label: 'Risk Score',
                    data: [0, 0, 0, 0, 0, 0],
                    borderColor: '#3498db',
                    backgroundColor: 'rgba(52, 152, 219, 0.1)',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100
                    }
                }
            }
        });
    }
}

/**
 * Update charts with new data
 */
function updateCharts(transactions) {
    if (!transactions || transactions.length === 0) return;
    
    console.log('📊 Updating charts with', transactions.length, 'transactions');
    
    // Update risk distribution chart
    if (riskDistributionChart) {
        // Calculate risk levels from risk_score (0-100 scale)
        const lowRisk = transactions.filter(tx => (tx.risk_score || 0) < 30).length;
        const mediumRisk = transactions.filter(tx => (tx.risk_score || 0) >= 30 && (tx.risk_score || 0) < 70).length;
        const highRisk = transactions.filter(tx => (tx.risk_score || 0) >= 70).length;
        
        console.log('📊 Risk distribution:', { lowRisk, mediumRisk, highRisk });
        
        riskDistributionChart.data.datasets[0].data = [lowRisk, mediumRisk, highRisk];
        riskDistributionChart.update();
    }
    
    // Update risk trends chart
    if (riskTrendsChart) {
        // Group by hour and calculate average risk
        const hourlyRisk = {};
        transactions.forEach(tx => {
            if (tx.timestamp) {
                const hour = new Date(tx.timestamp).getHours();
                if (!hourlyRisk[hour]) hourlyRisk[hour] = [];
                // Use risk_score (0-100) and convert to 0-1 scale for consistency
                const normalizedScore = (tx.risk_score || 0) / 100;
                hourlyRisk[hour].push(normalizedScore);
            }
        });
        
        const labels = Object.keys(hourlyRisk).sort((a, b) => parseInt(a) - parseInt(b));
        const data = labels.map(hour => {
            const scores = hourlyRisk[hour];
            const avgScore = scores.reduce((a, b) => a + b, 0) / scores.length;
            return Math.round(avgScore * 100);
        });
        
        console.log('📊 Risk trends:', { labels, data });
        
        riskTrendsChart.data.labels = labels.map(hour => `${hour}:00`);
        riskTrendsChart.data.datasets[0].data = data;
        riskTrendsChart.update();
    }
}

/**
 * Fill sample data for testing
 */
function fillSampleData() {
    const form = document.getElementById('riskAnalysisForm');
    if (!form) return;
    
    // Generate random sample data
    const merchants = ['Amazon', 'Walmart', 'Starbucks', 'LuxuryHotel', 'GasStation'];
    const devices = ['web-chrome', 'mobile-ios', 'mobile-android', 'tablet'];
    
    const randomMerchant = merchants[Math.floor(Math.random() * merchants.length)];
    const randomDevice = devices[Math.floor(Math.random() * devices.length)];
    const randomAmount = Math.random() * 1000 + 10; // $10 - $1010
    
    // Fill form fields
    form.querySelector('#amount').value = randomAmount.toFixed(2);
    form.querySelector('#merchant').value = randomMerchant;
    form.querySelector('#deviceType').value = randomDevice;
    form.querySelector('#timestamp').value = new Date().toISOString().slice(0, 16);
    
    showAlert('Sample data filled! Click "Analyze Risk" to test.', 'info');
}

/**
 * Show loading spinner
 */
function showLoading(show) {
    const spinner = document.getElementById('loadingSpinner');
    if (spinner) {
        spinner.style.display = show ? 'block' : 'none';
    }
}

/**
 * Generate AI-powered explanation using external AI service
 */
async function generateAIExplanation(analysis) {
    try {
        // This would integrate with OpenAI GPT API or similar service
        // For now, we'll create intelligent explanations based on the data
        
        const riskLevel = analysis.risk_level;
        const mlScore = analysis.ml_prediction;
        const ruleScore = analysis.rule_based_score;
        const finalScore = analysis.final_score;
        
        let explanation = "";
        let preventionTips = "";
        
        if (riskLevel === 'high') {
            explanation = `🚨 This transaction has been flagged as HIGH RISK (${Math.round(finalScore * 100)}%) due to multiple concerning factors. The ML model detected suspicious patterns with ${Math.round(mlScore * 100)}% confidence, while rule-based analysis identified ${Math.round(ruleScore * 100)}% risk factors. This combination of high ML confidence and multiple risk indicators suggests this transaction requires immediate attention.`;
            
            preventionTips = `
                <strong>🚨 Immediate Actions Required:</strong>
                <ul>
                    <li><strong>Hold Transaction:</strong> Immediately suspend processing and flag for manual review</li>
                    <li><strong>Customer Verification:</strong> Contact customer to verify transaction details and identity</li>
                    <li><strong>Enhanced Authentication:</strong> Request additional identity documents or step-up verification</li>
                    <li><strong>Account Monitoring:</strong> Consider temporary account restrictions and enhanced monitoring</li>
                </ul>
                <strong>🔍 Investigation Steps:</strong>
                <ul>
                    <li><strong>Pattern Analysis:</strong> Review customer's recent activity for similar suspicious patterns</li>
                    <li><strong>Device History:</strong> Check if this device has been associated with previous fraud attempts</li>
                    <li><strong>Location Verification:</strong> Verify if the transaction location matches customer's usual patterns</li>
                    <li><strong>Real-time Alerts:</strong> Enable immediate notifications for similar future transactions</li>
                </ul>
            `;
        } else if (riskLevel === 'medium') {
            explanation = `⚠️ This transaction shows MEDIUM RISK (${Math.round(finalScore * 100)}%) and requires enhanced monitoring. While not immediately suspicious, several risk factors suggest this transaction should be watched closely. The ML model and rule-based analysis both indicate elevated risk that warrants attention.`;
            
            preventionTips = `
                <strong>⚠️ Enhanced Monitoring Required:</strong>
                <ul>
                    <li><strong>Transaction Limits:</strong> Consider implementing lower transaction limits for this customer</li>
                    <li><strong>Step-up Authentication:</strong> Enable additional verification for similar transactions</li>
                    <li><strong>Pattern Monitoring:</strong> Watch for changes in customer behavior patterns</li>
                    <li><strong>Real-time Alerts:</strong> Set up notifications for similar risk transactions</li>
                </ul>
                <strong>🔒 Prevention Measures:</strong>
                <ul>
                    <li><strong>Customer Education:</strong> Provide security tips and fraud prevention guidance</li>
                    <li><strong>Regular Reviews:</strong> Schedule periodic risk assessment reviews</li>
                    <li><strong>Fraud Detection:</strong> Keep fraud detection systems active and updated</li>
                </ul>
            `;
        } else {
            explanation = `✅ This transaction appears to be LOW RISK (${Math.round(finalScore * 100)}%) and follows normal patterns. The ML model and rule-based analysis both indicate this is likely a legitimate transaction that can be processed normally.`;
            
            preventionTips = `
                <strong>✅ Standard Processing:</strong>
                <ul>
                    <li><strong>Normal Processing:</strong> Process transaction using standard security protocols</li>
                    <li><strong>Routine Monitoring:</strong> Continue standard fraud detection monitoring</li>
                    <li><strong>Security Maintenance:</strong> Keep all security measures active and updated</li>
                </ul>
                <strong>🛡️ Ongoing Protection:</strong>
                <ul>
                    <li><strong>Fraud Detection:</strong> Maintain active fraud detection systems</li>
                    <li><strong>Customer Education:</strong> Continue providing security awareness information</li>
                    <li><strong>Pattern Monitoring:</strong> Watch for sudden changes in customer behavior</li>
                </ul>
            `;
        }
        
        return { explanation, preventionTips };
        
    } catch (error) {
        console.error('Error generating AI explanation:', error);
        return {
            explanation: "Unable to generate AI explanation at this time.",
            preventionTips: "Please review the technical analysis above for risk assessment."
        };
    }
}

/**
 * Show alert message
 */
function showAlert(message, type = 'info') {
    const container = document.getElementById('alertContainer');
    if (!container) return;
    
    const alertId = 'alert-' + Date.now();
    const alertHtml = `
        <div class="alert alert-${type} alert-dismissible fade show" id="${alertId}" role="alert">
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;
    
    container.insertAdjacentHTML('beforeend', alertHtml);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        const alert = document.getElementById(alertId);
        if (alert) {
            alert.remove();
        }
    }, 5000);
}

/**
 * Get risk color for progress bars
 */
function getRiskColor(riskLevel) {
    switch(riskLevel) {
        case 'low': return 'bg-success';
        case 'medium': return 'bg-warning';
        case 'high': return 'bg-danger';
        default: return 'bg-secondary';
    }
}

/**
 * Get risk badge class
 */
function getRiskBadgeClass(riskLevel) {
    switch(riskLevel) {
        case 'low': return 'bg-success';
        case 'medium': return 'bg-warning';
        case 'high': return 'bg-danger';
        default: return 'bg-secondary';
    }
}

/**
 * Get risk level from score (fallback when risk_level is missing)
 */
function getRiskLevelFromScore(score) {
    // Handle both 0-1 and 0-100 scales
    const normalizedScore = score > 1 ? score / 100 : score;
    
    if (normalizedScore < 0.3) return 'low';
    if (normalizedScore < 0.7) return 'medium';
    return 'high';
}

/**
 * Show transaction details modal
 */
function showTransactionDetails(transaction) {
    console.log('🔍 Showing transaction details:', transaction);
    
    // Populate transaction info
    const transactionInfo = document.getElementById('transactionInfo');
    if (transactionInfo) {
        transactionInfo.innerHTML = `
            <div class="mb-2"><strong>ID:</strong> ${transaction.transaction_id || transaction.id || 'N/A'}</div>
            <div class="mb-2"><strong>Amount:</strong> $${(transaction.amount || 0).toFixed(2)}</div>
            <div class="mb-2"><strong>Merchant:</strong> ${transaction.merchant || 'N/A'}</div>
            <div class="mb-2"><strong>Device:</strong> ${transaction.device_type || 'N/A'}</div>
            <div class="mb-2"><strong>Location:</strong> (${transaction.latitude || 'N/A'}, ${transaction.longitude || 'N/A'})</div>
            <div class="mb-2"><strong>Time:</strong> ${transaction.timestamp ? new Date(transaction.timestamp).toLocaleString() : 'N/A'}</div>
        `;
    }
    
    // Populate risk analysis
    const riskAnalysis = document.getElementById('transactionRiskAnalysis');
    if (riskAnalysis) {
        const riskScore = Math.min(transaction.final_score || transaction.risk_score || 0, 1.0);
        const riskLevel = transaction.risk_level || getRiskLevelFromScore(riskScore);
        
        riskAnalysis.innerHTML = `
            <div class="mb-3">
                <div class="d-flex justify-content-between">
                    <span><strong>Final Risk Score:</strong></span>
                    <span class="badge ${getRiskBadgeClass(riskLevel)} fs-6">${Math.round(riskScore * 100)}%</span>
                </div>
                <div class="progress mt-2" style="height: 25px;">
                    <div class="progress-bar ${getRiskColor(riskLevel)}" 
                         style="width: ${Math.round(riskScore * 100)}%">
                        ${riskLevel.toUpperCase()}
                    </div>
                </div>
            </div>
            
            <div class="mb-2">
                <strong>ML Prediction:</strong> ${Math.round((transaction.ml_prediction || 0) * 100)}%
            </div>
            <div class="mb-2">
                <strong>Rule-based Score:</strong> ${Math.round((transaction.rule_based_score || 0) * 100)}%
            </div>
        `;
    }
    
    // Populate explanations (if available)
    if (transaction.ml_reasons) {
        const mlReasons = document.getElementById('modalMlReasons');
        if (mlReasons) {
            mlReasons.innerHTML = transaction.ml_reasons.map(reason => 
                `<div class="explanation-item"><i class="fas fa-chevron-right me-2"></i>${reason}</div>`
            ).join('');
        }
    }
    
    if (transaction.rule_reasons) {
        const ruleReasons = document.getElementById('modalRuleReasons');
        if (ruleReasons) {
            ruleReasons.innerHTML = transaction.rule_reasons.map(reason => 
                `<div class="explanation-item"><i class="fas fa-chevron-right me-2"></i>${reason}</div>`
            ).join('');
        }
    }
    
    if (transaction.final_reasoning) {
        const finalReasoning = document.getElementById('modalFinalReasoning');
        if (finalReasoning) {
            finalReasoning.innerHTML = transaction.final_reasoning.map(reason => 
                `<div class="explanation-item"><i class="fas fa-info-circle me-2"></i>${reason}</div>`
            ).join('');
        }
    }
    
    // Show the modal
    const modal = new bootstrap.Modal(document.getElementById('transactionModal'));
    modal.show();
}

/**
 * Re-analyze the current transaction
 */
function reanalyzeTransaction() {
    // Get transaction data from the modal
    const modal = document.getElementById('transactionModal');
    const transactionInfo = document.getElementById('transactionInfo');
    
    if (transactionInfo) {
        // Extract amount and merchant from the displayed info
        const amountText = transactionInfo.textContent.match(/Amount:\s*\$([\d.]+)/);
        const merchantText = transactionInfo.textContent.match(/Merchant:\s*([^\n]+)/);
        
        if (amountText && merchantText) {
            const amount = parseFloat(amountText[1]);
            const merchant = merchantText[1].trim();
            
            // Fill the analysis form
            const form = document.getElementById('riskAnalysisForm');
            if (form) {
                form.querySelector('#amount').value = amount;
                form.querySelector('#merchant').value = merchant;
                
                // Close modal and switch to analyze tab
                const modalInstance = bootstrap.Modal.getInstance(modal);
                modalInstance.hide();
                
                // Switch to analyze tab
                const analyzeTab = document.getElementById('analyze-tab');
                if (analyzeTab) {
                    const tab = new bootstrap.Tab(analyzeTab);
                    tab.show();
                }
                
                showAlert('Transaction data filled! Click "Analyze Risk" to re-analyze.', 'info');
            }
        }
    }
}

/**
 * Switch to transactions tab to view the newly analyzed transaction
 */
function viewInTransactions() {
    // Switch to transactions tab
    const transactionsTab = document.getElementById('transactions-tab');
    if (transactionsTab) {
        const tab = new bootstrap.Tab(transactionsTab);
        tab.show();
        
        // Go to first page to see newest transactions
        currentPage = 1;
        loadTransactions();
        
        showAlert('Switched to Transactions tab. Your new analysis should appear at the top!', 'info');
    }
}

/**
 * Update pagination controls
 */
function updatePaginationControls() {
    const prevBtn = document.getElementById('prevBtn');
    const nextBtn = document.getElementById('nextBtn');
    const pageInput = document.getElementById('pageInput');
    const totalPagesSpan = document.getElementById('totalPages');
    const paginationInfo = document.getElementById('paginationInfo');
    
    if (prevBtn) prevBtn.disabled = currentPage <= 1;
    if (nextBtn) nextBtn.disabled = currentPage >= totalPages;
    if (pageInput) pageInput.value = currentPage;
    if (totalPagesSpan) totalPagesSpan.textContent = totalPages;
    if (paginationInfo) {
        const start = (currentPage - 1) * perPage + 1;
        const end = Math.min(currentPage * perPage, totalTransactions);
        paginationInfo.textContent = `Showing ${start}-${end} of ${totalTransactions} transactions`;
    }
}

/**
 * Change transactions per page
 */
function changePerPage() {
    const select = document.getElementById('perPageSelect');
    if (select) {
        perPage = parseInt(select.value);
        currentPage = 1; // Reset to first page
        loadTransactions();
    }
}

/**
 * Go to previous page
 */
function previousPage() {
    if (currentPage > 1) {
        currentPage--;
        loadTransactions();
    }
}

/**
 * Go to next page
 */
function nextPage() {
    if (currentPage < totalPages) {
        currentPage++;
        loadTransactions();
    }
}

/**
 * Go to specific page
 */
function goToPage() {
    const pageInput = document.getElementById('pageInput');
    if (pageInput) {
        const page = parseInt(pageInput.value);
        if (page >= 1 && page <= totalPages) {
            currentPage = page;
            loadTransactions();
        } else {
            showAlert(`Please enter a page number between 1 and ${totalPages}`, 'warning');
        }
    }
}

/**
 * View the latest transactions (first page)
 */
function viewLatestTransactions() {
    currentPage = 1;
    loadTransactions();
    showAlert('Showing latest transactions!', 'info');
}
