
// =============================================================================
// 1. Application State & Chart Instances
// =============================================================================
let trendChartInstance = null;
let categoryChartInstance = null;
let forecastChartInstance = null;

// =============================================================================
// 2. DOM Helpers & Formatting Utilities
// =============================================================================
/**
 * Shorthand helper for document.getElementById
 * @param {string} id - HTML element ID
 * @returns {HTMLElement|null}
 */
const getEl = (id) => document.getElementById(id);

/**
 * Format numbers according to Indian locale standard (en-IN)
 * @param {number|string} value - Numerical value
 * @returns {string} Formatted number string
 */
const formatNumber = (value) => Number(value || 0).toLocaleString('en-IN');

/**
 * Build URLSearchParams from active user filter inputs
 * @returns {URLSearchParams} Query parameters object
 */
const getFilterParams = () => {
  return new URLSearchParams({
    state: getEl('state').value,
    district: getEl('district').value,
    category: getEl('category').value,
  });
};

/**
 * Standardized asynchronous JSON fetch helper with error handling
 * @param {string} url - API Endpoint
 * @returns {Promise<any>} Parsed JSON response
 */
async function fetchJSON(url) {
  const response = await fetch(url);
  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.error || 'Failed to load analytics data from server');
  }

  return data;
}

// =============================================================================
// 3. Chart Configuration Helpers
// =============================================================================
/**
 * Shared base Chart.js configuration options
 * @returns {object} Chart.js options dictionary
 */
function getBaseChartOptions() {
  return {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
    },
    scales: {
      x: {
        grid: { display: false },
        ticks: { color: '#8792a5', font: { size: 10 } },
      },
      y: {
        grid: { color: '#edf0f5' },
        ticks: { color: '#8792a5', font: { size: 10 } },
      },
    },
  };
}

// =============================================================================
// 4. Data Rendering Functions
// =============================================================================

/**
 * Update top-level KPI overview stat cards
 * @param {object} summary - Summary metrics payload
 */
function renderKPIs(summary) {
  getEl('totalCrimes').textContent = formatNumber(summary.total_crimes);
  getEl('yearRange').textContent = `${summary.first_year}–${summary.last_year}`;
  getEl('stateCount').textContent = formatNumber(summary.states);
  getEl('districtCount').textContent = formatNumber(summary.districts);
  getEl('recordCount').textContent = `${formatNumber(summary.records)} district-year records`;
}

/**
 * Render the yearly crime trends line chart
 * @param {Array} series - Yearly time series data
 */
function renderTrendChart(series) {
  const labels = series.map((item) => item.year);
  const values = series.map((item) => item.count);

  if (trendChartInstance) {
    trendChartInstance.destroy();
  }

  const canvas = getEl('trendChart');
  trendChartInstance = new Chart(canvas, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'Recorded Crimes',
          data: values,
          borderColor: '#4666e8',
          backgroundColor: '#4666e81a',
          fill: true,
          tension: 0.3,
          pointRadius: 3,
          borderWidth: 2,
        },
      ],
    },
    options: getBaseChartOptions(),
  });

  // Display textual YoY trend insight
  const lastItem = series.at(-1);
  const prevItem = series.at(-2);
  const categoryLabel = getEl('category').selectedOptions[0]?.text || 'Crime';

  if (lastItem && prevItem) {
    const pctChange = lastItem.change_pct === null ? '—' : `${lastItem.change_pct}%`;
    const directionWord = lastItem.change >= 0 ? 'increased' : 'decreased';
    const absDiff = formatNumber(Math.abs(lastItem.change || 0));

    getEl('trendInsight').textContent =
      `${categoryLabel}: ${pctChange} change in ${lastItem.year} compared with ${prevItem.year}. ` +
      `Recorded count ${directionWord} by ${absDiff}.`;
  } else if (lastItem) {
    getEl('trendInsight').textContent = `Latest available year: ${lastItem.year}.`;
  } else {
    getEl('trendInsight').textContent = 'No records found for this selection.';
  }
}

/**
 * Render the crime category breakdown horizontal bar chart
 * @param {Array} items - List of crime categories with aggregated counts
 */
function renderCategoryChart(items) {
  if (categoryChartInstance) {
    categoryChartInstance.destroy();
  }

  const labels = items.map((item) => item.category);
  const values = items.map((item) => item.count);
  const palette = ['#5672e8', '#8b6bdc', '#20a58a', '#e5a14a', '#e07878'];

  const canvas = getEl('categoryChart');
  categoryChartInstance = new Chart(canvas, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [
        {
          data: values,
          backgroundColor: palette,
          borderRadius: 5,
        },
      ],
    },
    options: {
      ...getBaseChartOptions(),
      indexAxis: 'y',
    },
  });
}

/**
 * Render machine learning forecast projections and evaluation metrics
 * @param {object} forecastData - Forecasting response payload
 */
function renderForecast(forecastData) {
  // Update metadata indicators
  getEl('direction').textContent = forecastData.direction;
  const changePrefix = forecastData.annual_change > 0 ? '+' : '';
  getEl('annualChange').textContent = `${changePrefix}${formatNumber(forecastData.annual_change)} cases/year`;
  getEl('mae').textContent = `${formatNumber(forecastData.mae)} cases`;
  getEl('rmse').textContent = `${formatNumber(forecastData.rmse)} cases`;
  getEl('forecastNote').textContent = forecastData.note;

  // Build continuous timeline labels and values
  const history = forecastData.history;
  const forecast = forecastData.forecast;

  const allLabels = [...history.map((h) => h.year), ...forecast.map((f) => f.year)];
  const actualValues = [...history.map((h) => h.count), ...forecast.map(() => null)];
  const predictedValues = [
    ...history.map(() => null),
    history.at(-1).count,
    ...forecast.map((f) => f.predicted_count),
  ];

  if (forecastChartInstance) {
    forecastChartInstance.destroy();
  }

  const canvas = getEl('forecastChart');
  forecastChartInstance = new Chart(canvas, {
    type: 'line',
    data: {
      labels: allLabels,
      datasets: [
        {
          label: 'Historical',
          data: actualValues,
          borderColor: '#4566df',
          backgroundColor: '#4566df15',
          fill: false,
          tension: 0.25,
          pointRadius: 3,
        },
        {
          label: 'Forecast',
          data: predictedValues,
          borderColor: '#8b63d8',
          borderDash: [6, 5],
          backgroundColor: '#8b63d815',
          fill: false,
          tension: 0.25,
          pointRadius: 3,
        },
      ],
    },
    options: {
      ...getBaseChartOptions(),
      plugins: {
        legend: {
          display: true,
          position: 'bottom',
        },
      },
    },
  });
}

/**
 * Fetch and render the geographic ranking leaderboard
 */
async function loadPlaces() {
  const params = new URLSearchParams({
    state: getEl('state').value,
    category: getEl('category').value,
    year: getEl('rankingYear').value,
  });

  const response = await fetchJSON(`/api/places?${params}`);
  const items = response.items || [];
  const maxCount = Math.max(1, ...items.map((item) => item.count));

  if (!items.length) {
    getEl('rankList').innerHTML = '<p class="hint">No records available for this filter.</p>';
    return;
  }

  getEl('rankList').innerHTML = items
    .map((item, index) => {
      const percentage = (item.count / maxCount) * 100;
      return `
        <div class="rank-row">
          <div class="rank-num">${index + 1}</div>
          <div>
            <div class="rank-name">${item.name}</div>
            <div class="rank-track">
              <div class="rank-fill" style="width: ${percentage}%"></div>
            </div>
          </div>
          <div class="rank-count">${formatNumber(item.count)}</div>
        </div>
      `;
    })
    .join('');
}

// =============================================================================
// 5. Controller & Initialization
// =============================================================================

/**
 * Update the district dropdown options based on the currently selected state
 */
async function updateDistricts() {
  const state = getEl('state').value;
  const data = await fetchJSON(`/api/options?state=${encodeURIComponent(state)}`);

  getEl('district').innerHTML = data.districts
    .map((district) => `<option>${district}</option>`)
    .join('');
}

/**
 * Refresh all dashboard widgets and charts according to current filter inputs
 */
async function refreshDashboard() {
  try {
    const params = getFilterParams();
    const horizon = getEl('horizon').value;

    const [summaryData, trendsData, breakdownData, forecastData] = await Promise.all([
      fetchJSON(`/api/summary?${params}`),
      fetchJSON(`/api/trends?${params}`),
      fetchJSON(`/api/breakdown?${params}`),
      fetchJSON(`/api/forecast?${params}&horizon=${horizon}`),
    ]);

    renderKPIs(summaryData);
    renderTrendChart(trendsData.series);
    renderCategoryChart(breakdownData.items);
    renderForecast(forecastData);
    await loadPlaces();
  } catch (error) {
    console.error('Error refreshing dashboard:', error);
    alert(error.message);
  }
}

/**
 * Initialize application options, filters, and initial dashboard view
 */
async function initDashboard() {
  const options = await fetchJSON('/api/options');

  // Populate States
  getEl('state').innerHTML = options.states
    .map((state) => `<option>${state}</option>`)
    .join('');

  // Populate Ranking Year Selector
  const yearOptions = options.years
    .map((year) => `<option>${year}</option>`)
    .join('');
  getEl('rankingYear').innerHTML = `<option value="all">All available years</option>${yearOptions}`;

  // Update Data Coverage Badge
  const firstYear = options.years[0];
  const lastYear = options.years[options.years.length - 1];
  const totalStates = options.states.length - 1;
  getEl('coverage').textContent = `${firstYear}–${lastYear} · ${totalStates} States/UTs`;

  // Populate dependent districts and trigger initial render
  await updateDistricts();
  await refreshDashboard();
}

// =============================================================================
// 6. Event Listeners & Startup
// =============================================================================
getEl('state').addEventListener('change', async () => {
  await updateDistricts();
});

getEl('apply').addEventListener('click', refreshDashboard);
getEl('rankingYear').addEventListener('change', loadPlaces);

// Bootstrap
initDashboard().catch((error) => {
  console.error('Initialization error:', error);
  alert(error.message);
});
