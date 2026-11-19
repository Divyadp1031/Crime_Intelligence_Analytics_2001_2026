
// =============================================================================
// 1. Application State & Chart Instances
// =============================================================================
let trendChartInstance = null;
let categoryChartInstance = null;
let forecastChartInstance = null;
let travelChartInstance = null;
let manpowerChartInstance = null;
let mapRiskData = null;
let activeMapStateCode = null;
let indiaSvgLoaded = false;
let districtRiskData = null;

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
// 4a. India Crime Risk Map Rendering & Interactivity
// =============================================================================

/**
 * Position the floating map tooltip cleanly within viewport bounds
 * @param {MouseEvent} e - Mouse event
 * @param {HTMLElement} tooltip - Floating tooltip element
 */
function positionMapTooltip(e, tooltip) {
  const pad = 14;
  let x = e.clientX + pad;
  let y = e.clientY + pad;

  const tooltipWidth = tooltip.offsetWidth || 190;
  const tooltipHeight = tooltip.offsetHeight || 110;

  if (x + tooltipWidth > window.innerWidth - 10) {
    x = e.clientX - tooltipWidth - pad;
  }
  if (y + tooltipHeight > window.innerHeight - 10) {
    y = e.clientY - tooltipHeight - pad;
  }

  tooltip.style.left = `${x}px`;
  tooltip.style.top = `${y}px`;
}

/**
 * Load and inject the vector SVG map of India, binding interactive mouse events
 */
async function loadIndiaSVGMap() {
  if (indiaSvgLoaded) return;
  const container = getEl('indiaMapContainer');
  if (!container) return;

  try {
    const res = await fetch('/static/india.svg');
    if (!res.ok) {
      throw new Error(`Failed to load vector map (HTTP ${res.status})`);
    }
    const svgText = await res.text();
    container.innerHTML = svgText;
    indiaSvgLoaded = true;

    const svg = container.querySelector('svg');
    if (svg) {
      svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
    }

    const tooltip = getEl('mapTooltip');
    const paths = container.querySelectorAll('path[id]');

    paths.forEach((path) => {
      const code = path.id.toLowerCase();

      path.addEventListener('mouseenter', (e) => {
        const info = mapRiskData?.states_map?.[code];
        if (!info) return;

        highlightState(code, false);

        if (tooltip) {
          tooltip.innerHTML = `
            <div class="map-tooltip-top">
              <b>${info.state}</b>
              <span class="tier-badge ${info.risk_badge_class}" style="font-size: 9px; padding: 2px 6px;">${info.risk_label}</span>
            </div>
            <div style="font-size: 11px; margin-top: 4px; line-height: 1.5;">
              <div>Reported Cases: <b>${formatNumber(info.count)}</b></div>
              <div>National Share: <b>${info.share_pct}%</b></div>
              <div>All-India Rank: <b>#${info.rank} of 36</b></div>
              <div>Top Hotspot: <b>${info.top_district}</b> (${formatNumber(info.top_district_count)})</div>
            </div>
          `;
          tooltip.style.display = 'block';
          positionMapTooltip(e, tooltip);
        }
      });

      path.addEventListener('mousemove', (e) => {
        if (tooltip && tooltip.style.display === 'block') {
          positionMapTooltip(e, tooltip);
        }
      });

      path.addEventListener('mouseleave', () => {
        if (tooltip) {
          tooltip.style.display = 'none';
        }
        if (activeMapStateCode) {
          highlightState(activeMapStateCode, false);
        }
      });

      path.addEventListener('click', () => {
        activeMapStateCode = code;
        highlightState(code, true);

        // Highlight and scroll to matching row in leaderboard
        const row = getEl('mapLeaderboardList')?.querySelector(`.leaderboard-row[data-code="${code}"]`);
        if (row) {
          row.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
          row.style.background = '#eef2fb';
          setTimeout(() => {
            row.style.background = '';
          }, 1500);
        }

        // Sync with District Crime Risk Predictor
        const stateInfo = mapRiskData?.states_map?.[code];
        if (stateInfo && getEl('districtRiskState')) {
          const matchOption = Array.from(getEl('districtRiskState').options).find(
            (opt) => opt.value.toUpperCase() === stateInfo.state.toUpperCase()
          );
          if (matchOption) {
            getEl('districtRiskState').value = matchOption.value;
            updateDistrictRiskDropdown().then(() => fetchDistrictRiskAnalysis());
          }
        }
      });
    });
  } catch (err) {
    console.error('Error loading India SVG:', err);
    container.innerHTML = `<div class="hint" style="padding: 24px; text-align: center;">Unable to load India vector map: ${err.message}</div>`;
  }
}

/**
 * Highlight a state in the spotlight card and optionally on the SVG map
 * @param {string} code - Two-letter state code (e.g. 'mh', 'tn')
 * @param {boolean} persist - Whether this state is firmly selected (clicked)
 */
function highlightState(code, persist = false) {
  if (!mapRiskData || !mapRiskData.states_map) return;
  const info = mapRiskData.states_map[code];
  if (!info) return;

  const nameEl = getEl('spotlightName');
  const badgeEl = getEl('spotlightBadge');
  const countEl = getEl('spotlightCount');
  const labelEl = getEl('spotlightCategoryLabel');
  const shareEl = getEl('spotlightShare');
  const rankEl = getEl('spotlightRank');
  const topDistEl = getEl('spotlightTopDistrict');

  if (nameEl) nameEl.textContent = info.state;
  if (badgeEl) {
    badgeEl.textContent = info.risk_label;
    badgeEl.className = `tier-badge ${info.risk_badge_class}`;
  }
  if (countEl) countEl.textContent = formatNumber(info.count);
  if (labelEl) labelEl.textContent = `Cases in ${mapRiskData.category_label} (${mapRiskData.year})`;
  if (shareEl) shareEl.textContent = `${info.share_pct}% of National Total`;
  if (rankEl) rankEl.textContent = `#${info.rank} of 36 States & UTs`;
  if (topDistEl) topDistEl.textContent = `${info.top_district} (${formatNumber(info.top_district_count)} cases)`;

  if (persist) {
    activeMapStateCode = code;
    const container = getEl('indiaMapContainer');
    if (container) {
      container.querySelectorAll('path.active-state').forEach((p) => p.classList.remove('active-state'));
      const activePath = container.querySelector(`path#${code}`);
      if (activePath) activePath.classList.add('active-state');
    }
  }
}

/**
 * Render the risk levels on the SVG map, update summary pills, and populate ranked leaderboard
 * @param {object} data - India risk map API response payload
 */
function renderIndiaRiskMap(data) {
  mapRiskData = data;

  // 1. Update Category & Year Header
  const catDisplay = getEl('mapCategoryDisplay');
  if (catDisplay) {
    catDisplay.textContent = `${data.category_label} (${data.year})`;
  }

  // 2. Update Risk KPI Pills
  getEl('mapHighCount').textContent = `${data.high_count} States`;
  getEl('mapMedCount').textContent = `${data.medium_count} States`;
  getEl('mapLowCount').textContent = `${data.low_count} States`;

  // 3. Color SVG Map Paths
  const container = getEl('indiaMapContainer');
  if (container) {
    const paths = container.querySelectorAll('path[id]');
    paths.forEach((path) => {
      const code = path.id.toLowerCase();
      const info = data.states_map[code];
      if (info) {
        path.style.fill = info.risk_color;
        path.setAttribute('fill', info.risk_color);
      } else {
        path.style.fill = '#e2e8f0';
        path.setAttribute('fill', '#e2e8f0');
      }
    });

    if (activeMapStateCode) {
      container.querySelectorAll('path.active-state').forEach((p) => p.classList.remove('active-state'));
      const activePath = container.querySelector(`path#${activeMapStateCode}`);
      if (activePath) activePath.classList.add('active-state');
    }
  }

  // 4. Update Leaderboard
  const totalCasesEl = getEl('leaderboardTotal');
  if (totalCasesEl) {
    totalCasesEl.textContent = `Total: ${formatNumber(data.total_national_crimes)} Cases`;
  }

  const listEl = getEl('mapLeaderboardList');
  if (listEl) {
    listEl.innerHTML = data.leaderboard
      .map(
        (item) => `
        <div class="leaderboard-row" data-code="${item.code}">
          <div class="leaderboard-left">
            <span class="leaderboard-rank">#${item.rank}</span>
            <span class="legend-dot" style="background-color: ${item.risk_color};"></span>
            <span class="leaderboard-name">${item.state}</span>
          </div>
          <div class="leaderboard-right">
            <span class="leaderboard-count">${formatNumber(item.count)}</span>
            <span class="tier-badge ${item.risk_badge_class}">${item.risk_level}</span>
          </div>
        </div>
      `
      )
      .join('');

    listEl.querySelectorAll('.leaderboard-row').forEach((row) => {
      row.addEventListener('click', () => {
        const code = row.dataset.code;
        highlightState(code, true);
      });
    });
  }

  // 5. Default Spotlight Selection
  if (activeMapStateCode && data.states_map[activeMapStateCode]) {
    highlightState(activeMapStateCode, true);
  } else if (data.leaderboard.length > 0) {
    highlightState(data.leaderboard[0].code, true);
  }
}

/**
 * Fetch risk map payload for selected category and year
 */
async function fetchIndiaRiskMap() {
  try {
    const category = getEl('mapCategory').value;
    const year = getEl('mapYear').value;
    const data = await fetchJSON(`/api/map/risk?category=${encodeURIComponent(category)}&year=${encodeURIComponent(year)}`);
    renderIndiaRiskMap(data);
  } catch (err) {
    console.error('Error fetching India risk map data:', err);
  }
}

// =============================================================================
// 4a-2. District Crime Risk & Incident Predictor
// =============================================================================

/**
 * Update the district dropdown options based on the chosen state
 * @param {string} [targetDistrict] - Optional district to preselect
 */
async function updateDistrictRiskDropdown(targetDistrict = null) {
  const stateEl = getEl('districtRiskState');
  const districtEl = getEl('districtRiskDistrict');
  if (!stateEl || !districtEl) return;

  const state = stateEl.value;
  try {
    const data = await fetchJSON(`/api/options?state=${encodeURIComponent(state)}`);
    const validDistricts = data.districts.filter((d) => d !== 'All districts');
    districtEl.innerHTML = validDistricts
      .map((district) => `<option value="${district}">${district}</option>`)
      .join('');

    if (targetDistrict && validDistricts.includes(targetDistrict.toUpperCase())) {
      districtEl.value = targetDistrict.toUpperCase();
    } else if (validDistricts.length > 0) {
      districtEl.value = validDistricts[0];
    }
  } catch (err) {
    console.error('Error updating district risk dropdown:', err);
  }
}

/**
 * Fetch district risk prediction and incident assessment from the server
 */
async function fetchDistrictRiskAnalysis() {
  const stateEl = getEl('districtRiskState');
  const districtEl = getEl('districtRiskDistrict');
  const incidentEl = getEl('districtRiskIncident');
  const yearEl = getEl('districtRiskYear');

  if (!stateEl || !districtEl) return;

  const state = stateEl.value;
  const district = districtEl.value;
  const category = incidentEl ? incidentEl.value : 'theft';
  const year = yearEl ? yearEl.value : '2026';

  try {
    const url = `/api/district/risk?state=${encodeURIComponent(state)}&district=${encodeURIComponent(district)}&category=${encodeURIComponent(category)}&year=${encodeURIComponent(year)}`;
    const data = await fetchJSON(url);
    renderDistrictRiskAnalysis(data);
  } catch (err) {
    console.error('Error fetching district risk analysis:', err);
  }
}

/**
 * Render the district risk assessment, primary crime banner, category matrix, and advice
 * @param {object} data - District risk API response payload
 */
function renderDistrictRiskAnalysis(data) {
  districtRiskData = data;

  // 1. Primary Most Common Crime Banner
  const titleEl = getEl('primaryCrimeTitle');
  const descEl = getEl('primaryCrimeDesc');
  const totalEl = getEl('districtTotalCrimes');

  if (titleEl) {
    titleEl.textContent = `${data.primary_crime.label} is the #1 Most Frequent Crime`;
  }
  if (descEl) {
    descEl.innerHTML = `In <b>${data.district} (${data.state})</b>, ${data.primary_crime.label.toLowerCase()} is the leading specific crime concern with <b>${formatNumber(data.primary_crime.count)} projected cases/year</b> (${data.primary_crime.share_pct}% of total reported crimes).`;
  }
  if (totalEl) {
    totalEl.textContent = `${formatNumber(data.total_crimes)} Cases`;
  }

  // 2. Incident Assessment Card
  const inc = data.incident_assessment;
  const incTitleEl = getEl('incidentTitle');
  const incBadgeEl = getEl('incidentBadge');
  const incCasesEl = getEl('incidentCases');
  const incRankEl = getEl('incidentRank');
  const incTrendEl = getEl('incidentTrend');
  const incVsAvgEl = getEl('incidentVsAvg');
  const adviceTitleEl = getEl('adviceTitle');
  const adviceListEl = getEl('incidentAdviceList');
  const helplineEl = getEl('incidentHelpline');

  if (incTitleEl) incTitleEl.textContent = `${inc.category_label} Risk Assessment`;
  if (incBadgeEl) {
    incBadgeEl.textContent = inc.risk_label;
    incBadgeEl.className = `tier-badge ${inc.risk_badge_class}`;
  }
  if (incCasesEl) incCasesEl.textContent = `${formatNumber(inc.count)} cases`;
  if (incRankEl) incRankEl.textContent = `#${inc.rank} of ${inc.total_districts} in State`;
  if (incTrendEl) incTrendEl.textContent = inc.trend;
  if (incVsAvgEl) incVsAvgEl.textContent = inc.vs_state_avg;

  if (adviceTitleEl) adviceTitleEl.textContent = `${inc.advice_title} for ${data.district}`;
  if (adviceListEl) {
    adviceListEl.innerHTML = inc.action_steps
      .map((step) => `<li>${step}</li>`)
      .join('');
  }
  if (helplineEl) helplineEl.textContent = inc.emergency_number;

  // 3. Category Risk Matrix
  const matrixSubEl = getEl('matrixSubtitle');
  if (matrixSubEl) {
    matrixSubEl.textContent = `${data.district}, ${data.state} (${data.year})`;
  }

  const matrixEl = getEl('districtCategoryMatrix');
  if (matrixEl && data.category_matrix) {
    const cats = ['theft', 'rape', 'murder', 'robbery', 'riots', 'total'];
    matrixEl.innerHTML = cats
      .filter((c) => data.category_matrix[c])
      .map((c) => {
        const item = data.category_matrix[c];
        const isSelected = (c === data.selected_category);
        return `
          <div class="cat-risk-row ${isSelected ? 'highlighted' : ''}" data-cat="${item.category}">
            <div class="cat-risk-left">
              <span class="legend-dot" style="background-color: ${item.risk_color};"></span>
              <div>
                <b>${item.category_label}</b>
                <div class="cat-share">${item.share_of_district}% of district crime · ${item.trend}</div>
              </div>
            </div>
            <div class="cat-risk-right">
              <span>${formatNumber(item.count)} cases</span>
              <span class="tier-badge ${item.risk_badge_class}">${item.risk_level}</span>
            </div>
          </div>
        `;
      })
      .join('');

    // Clicking a row in the matrix switches the incident filter
    matrixEl.querySelectorAll('.cat-risk-row').forEach((row) => {
      row.addEventListener('click', () => {
        const cat = row.dataset.cat;
        const incidentSelect = getEl('districtRiskIncident');
        if (incidentSelect && incidentSelect.querySelector(`option[value="${cat}"]`)) {
          incidentSelect.value = cat;
          fetchDistrictRiskAnalysis();
        }
      });
    });
  }

  // 4. Top Districts in State for Selected Crime
  const hotspotContainer = getEl('districtTopInState');
  if (hotspotContainer && data.top_districts_in_state) {
    hotspotContainer.innerHTML = data.top_districts_in_state
      .map(
        (d) => `
        <span class="hotspot-pill ${d.is_current ? 'active-pill' : ''}" data-district="${d.district}">
          #${d.rank} ${d.district} (${formatNumber(d.count)})
        </span>
      `
      )
      .join('');

    // Clicking a hotspot pill switches the district
    hotspotContainer.querySelectorAll('.hotspot-pill').forEach((pill) => {
      pill.addEventListener('click', () => {
        const dName = pill.dataset.district;
        const districtSelect = getEl('districtRiskDistrict');
        if (districtSelect) {
          const match = Array.from(districtSelect.options).find(
            (opt) => opt.value.toUpperCase() === dName.toUpperCase()
          );
          if (match) {
            districtSelect.value = match.value;
            fetchDistrictRiskAnalysis();
          }
        }
      });
    });
  }
}

// =============================================================================
// 4b. Safe Travel & Area Comparison Rendering
// =============================================================================

/**
 * Fetch and update district dropdown options for Area 1 in the Travel Advisor
 */
async function updateTravelDistrictsA() {
  const state = getEl('travelStateA').value;
  const data = await fetchJSON(`/api/options?state=${encodeURIComponent(state)}`);
  getEl('travelDistrictA').innerHTML = data.districts
    .map((district) => `<option value="${district}">${district}</option>`)
    .join('');
}

/**
 * Fetch and update district dropdown options for Area 2 in the Travel Advisor
 */
async function updateTravelDistrictsB() {
  const state = getEl('travelStateB').value;
  const data = await fetchJSON(`/api/options?state=${encodeURIComponent(state)}`);
  getEl('travelDistrictB').innerHTML = data.districts
    .map((district) => `<option value="${district}">${district}</option>`)
    .join('');
}

/**
 * Render the safe travel comparison view, cards, and side-by-side crime chart
 * @param {object} data - Travel comparison API response payload
 */
function renderTravelComparison(data) {
  // 1. Profile banner
  getEl('travelProfileTag').textContent = data.profile.label;
  getEl('travelProfileDesc').textContent = data.profile.description;

  // 2. Verdict Banner
  getEl('travelVerdictTitle').textContent = data.verdict.title;
  getEl('travelVerdictSummary').textContent = data.verdict.summary;

  // 3. Area A Card
  getEl('travelNameA').textContent = data.area_a.name;
  getEl('travelScoreA').textContent = data.area_a.safety_score;
  getEl('travelBadgeA').textContent = data.area_a.safety_label;
  getEl('travelBadgeA').className = `safety-badge ${
    data.area_a.safety_score >= 70
      ? 'tier-baseline'
      : data.area_a.safety_score >= 50
      ? 'tier-elevated'
      : 'tier-critical'
  }`;
  getEl('travelDescA').textContent = data.area_a.safety_desc;

  const fillA = getEl('travelFillA');
  fillA.style.width = `${data.area_a.safety_score}%`;
  fillA.style.backgroundColor =
    data.area_a.safety_score >= 70
      ? '#27ae60'
      : data.area_a.safety_score >= 50
      ? '#f39c12'
      : '#e74c3c';

  getEl('travelTrendA').textContent = `${data.area_a.trend_icon} ${data.area_a.trend}`;
  getEl('travelTotalA').textContent = `${formatNumber(data.area_a.total_annual_crimes)} cases/year`;
  getEl('travelTipsA').innerHTML = data.area_a.travel_tips
    .map((t) => `<li>${t}</li>`)
    .join('');

  // 4. Area B Card
  getEl('travelNameB').textContent = data.area_b.name;
  getEl('travelScoreB').textContent = data.area_b.safety_score;
  getEl('travelBadgeB').textContent = data.area_b.safety_label;
  getEl('travelBadgeB').className = `safety-badge ${
    data.area_b.safety_score >= 70
      ? 'tier-baseline'
      : data.area_b.safety_score >= 50
      ? 'tier-elevated'
      : 'tier-critical'
  }`;
  getEl('travelDescB').textContent = data.area_b.safety_desc;

  const fillB = getEl('travelFillB');
  fillB.style.width = `${data.area_b.safety_score}%`;
  fillB.style.backgroundColor =
    data.area_b.safety_score >= 70
      ? '#27ae60'
      : data.area_b.safety_score >= 50
      ? '#f39c12'
      : '#e74c3c';

  getEl('travelTrendB').textContent = `${data.area_b.trend_icon} ${data.area_b.trend}`;
  getEl('travelTotalB').textContent = `${formatNumber(data.area_b.total_annual_crimes)} cases/year`;
  getEl('travelTipsB').innerHTML = data.area_b.travel_tips
    .map((t) => `<li>${t}</li>`)
    .join('');

  // 5. Chart.js Grouped Bar Chart
  const ctx = getEl('travelChart').getContext('2d');
  if (travelChartInstance) {
    travelChartInstance.destroy();
  }

  travelChartInstance = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: data.categories_comparison.map((c) => c.category),
      datasets: [
        {
          label: data.area_a.name,
          data: data.categories_comparison.map((c) => c.count_a),
          backgroundColor: 'rgba(54, 89, 217, 0.85)',
          borderRadius: 4,
        },
        {
          label: data.area_b.name,
          data: data.categories_comparison.map((c) => c.count_b),
          backgroundColor: 'rgba(39, 174, 96, 0.85)',
          borderRadius: 4,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: true,
          position: 'top',
          labels: { font: { size: 12, weight: 'bold' } },
        },
        tooltip: {
          callbacks: {
            label: function (context) {
              return ` ${context.dataset.label}: ${formatNumber(context.raw)} cases/year`;
            },
          },
        },
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: { color: '#8792a5', font: { size: 11 } },
        },
        y: {
          grid: { color: '#edf0f5' },
          ticks: { color: '#8792a5', font: { size: 10 } },
          title: {
            display: true,
            text: 'Average Reported Cases / Year',
            font: { size: 10, weight: 'bold' },
            color: '#8792a5',
          },
        },
      },
    },
  });
}

/**
 * Fetch and display travel safety comparison
 */
async function fetchTravelComparison() {
  const btn = getEl('travelCompareBtn');
  const originalText = btn.textContent;
  btn.textContent = 'Comparing safety…';
  btn.disabled = true;

  try {
    const stateA = getEl('travelStateA').value;
    const districtA = getEl('travelDistrictA').value;
    const stateB = getEl('travelStateB').value;
    const districtB = getEl('travelDistrictB').value;
    const profile = getEl('travelProfile').value;

    const params = new URLSearchParams({
      state_a: stateA,
      district_a: districtA,
      state_b: stateB,
      district_b: districtB,
      profile: profile,
    });

    const data = await fetchJSON(`/api/travel/compare?${params}`);
    renderTravelComparison(data);
  } catch (error) {
    console.error('Error fetching travel comparison:', error);
    alert(error.message);
  } finally {
    btn.textContent = originalText;
    btn.disabled = false;
  }
}

// =============================================================================
// 4c. Manpower & Tactical Allocation Rendering
// =============================================================================

/**
 * Friendly label helper for risk tiers
 * @param {string} tier - Internal tier key (CRITICAL, HIGH, ELEVATED, BASELINE)
 * @returns {string} Human-friendly label with emoji
 */
function getFriendlyRiskLabel(tier) {
  switch ((tier || '').toUpperCase()) {
    case 'CRITICAL':
      return '🔴 High Danger';
    case 'HIGH':
      return '🟠 Elevated Risk';
    case 'ELEVATED':
      return '🟡 Moderate';
    case 'BASELINE':
    default:
      return '🟢 Normal / Safe';
  }
}

/**
 * Render the tactical police manpower allocation view, chart, hotspots, and table
 * @param {object} data - Manpower optimization API response payload
 */
function renderManpowerAllocation(data) {
  // 1. Strategy Banner
  getEl('mpStrategyTag').textContent = data.strategy.label;
  getEl('mpStrategyDesc').textContent = data.strategy.description;

  // 2. KPI Cards
  getEl('mpTotalDeployed').textContent = `${formatNumber(data.total_manpower)} Officers`;
  getEl('mpEntityCount').textContent = `Across ${formatNumber(data.total_entities)} ${data.entity_label.toLowerCase()}`;
  getEl('mpCriticalCount').textContent = `${data.critical_count} High-Risk Areas`;
  getEl('mpMitigationGain').textContent = `+${data.mitigation_improvement_pct}% Better Safety`;
  getEl('mpProjectedCrimes').textContent = `${formatNumber(data.total_predicted_crimes)} Crimes`;
  getEl('mpThreatTotal').textContent = `Total Danger Score: ${formatNumber(data.total_threat_score)}`;

  // 3. Render Chart (Top 12 Entities)
  const topItems = data.allocations.slice(0, 12);
  const ctx = getEl('manpowerChart').getContext('2d');

  if (manpowerChartInstance) {
    manpowerChartInstance.destroy();
  }

  manpowerChartInstance = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: topItems.map((item) => item.entity),
      datasets: [
        {
          label: 'Police to Send (Officers)',
          data: topItems.map((item) => item.allocated_manpower),
          backgroundColor: '#27ae60',
          borderRadius: 4,
          yAxisID: 'yForce',
        },
        {
          label: 'Danger Score',
          data: topItems.map((item) => item.threat_index),
          backgroundColor: 'rgba(54, 89, 217, 0.65)',
          borderRadius: 4,
          yAxisID: 'yThreat',
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: true,
          position: 'top',
          labels: { font: { size: 11 }, boxWidth: 12 },
        },
        tooltip: {
          callbacks: {
            afterBody: function (context) {
              const idx = context[0].dataIndex;
              const item = topItems[idx];
              return `Risk Level: ${getFriendlyRiskLabel(item.risk_tier)}\nMain Issue: ${item.primary_driver}\nExpected Crimes: ${formatNumber(item.predicted_crimes)}`;
            },
          },
        },
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: {
            color: '#8792a5',
            font: { size: 10 },
            maxRotation: 45,
            minRotation: 25,
          },
        },
        yForce: {
          type: 'linear',
          position: 'left',
          title: {
            display: true,
            text: 'Police to Send',
            font: { size: 10, weight: 'bold' },
            color: '#27ae60',
          },
          grid: { color: '#edf0f5' },
          ticks: { color: '#27ae60', font: { size: 10 } },
        },
        yThreat: {
          type: 'linear',
          position: 'right',
          title: {
            display: true,
            text: 'Danger Score',
            font: { size: 10, weight: 'bold' },
            color: '#3659d9',
          },
          grid: { display: false },
          ticks: { color: '#3659d9', font: { size: 10 } },
        },
      },
    },
  });

  // 4. Render High-Alert Hotspots Directives
  const hotspotsContainer = getEl('mpHotspotsList');
  if (data.top_hotspots && data.top_hotspots.length > 0) {
    hotspotsContainer.innerHTML = data.top_hotspots
      .map((h) => {
        const borderCol = h.risk_tier === 'CRITICAL' ? '#e74c3c' : '#f39c12';
        return `
          <div class="hotspot-card" style="border-left-color: ${borderCol};">
            <div class="hotspot-top">
              <span class="hotspot-name">#${h.rank} ${h.entity}</span>
              <span class="hotspot-force">Send ${formatNumber(h.allocated_manpower)} Officers (${h.share_pct}% of force)</span>
            </div>
            <div class="hotspot-metrics">
              <span><b>Danger Score:</b> ${formatNumber(h.threat_index)}</span>
              <span><b>Expected Crimes:</b> ${formatNumber(h.predicted_crimes)}</span>
              <span><b>Main Issue:</b> ${h.primary_driver}</span>
            </div>
            <div class="hotspot-directive">
              <b>What to do:</b> ${h.tactical_directive}
            </div>
          </div>
        `;
      })
      .join('');
  } else {
    hotspotsContainer.innerHTML = '<div style="color: #8792a5; padding: 20px;">No high-danger areas found.</div>';
  }

  // 5. Render Full Ranked Allocation Table
  const tableBody = getEl('mpTableBody');
  tableBody.innerHTML = data.allocations
    .map((a) => {
      const tierClass = `tier-${a.risk_tier.toLowerCase()}`;
      return `
        <tr>
          <td><b>#${a.rank}</b></td>
          <td><b>${a.entity}</b></td>
          <td><span class="tier-badge ${tierClass}">${getFriendlyRiskLabel(a.risk_tier)}</span></td>
          <td>${formatNumber(a.predicted_crimes)}</td>
          <td>${formatNumber(a.threat_index)}</td>
          <td><b class="force-badge">${formatNumber(a.allocated_manpower)}</b></td>
          <td>${a.share_pct}%</td>
          <td>${a.primary_driver}</td>
          <td><small style="color: #4a5568;">${a.tactical_directive}</small></td>
        </tr>
      `;
    })
    .join('');
}

/**
 * Fetch and display optimized police manpower resource plan
 */
async function fetchManpowerPlan() {
  const btn = getEl('mpOptimizeBtn');
  const originalText = btn.textContent;
  btn.textContent = 'Calculating police needed…';
  btn.disabled = true;

  try {
    const state = getEl('mpState').value;
    const year = getEl('mpYear').value;
    const pool = getEl('mpPool').value;
    const strategy = getEl('mpStrategy').value;
    const base = getEl('mpBase').value;

    const params = new URLSearchParams({
      state: state,
      year: year,
      manpower: pool,
      strategy: strategy,
      min_per_district: base,
    });

    const data = await fetchJSON(`/api/manpower/optimize?${params}`);
    renderManpowerAllocation(data);
  } catch (error) {
    console.error('Error computing manpower allocation:', error);
    alert(error.message);
  } finally {
    btn.textContent = originalText;
    btn.disabled = false;
  }
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

  // Populate Manpower Jurisdiction Selector
  getEl('mpState').innerHTML = options.states
    .map((state) => `<option value="${state}">${state === 'All India' ? 'All India (National Reserve)' : state}</option>`)
    .join('');

  // Populate Travel Comparison State Selectors
  const individualStates = options.states.filter((s) => s !== 'All India');
  const travelStateOptions = individualStates
    .map((state) => `<option value="${state}">${state}</option>`)
    .join('');
  getEl('travelStateA').innerHTML = travelStateOptions;
  getEl('travelStateB').innerHTML = travelStateOptions;

  // Set friendly defaults (e.g. Tamil Nadu: Chennai vs Coimbatore)
  if (individualStates.includes('TAMIL NADU')) {
    getEl('travelStateA').value = 'TAMIL NADU';
    getEl('travelStateB').value = 'TAMIL NADU';
  }

  await updateTravelDistrictsA();
  await updateTravelDistrictsB();

  if (getEl('travelDistrictA').querySelector('option[value="CHENNAI"]')) {
    getEl('travelDistrictA').value = 'CHENNAI';
  }
  if (getEl('travelDistrictB').querySelector('option[value="COIMBATORE"]')) {
    getEl('travelDistrictB').value = 'COIMBATORE';
  }

  // Populate District Risk State Selector
  if (getEl('districtRiskState')) {
    getEl('districtRiskState').innerHTML = individualStates
      .map((state) => `<option value="${state}">${state}</option>`)
      .join('');
    if (individualStates.includes('TAMIL NADU')) {
      getEl('districtRiskState').value = 'TAMIL NADU';
    }
    await updateDistrictRiskDropdown('CHENNAI');
    await fetchDistrictRiskAnalysis();
  }

  // Populate dependent districts and trigger initial render
  await updateDistricts();
  await refreshDashboard();
  await loadIndiaSVGMap();
  await fetchIndiaRiskMap();
  await fetchTravelComparison();
  await fetchManpowerPlan();
}

// =============================================================================
// 6. Event Listeners & Startup
// =============================================================================
getEl('state').addEventListener('change', async () => {
  await updateDistricts();
});

getEl('apply').addEventListener('click', refreshDashboard);
getEl('rankingYear').addEventListener('change', loadPlaces);

// India Crime Risk Map Listeners
getEl('mapCategory').addEventListener('change', fetchIndiaRiskMap);
getEl('mapYear').addEventListener('change', fetchIndiaRiskMap);

// District Crime Risk Predictor Listeners
if (getEl('districtRiskCheckBtn')) {
  getEl('districtRiskCheckBtn').addEventListener('click', fetchDistrictRiskAnalysis);
}
if (getEl('districtRiskState')) {
  getEl('districtRiskState').addEventListener('change', async () => {
    await updateDistrictRiskDropdown();
    fetchDistrictRiskAnalysis();
  });
}
if (getEl('districtRiskDistrict')) {
  getEl('districtRiskDistrict').addEventListener('change', fetchDistrictRiskAnalysis);
}
if (getEl('districtRiskIncident')) {
  getEl('districtRiskIncident').addEventListener('change', fetchDistrictRiskAnalysis);
}
if (getEl('districtRiskYear')) {
  getEl('districtRiskYear').addEventListener('change', fetchDistrictRiskAnalysis);
}

// Travel Safety Comparison Listeners
getEl('travelCompareBtn').addEventListener('click', fetchTravelComparison);
getEl('travelProfile').addEventListener('change', fetchTravelComparison);
getEl('travelStateA').addEventListener('change', async () => {
  await updateTravelDistrictsA();
  fetchTravelComparison();
});
getEl('travelStateB').addEventListener('change', async () => {
  await updateTravelDistrictsB();
  fetchTravelComparison();
});
getEl('travelDistrictA').addEventListener('change', fetchTravelComparison);
getEl('travelDistrictB').addEventListener('change', fetchTravelComparison);

// Manpower Optimization Listeners
getEl('mpOptimizeBtn').addEventListener('click', fetchManpowerPlan);
getEl('mpStrategy').addEventListener('change', fetchManpowerPlan);
getEl('mpState').addEventListener('change', fetchManpowerPlan);

// Bootstrap
initDashboard().catch((error) => {
  console.error('Initialization error:', error);
  alert(error.message);
});
