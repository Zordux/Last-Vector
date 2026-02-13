(function () {
  const runSearch = document.getElementById('runSearch');
  const runButtons = Array.from(document.querySelectorAll('.run-select'));
  const runItems = Array.from(document.querySelectorAll('.run-item'));
  const runPanels = Array.from(document.querySelectorAll('.run-panel[data-content="details"]'));
  const tabs = Array.from(document.querySelectorAll('.tab'));
  const themeToggle = document.getElementById('themeToggle');
  const root = document.documentElement;

  // Chart.js instances
  const charts = {};
  let lastRefreshedTime = null;

  function showRunPanel(panelId) {
    runPanels.forEach((panel) => panel.classList.toggle('hidden', panel.id !== panelId));
    runButtons.forEach((button) => button.classList.toggle('active', button.dataset.runTarget === panelId));
  }

  runButtons.forEach((button) => {
    button.addEventListener('click', () => showRunPanel(button.dataset.runTarget));
  });

  if (runSearch) {
    runSearch.addEventListener('input', () => {
      const query = runSearch.value.trim().toLowerCase();
      runItems.forEach((item) => {
        const runId = (item.dataset.runId || '').toLowerCase();
        item.classList.toggle('hidden', query.length > 0 && !runId.includes(query));
      });
    });
  }

  tabs.forEach((tab) => {
    tab.addEventListener('click', () => {
      tabs.forEach((candidate) => candidate.classList.remove('active'));
      tab.classList.add('active');
      const target = tab.dataset.tab;
      document.querySelectorAll('.run-panel').forEach((panel) => {
        if (panel.dataset.content === 'settings') {
          panel.classList.toggle('hidden', target !== 'settings');
        } else if (target === 'settings') {
          panel.classList.add('hidden');
        } else if (!document.querySelector('.run-select.active')) {
          panel.classList.add('hidden');
        }
      });
      if (target === 'details') {
        const active = document.querySelector('.run-select.active');
        if (active) {
          showRunPanel(active.dataset.runTarget);
        }
      }
    });
  });

  if (themeToggle) {
    themeToggle.addEventListener('click', () => {
      const theme = root.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
      root.setAttribute('data-theme', theme);
      window.localStorage.setItem('lv-theme', theme);
    });
  }

  const savedTheme = window.localStorage.getItem('lv-theme');
  if (savedTheme === 'light' || savedTheme === 'dark') {
    root.setAttribute('data-theme', savedTheme);
  }

  // Initialize Chart.js for each run
  function initCharts() {
    const seriesScripts = document.querySelectorAll('[id^="series-data-"]');
    seriesScripts.forEach((script) => {
      const index = script.id.replace('series-data-', '');
      const canvas = document.getElementById(`chart-${index}`);
      if (!canvas) return;

      const seriesData = JSON.parse(script.textContent);
      
      const ctx = canvas.getContext('2d');
      charts[index] = new Chart(ctx, {
        type: 'line',
        data: {
          labels: seriesData.steps,
          datasets: [
            {
              label: 'Reward',
              data: seriesData.reward,
              borderColor: '#a855f7',
              backgroundColor: 'rgba(168, 85, 247, 0.1)',
              borderWidth: 2,
              pointRadius: 0,
              tension: 0.1,
            },
            {
              label: 'Episode Length',
              data: seriesData.episode_length,
              borderColor: '#c084fc',
              backgroundColor: 'rgba(192, 132, 252, 0.1)',
              borderWidth: 2,
              pointRadius: 0,
              tension: 0.1,
            },
            {
              label: 'Kills',
              data: seriesData.kills,
              borderColor: '#f472b6',
              backgroundColor: 'rgba(244, 114, 182, 0.1)',
              borderWidth: 2,
              pointRadius: 0,
              tension: 0.1,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              labels: {
                color: '#e8eaf7',
              },
            },
          },
          scales: {
            x: {
              grid: {
                color: '#2d2744',
              },
              ticks: {
                color: '#9a9cbc',
              },
            },
            y: {
              grid: {
                color: '#2d2744',
              },
              ticks: {
                color: '#9a9cbc',
              },
            },
          },
        },
      });
    });
  }

  // Update chart data
  function updateChart(index, seriesData) {
    const chart = charts[index];
    if (!chart) return;

    chart.data.labels = seriesData.steps;
    chart.data.datasets[0].data = seriesData.reward;
    chart.data.datasets[1].data = seriesData.episode_length;
    chart.data.datasets[2].data = seriesData.kills;
    chart.update();
  }

  // Format number with specified decimal places
  function formatNumber(value, decimals) {
    const num = parseFloat(value);
    if (isNaN(num)) return value;
    return num.toFixed(decimals);
  }

  // Update runs data from API
  async function updateRuns() {
    try {
      const response = await fetch('/api/runs');
      if (!response.ok) return;
      
      const runs = await response.json();
      lastRefreshedTime = Date.now();

      runs.forEach((run, index) => {
        // Update metric cards
        document.querySelectorAll(`[data-run-id="${run.run_id}"][data-field="state"]`).forEach((el) => {
          el.innerHTML = `<span class="status-badge ${run.state}"></span>${run.state}`;
        });
        
        document.querySelectorAll(`[data-run-id="${run.run_id}"][data-field="progress_pct"]`).forEach((el) => {
          el.textContent = `${formatNumber(run.metrics.progress_pct, 1)}%`;
        });
        
        document.querySelectorAll(`[data-run-id="${run.run_id}"][data-field="progress_bar"]`).forEach((el) => {
          el.style.width = `${run.metrics.progress_pct}%`;
        });
        
        document.querySelectorAll(`[data-run-id="${run.run_id}"][data-field="steps"]`).forEach((el) => {
          el.textContent = `${run.metrics.steps} / ${run.metrics.total_steps}`;
        });
        
        document.querySelectorAll(`[data-run-id="${run.run_id}"][data-field="fps"]`).forEach((el) => {
          el.textContent = formatNumber(run.metrics.fps, 1);
        });
        
        document.querySelectorAll(`[data-run-id="${run.run_id}"][data-field="episodes"]`).forEach((el) => {
          el.textContent = run.metrics.episodes;
        });
        
        document.querySelectorAll(`[data-run-id="${run.run_id}"][data-field="latest_r_len"]`).forEach((el) => {
          el.textContent = `${formatNumber(run.metrics.latest_reward, 3)} / ${formatNumber(run.metrics.latest_ep_len, 1)}`;
        });
        
        document.querySelectorAll(`[data-run-id="${run.run_id}"][data-field="best_score"]`).forEach((el) => {
          el.textContent = formatNumber(run.best_model_info.score, 3);
        });
        
        document.querySelectorAll(`[data-run-id="${run.run_id}"][data-field="last_update"]`).forEach((el) => {
          el.textContent = run.metrics.last_update;
        });

        // Update sidebar
        document.querySelectorAll(`[data-run-id="${run.run_id}"][data-field="sidebar-state-progress"]`).forEach((el) => {
          el.textContent = `${run.state} • progress=${formatNumber(run.metrics.progress_pct, 1)}%`;
        });
        
        document.querySelectorAll(`[data-run-id="${run.run_id}"][data-field="sidebar-fps-best"]`).forEach((el) => {
          el.textContent = `fps=${formatNumber(run.metrics.fps, 1)} • best=${formatNumber(run.metrics.best_reward, 2)}`;
        });
        
        document.querySelectorAll(`[data-run-id="${run.run_id}"][data-field="sidebar-update"]`).forEach((el) => {
          el.textContent = `updated=${run.metrics.last_update}`;
        });

        // Update chart if series data changed
        if (run.series && charts[index]) {
          updateChart(index, run.series);
        }
      });
    } catch (error) {
      // Silently fail, will retry on next interval
      console.error('Failed to fetch runs:', error);
    }
  }

  // Update hardware data from API
  async function updateHardware() {
    try {
      const response = await fetch('/api/hw');
      if (!response.ok) return;
      
      const hw = await response.json();
      
      const cpuEl = document.getElementById('hwCpu');
      const ramEl = document.getElementById('hwRam');
      
      if (cpuEl) cpuEl.textContent = formatNumber(hw.cpu_percent, 1);
      if (ramEl) ramEl.textContent = formatNumber(hw.ram_percent, 1);
    } catch (error) {
      // Silently fail, will retry on next interval
      console.error('Failed to fetch hardware:', error);
    }
  }

  // Update "last refreshed" counter
  function updateRefreshCounter() {
    const el = document.getElementById('lastRefreshed');
    if (!el || !lastRefreshedTime) return;

    const secondsAgo = Math.floor((Date.now() - lastRefreshedTime) / 1000);
    el.textContent = `Last refreshed: ${secondsAgo}s ago`;
  }

  // Initialize charts on page load
  if (typeof Chart !== 'undefined') {
    initCharts();
  }

  // Start polling loops
  setInterval(updateRuns, 5000);
  setInterval(updateHardware, 3000);
  setInterval(updateRefreshCounter, 1000);

  // Do initial updates
  updateRuns();
  updateHardware();
})();
