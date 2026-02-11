(function () {
  const runSearch = document.getElementById('runSearch');
  const runButtons = Array.from(document.querySelectorAll('.run-select'));
  const runItems = Array.from(document.querySelectorAll('.run-item'));
  const runPanels = Array.from(document.querySelectorAll('.run-panel[data-content="details"]'));
  const tabs = Array.from(document.querySelectorAll('.tab'));
  const themeToggle = document.getElementById('themeToggle');
  const root = document.documentElement;

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
})();
