// EigenLoom Web Dashboard Surface — Paper & Ink Light Palette (#f5f2ea / #141413)
(function registerEigenLoomDashboard() {
  const PALETTE = {
    bg: '#f5f2ea',
    card: '#faf8f3',
    ink: '#141413',
    muted: '#4a4843',
    border: '#cfc8b8',
    sage: '#2f5d50'
  };

  function renderEigenLoomPanel(container, statusPayload) {
    if (!container) return;
    container.textContent = '';
    const root = document.createElement('section');
    root.style.backgroundColor = PALETTE.bg;
    root.style.color = PALETTE.ink;
    root.style.colorScheme = 'only light';
    root.style.border = '1px solid ' + PALETTE.border;
    root.style.borderRadius = '8px';
    root.style.padding = '16px';
    root.style.fontFamily = 'ui-sans-serif, system-ui, sans-serif';

    const heading = document.createElement('h2');
    heading.textContent = 'EigenLoom Cybernetic Engine Manifold';
    heading.style.margin = '0 0 6px 0';
    heading.style.fontSize = '18px';
    root.appendChild(heading);

    const sub = document.createElement('p');
    const eCount = statusPayload && statusPayload.engine_count ? statusPayload.engine_count : 20;
    const sCount = statusPayload && statusPayload.sub_engine_count ? statusPayload.sub_engine_count : 72;
    const dims = statusPayload && statusPayload.hypergrid_total_dimensions ? statusPayload.hypergrid_total_dimensions : 100;
    sub.textContent = eCount + ' Functional Engines • ' + sCount + ' Sub-Engines • ' + dims + 'D Fluid Matryoshka HyperGrid';
    sub.style.margin = '0 0 12px 0';
    sub.style.color = PALETTE.muted;
    sub.style.fontSize = '13px';
    root.appendChild(sub);

    const card = document.createElement('div');
    card.style.backgroundColor = PALETTE.card;
    card.style.border = '1px solid ' + PALETTE.border;
    card.style.borderRadius = '6px';
    card.style.padding = '12px';
    const tasks = statusPayload && statusPayload.persisted_tasks ? statusPayload.persisted_tasks : 0;
    const verifs = statusPayload && statusPayload.persisted_verifications ? statusPayload.persisted_verifications : 0;
    const atts = statusPayload && statusPayload.persisted_attractors ? statusPayload.persisted_attractors : 0;
    card.textContent = 'SQLite WAL Read-Only Snapshot — Tasks: ' + tasks + ' | Maker≠Checker Verifications: ' + verifs + ' | Hopfield Attractors: ' + atts;
    root.appendChild(card);

    container.appendChild(root);
  }

  if (typeof window !== 'undefined' && typeof window.registerDashboardPlugin === 'function') {
    window.registerDashboardPlugin('eigenloom', {
      mount(el) {
        renderEigenLoomPanel(el, null);
        if (typeof window.fetch === 'function') {
          window.fetch('/api/plugins/eigenloom/status')
            .then(function (r) { return r.ok ? r.json() : null; })
            .then(function (data) { if (data) renderEigenLoomPanel(el, data); })
            .catch(function () {});
        }
      }
    });
  }
})();
