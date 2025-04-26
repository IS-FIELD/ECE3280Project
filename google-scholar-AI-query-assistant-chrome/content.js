const API_ENDPOINT = 'http://localhost:8008/classify';
let currentState = {
  originalInput: '',
  infer1: null,
  infer2: null
};

function createUI() {
  const container = document.createElement('div');
  container.id = 'assistant-container';
  container.innerHTML = `
    <style>
      .confidence { color: #4CAF50; font-size: 0.9em; }
      .explanation { color: #666; font-size: 0.85em; margin-top: 4px; }
    </style>
    <div class="header">
      <h3>Scholar AI Query Assistant</h3>
    </div>
    <div class="input-section">
      <textarea id="desc-input" placeholder="Describe your research..."></textarea>
      <button id="analyze-btn">Analyze</button>
    </div>
    <div id="selection-container"></div>
    <div class="loading-overlay">
      <div class="spinner"></div>
    </div>
  `;
  document.body.appendChild(container);

  document.getElementById('analyze-btn').addEventListener('click', startAnalysis);
}

async function startAnalysis() {
  const input = document.getElementById('desc-input').value.trim();
  if (!input) return;

  currentState.originalInput = input;
  showLoading(true);

  try {
    // get infer_1
    const infer1Result = await fetchLabels(input, 'infer_1');
    renderSelection(infer1Result, 'Select Primary Category', async (selected) => {
      currentState.infer1 = selected;
      
      // get infer_2
      const infer2Result = await fetchLabels(input, 'infer_2', selected);
      renderSelection(infer2Result, 'Select Secondary Category', (selected) => {
        currentState.infer2 = selected;
        performSearch();
      }, 5);
    }, 3);
  } catch (error) {
    showToast(`Error: ${error.message}`, true);
  } finally {
    showLoading(false);
  }
}

async function fetchLabels(text, infer, context) {
  try {
    const response = await fetch(API_ENDPOINT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: text,
        infer: infer,
        context: context || null
      })
    });

    const data = await response.json();
    
    if (!response.ok || data.error) {
      throw new Error(data.error || 'Request failed');
    }

    // check
    if (!data?.labels?.length || !data.labels[0].label) {
      throw new Error('Invalid response format');
    }

    return data;
  } catch (error) {
    showToast(`API Error: ${error.message}`, true);
    throw error;
  }
}

function renderSelection(data, title, callback, maxItems = 5) {
  const container = document.getElementById('selection-container');
  container.innerHTML = `
    <div class="selection-panel">
      <h4>${title}</h4>
      <div class="items-container"></div>
    </div>
  `;

  const itemsContainer = container.querySelector('.items-container');
  data.labels.slice(0, maxItems).forEach((item, index) => {
    const div = document.createElement('div');
    div.className = 'selection-item';
    div.innerHTML = `
      <div class="item-header">
        <span class="item-rank">#${index + 1}</span>
        <h5>${item.label}</h5>
        <span class="confidence">${(item.score * 100).toFixed(1)}%</span>
      </div>
      ${item.explanation ? `<p class="explanation">${item.explanation}</p>` : ''}
    `;
    div.addEventListener('click', () => callback(item.label));
    itemsContainer.appendChild(div);
  });
}

function performSearch() {
  const queryParts = [];
  if (currentState.infer1) queryParts.push(currentState.infer1);
  if (currentState.infer2) queryParts.push(currentState.infer2);
  if (queryParts.length === 0) return;
  const query = queryParts.join(' ');
  
  const form = document.querySelector('form[id="gs_hdr_frm"]');
  const input = document.querySelector('input[name="q"]');
  
  if (form && input) {
    input.value = query;
    
    const event = new Event('input', { bubbles: true });
    input.dispatchEvent(event);
    
    setTimeout(() => {
      form.submit();
    }, 100);
  } else {
    const encodedQuery = encodeURIComponent(query)
      .replace(/%20/g,'+')
      .replace(/%2B/g,'+');
    window.location.href = `https://scholar.google.com/scholar?q=${encodedQuery}`;
  }
}

function showLoading(show) {
  document.querySelector('.loading-overlay').style.display = show ? 'flex' : 'none';
}

function showToast(message, isError = false) {
  const toast = document.createElement('div');
  toast.className = `toast ${isError ? 'error' : ''}`;
  toast.textContent = message;
  document.body.appendChild(toast);

  setTimeout(() => toast.remove(), 3000);
}

// init
if (location.hostname === 'scholar.google.com') {
  const observer = new MutationObserver(() => {
    if (document.body && !document.getElementById('assistant-container')) {
      createUI();
    }
  });
  observer.observe(document.documentElement, { childList: true, subtree: true });
}