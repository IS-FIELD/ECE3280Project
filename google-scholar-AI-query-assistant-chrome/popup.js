document.addEventListener('DOMContentLoaded', () => {
  chrome.storage.sync.get(['apiEndpoint'], (data) => {
    document.getElementById('api-endpoint').value = data.apiEndpoint || 'http://localhost:8008/classify';
  });

  document.getElementById('save-settings').addEventListener('click', () => {
    const endpoint = document.getElementById('api-endpoint').value.trim();
    chrome.storage.sync.set({ apiEndpoint: endpoint });
  });
});