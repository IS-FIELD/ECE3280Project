// popup.js
document.addEventListener('DOMContentLoaded', () => {
    // 保留原有设置功能
    const configForm = document.createElement('div');
    configForm.innerHTML = `
      <h3>API Settings</h3>
      <input type="text" id="api-endpoint" placeholder="API Endpoint">
      <button id="save-settings">Save</button>
    `;
    
    document.body.prepend(configForm);
    
    // 加载保存的API设置
    chrome.storage.sync.get(['apiEndpoint'], (data) => {
      document.getElementById('api-endpoint').value = data.apiEndpoint || '';
    });
    
    // 保存设置
    document.getElementById('save-settings').onclick = () => {
      const endpoint = document.getElementById('api-endpoint').value.trim();
      chrome.storage.sync.set({ apiEndpoint: endpoint });
    };
  });