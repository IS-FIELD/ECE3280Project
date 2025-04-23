// content.js
let currentSelectedLabels = { level1: null, level2: null };

// 创建描述输入框
function createDescriptionInput() {
  const container = document.createElement('div');
  container.style.cssText = `/* 保持与原插件一致的样式 */`;
  
  const textarea = document.createElement('textarea');
  textarea.placeholder = "Enter research description...";
  
  const analyzeBtn = document.createElement('button');
  analyzeBtn.textContent = "Analyze";
  analyzeBtn.onclick = async () => {
    const description = textarea.value.trim();
    if (description) {
      const labels = await getTopLabels(description, 'level1');
      showLabelSelection(labels, 'level1');
    }
  };

  container.append(textarea, analyzeBtn);
  document.body.prepend(container);
}

// API请求封装
async function getTopLabels(text, level) {
  try {
    const response = await fetch('YOUR_API_ENDPOINT', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, level })
    });
    
    const data = await response.json();
    return data.labels.slice(0, level === 'level1' ? 3 : 5);
  } catch (error) {
    console.error('API请求失败:', error);
    showToast('Failed to get labels', true);
  }
}

// 显示标签选择界面
function showLabelSelection(labels, level) {
  const modal = document.createElement('div');
  modal.className = 'label-selection-modal'; // 使用原有样式体系
  
  labels.forEach(label => {
    const item = document.createElement('div');
    item.className = 'label-item';
    
    const header = document.createElement('h3');
    header.textContent = Object.keys(label)[0];
    
    const desc = document.createElement('p');
    desc.textContent = Object.values(label)[0];
    
    const selectBtn = document.createElement('button');
    selectBtn.textContent = "Select";
    selectBtn.onclick = () => handleLabelSelect(level, Object.keys(label)[0]);
    
    item.append(header, desc, selectBtn);
    modal.appendChild(item);
  });
  
  document.body.appendChild(modal);
}

// 处理标签选择
async function handleLabelSelect(level, label) {
  currentSelectedLabels[level] = label;
  
  if (level === 'level1') {
    const subLabels = await getTopLabels(label, 'level2');
    showLabelSelection(subLabels, 'level2');
  } else {
    executeSearch();
  }
}

// 执行最终搜索
function executeSearch() {
  const searchQuery = `${currentSelectedLabels.level1} ${currentSelectedLabels.level2}`;
  
  // 根据原有代码的搜索方式
  const searchBox = document.querySelector('input[name="q"]');
  if (searchBox) {
    searchBox.value = searchQuery;
    document.querySelector('form').submit();
  } else {
    window.location.href = `https://scholar.google.com/scholar?q=${encodeURIComponent(searchQuery)}`;
  }
}

// 初始化
function init() {
  createDescriptionInput();
  // 保留原有的事件监听和样式
}

// 保留原有代码的样式和工具函数
const originalStyles = document.createElement('style');
originalStyles.textContent = `/* 原有样式 + 新增选择界面样式 */`;
document.head.appendChild(originalStyles);

init();