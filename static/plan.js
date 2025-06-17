// plan.js - handles loading spinner and AJAX form submit for plan.html

document.addEventListener('DOMContentLoaded', function() {
  const planForm = document.getElementById('planForm');
  const resultContainer = document.getElementById('plan-result-container');
  const spinnerHtml = `
    <div class="loading-spinner">
      <div class="spinner"></div>
      <div class="spinner-label">최적화 계산 중...<br>Please wait</div>
    </div>
  `;
  // Restore form and result from localStorage
  const LS_KEY = 'plan_state_v1';
  function saveState(budget, hours, resultHtml) {
    localStorage.setItem(LS_KEY, JSON.stringify({budget, hours, resultHtml}));
  }
  function loadState() {
    try {
      return JSON.parse(localStorage.getItem(LS_KEY));
    } catch { return null; }
  }
  function clearState() {
    localStorage.removeItem(LS_KEY);
    if (planForm) {
      planForm.reset();
    }
    if (resultContainer) {
      resultContainer.innerHTML = '';
    }
  }
  // Add reset button
  if (planForm && !document.getElementById('plan-reset-btn')) {
    const resetBtn = document.createElement('button');
    resetBtn.type = 'button';
    resetBtn.id = 'plan-reset-btn';
    resetBtn.textContent = '초기화';
    resetBtn.className = 'save-btn';
    resetBtn.style.marginLeft = '1em';
    resetBtn.onclick = clearState;
    planForm.appendChild(resetBtn);
  }
  // On page load, restore
  const state = loadState();
  if (state) {
    if (planForm) {
      if (state.budget) planForm.elements['budget'].value = state.budget;
      if (state.hours) planForm.elements['hours'].value = state.hours;
    }
    if (resultContainer && state.resultHtml) {
      resultContainer.innerHTML = state.resultHtml;
    }
  }
  if (planForm) {
    planForm.addEventListener('submit', function(e) {
      if (!resultContainer) return;
      e.preventDefault();
      resultContainer.innerHTML = spinnerHtml;
      const formData = new FormData(planForm);
      const budget = planForm.elements['budget'].value;
      const hours = planForm.elements['hours'].value;
      fetch(planForm.action || window.location.pathname, {
        method: 'POST',
        body: formData,
      })
      .then(resp => resp.text())
      .then(html => {
        const parser = new DOMParser();
        const doc = parser.parseFromString(html, 'text/html');
        const newResultContainer = doc.getElementById('plan-result-container');
        if (newResultContainer) {
          resultContainer.replaceWith(newResultContainer);
          // Save latest state
          saveState(budget, hours, newResultContainer.innerHTML);
        } else {
          document.body.innerHTML = doc.body.innerHTML;
        }
      })
      .catch(err => {
        resultContainer.innerHTML = '<div style="color:red;padding:2em;text-align:center;">오류가 발생했습니다.<br>'+err+'</div>';
      });
    });
  }
});
