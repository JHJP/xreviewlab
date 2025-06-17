// plan.js - handles loading spinner and AJAX form submit for plan.html

document.addEventListener('DOMContentLoaded', function() {
  const planForm = document.getElementById('planForm');
  const resultCard = document.getElementById('plan-result-card');
  const resultContainer = document.getElementById('plan-result-container');
  const spinnerHtml = `
    <div class="loading-spinner">
      <div class="spinner"></div>
      <div class="spinner-label">최적화 계산 중...<br>Please wait</div>
    </div>
  `;

  if (planForm) {
    planForm.addEventListener('submit', function(e) {
      // Only use AJAX if result container exists
      if (!resultContainer) return;
      e.preventDefault();
      // Show spinner
      resultContainer.innerHTML = spinnerHtml;
      // Prepare form data
      const formData = new FormData(planForm);
      fetch(planForm.action || window.location.pathname, {
        method: 'POST',
        body: formData,
      })
      .then(resp => resp.text())
      .then(html => {
        // Replace the result container with new HTML
        const parser = new DOMParser();
        const doc = parser.parseFromString(html, 'text/html');
        const newResultContainer = doc.getElementById('plan-result-container');
        if (newResultContainer) {
          resultContainer.replaceWith(newResultContainer);
        } else {
          // fallback: full page replace
          document.body.innerHTML = doc.body.innerHTML;
        }
      })
      .catch(err => {
        resultContainer.innerHTML = '<div style="color:red;padding:2em;text-align:center;">오류가 발생했습니다.<br>'+err+'</div>';
      });
    });
  }
});
