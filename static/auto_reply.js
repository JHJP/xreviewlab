// 자동응답 버튼 기능 (프론트엔드 목업)
document.addEventListener("DOMContentLoaded", function() {
  document.querySelectorAll(".auto-reply-btn").forEach(function(btn) {
    btn.addEventListener("click", function() {
      const row = btn.closest("tr");
      const reviewContent = btn.getAttribute("data-review-content");
      const resultTd = btn.parentElement.querySelector(".auto-reply-result");
      btn.disabled = true;
      btn.textContent = "생성 중...";
      // 실제 백엔드 연동
      fetch('/generate_auto_reply', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ review_content: reviewContent })
      })
        .then(response => response.text())
        .then(text => {
          resultTd.textContent = text;
        })
        .catch(err => {
          resultTd.textContent = '[오류] 자동응답 생성 실패';
        })
        .finally(() => {
          btn.textContent = "자동응답 재생성";
          btn.disabled = false;
        });
    });
  });
});
