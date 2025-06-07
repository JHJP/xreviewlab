/* X Review Lab – tiny UX helpers */
document.addEventListener("DOMContentLoaded",()=>{
    const form   = document.getElementById("plannerForm");
    const loader = document.getElementById("loading-overlay");
    const progressBar = document.getElementById("progress-bar");
    const progressText = document.getElementById("progress-text");
    function pollProgress() {
      fetch('/progress')
        .then(res => res.json())
        .then(data => {
          console.log('[progress]', data); // 디버깅용
          const percent = Math.round((data.current / data.total) * 100);
          if(progressBar) progressBar.style.width = percent + "%";
          if(progressText) progressText.textContent = `AI 분석 진행 중... (${data.current} / ${data.total})`;
          if(data.current < data.total) {
            setTimeout(pollProgress, 500);
          } else {
            progressBar.style.width = "100%";
            progressText.textContent = "AI 분석 완료!";
          }
        });
    }
    if(form){
      form.addEventListener("submit", ()=>{
        loader.style.display = "flex";       // show spinner as soon as user clicks
        form.querySelector("button[type=submit]").disabled = true;
        pollProgress();
      });
    }
    /* Hide welcome overlay when tutorial button clicked */
    const startBtn = document.getElementById("startTutorialBtn");
    if(startBtn){
        startBtn.addEventListener("click", ()=>{
            document.getElementById("welcome-overlay").style.display="none";
            if(window.startTutorial) window.startTutorial();   // ← launch wizard
        });
    }
  });
  