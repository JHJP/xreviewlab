/* X Review Lab · interactive onboarding  (mobile-safe version)
   ───────────────────────────────────────────────────────────── */
   (function () {
  // 튜토리얼 활성/비활성 제어 변수
  const TUTORIAL_ENABLED = false; // true로 바꾸면 다시 활성화
  if (!TUTORIAL_ENABLED) return;

    /* ❶ Steps list (leave selectors & copy unchanged) */
    const steps = [
      { el: '.methods-grid',           msg: '1단계: 체크박스로 대응 방법을 고르세요.' },
      { el: '#budget',                 msg: '2단계: 예산상자에 50,000을 입력하세요(만원).' },
      { el: '#total_hours_available',  msg: '3단계: 시간상자에 5을 입력하여, 채팅 또는 전화 대응에 5시간을 할당하세요.' },
      { el: 'button[type=submit]',     msg: '4단계: ‘대응 플랜 생성’ 버튼을 눌러서 결과를 확인하세요!' }
    ];
  
    /* ❷ Reusable DOM bits */
    const tip = document.createElement('div');
    tip.className = 'tutorial-tip';
    document.body.appendChild(tip);     // single floating tooltip
  
    let idx           = -1;             // current step index
    let activeTarget  = null;           // HTMLElement that tip is pinned to
  
    /* ❸ Helpers ------------------------------------------------------------- */
    function clearHighlight () {
      document.querySelectorAll('.tutorial-highlight')
              .forEach(el => el.classList.remove('tutorial-highlight'));
    }
  
    /* Responsive smart-placement
       – tries right → left → centred under the element,
         then clamps to viewport so it never disappears.               */
    function placeTip (target) {
      const GAP = 12;
      const r        = target.getBoundingClientRect();
      const tipW     = tip.offsetWidth;
      const tipH     = tip.offsetHeight;
  
      /* ① Horizontal -------------------------------------------------------- */
      let left;                     // px value we’ll finally assign
      let arrow;                    // 'left' | 'right' | ''  (centre = no arrow)
      const roomRight = window.innerWidth - r.right - GAP;
      const roomLeft  = r.left - GAP;
  
      if (roomRight >= tipW) {                    // enough space on the right
        left  = r.right + GAP;
        arrow = 'left';
      } else if (roomLeft >= tipW) {              // enough space on the left
        left  = r.left - tipW - GAP;
        arrow = 'right';
      } else {                                    // nowhere → centre below
        left  = r.left + (r.width - tipW) / 2;
        arrow = '';                              // hide the arrow via data-attr
      }
      /* Clamp so we never overflow the viewport horizontally */
      left = Math.max(GAP, Math.min(left, window.innerWidth - tipW - GAP));
  
      /* ② Vertical ---------------------------------------------------------- */
      let top = r.bottom + GAP;                  // default = below the target
      const bottomOverflow = top + tipH - (window.innerHeight + window.scrollY);
  
      if (bottomOverflow > 0 && r.top - GAP - tipH > 0) {
        /* not enough room below → put it above if there is space */
        top = r.top - tipH - GAP;
      }
      /* Clamp vertically too (rare on extremely small heights) */
      top = Math.max(GAP + window.scrollY, top);
  
      /* ③ Apply ------------------------------------------------------------- */
      tip.style.left = `${left + window.scrollX}px`;
      tip.style.top  = `${top}px`;
      tip.dataset.arrow = arrow;                 // triggers CSS :after direction
    }
  
    /* ④ Show step `i` or finish if done ------------------------------------ */
    function show (i) {
      if (i >= steps.length) { tip.remove(); clearHighlight(); return; }
      idx = i;
  
      const { el, msg } = steps[idx];
      const tgt = document.querySelector(el);
      if (!tgt) { show(idx + 1); return; }       // selector missing → skip
  
      activeTarget = tgt;                        // remember for resize events
      clearHighlight();
      tgt.classList.add('tutorial-highlight');
      tgt.scrollIntoView({ block: 'center', behavior: 'smooth' });
  
      tip.innerHTML = `
        <p style="margin-bottom:.8rem">${msg}</p>
        <button class="tutorial-next">다음 →</button>
      `;
      /* Wait one frame so the browser has the real width/height, then place */
      requestAnimationFrame(() => placeTip(tgt));
  
      tip.querySelector('.tutorial-next').onclick = () => show(idx + 1);
    }
  
    /* ⑤ Keep the tooltip in place if the user rotates phone or resizes window */
    window.addEventListener('resize', () => {
      if (activeTarget) placeTip(activeTarget);
    });
  
    /* Public API so ui.js can trigger the wizard --------------------------- */
    window.startTutorial = () => show(0);
  
  })();
  