"""
========================================================
 Brand-Risk Optimizer & Sensitivity Toolkit  v2
--------------------------------------------------------
  • 0-1 MILP  (CBC / HiGHS)
  • LP-relaxation dual  (upper-bound shadow price)
  • Break-point scanning  →  Discrete shadow price table
--------------------------------------------------------
 author : (your name) | updated : 2025-06-17
========================================================
"""
# ----------------- 0. Imports & Parameters -----------------
import pandas as pd, numpy as np, ast
import pulp, math
# ─── 한글 글꼴 수동 등록 & 설정 ──────────────────────────
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from pathlib import Path

# ── ❷ 내 폰트 파일 경로 지정 ────────────────────
MY_FONT_PATH = Path(
    r"C:\Users\papag\OneDrive\desktop\Business\ORM\App\MVP_gurobi\fonts\GamjaFlower-Regular.ttf"
)  # ← 여기만 내 파일로 바꿔 주세요

# ── ❸ Matplotlib 폰트 매니저에 등록(한 번만) ────
fm.fontManager.addfont(str(MY_FONT_PATH))        # Matplotlib >= 3.2
font_prop = fm.FontProperties(fname=str(MY_FONT_PATH))

# ── ❹ 전역 기본 글꼴로 설정 ─────────────────────
plt.rcParams["font.family"] = font_prop.get_name()   # 내부 메타데이터의 ‘폰트 이름’ 자동 추출
plt.rcParams["axes.unicode_minus"] = False           # 마이너스 기호 깨짐 방지

print("✅ Matplotlib 기본 글꼴:", font_prop.get_name())

CSV_PATH       = "total_brand_reviews_df.csv"

# 기본 제약
BUDGET_DEFAULT = 300_000      # ₩
TIME_DEFAULT   = 120          # person-hours

# 스캔 단위
STEP_BUDGET    = 10_000       # 1 만원
STEP_TIME      = 1            # 1 시간

# ----------------- 1. Data Pre-processing ------------------
df = pd.read_csv(CSV_PATH)

# --- 1-A. keyword ↔ damage -------------------------------
km_rows = []
for d in df["real_keywords_dmg_dict"].dropna():
    km_rows.extend(ast.literal_eval(d).items())

kw_damage = (pd.DataFrame(km_rows, columns=["keyword","damage"])
               .groupby("keyword", as_index=False)
               .sum())

# --- 1-B. keyword ↔ cost / time --------------------------
plan_rows = []
for d in df["keyword_plan_info"].dropna():
    for kw, info in ast.literal_eval(d).items():
        plan_rows.append({
            "keyword": kw,
            "cost":   pd.to_numeric(info.get("cost", ""), errors="coerce"),
            "time":   pd.to_numeric(info.get("time", ""), errors="coerce")
        })

kw_plan = (pd.DataFrame(plan_rows)
             .groupby("keyword")
             .agg({"cost":"max", "time":"max"})
             .fillna(0)
             .reset_index())

# --- 1-C. Final table for optimization -------------------
tbl = (kw_damage.merge(kw_plan, on="keyword", how="left")
                .fillna({"cost":0,"time":0})
                .query("cost>0 | time>0")          # 둘 다 0 → 제외
                .reset_index(drop=True))

print(f"[INFO] 최적화 대상 키워드 : {len(tbl)}개")

# ----------------- 2. MILP Model Builder -------------------
def build_model(budget, time_av):
    m = pulp.LpProblem("BrandRisk", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", tbl.index, cat="Binary")
    m += pulp.lpSum(tbl.loc[i,"damage"] * x[i] for i in tbl.index), "TotalDamage"
    m += pulp.lpSum(tbl.loc[i,"cost"]  * x[i] for i in tbl.index) <= budget, "budget"
    m += pulp.lpSum(tbl.loc[i,"time"]  * x[i] for i in tbl.index) <= time_av, "time"
    return m, x

def solve(budget=BUDGET_DEFAULT, time_av=TIME_DEFAULT):
    m, x = build_model(budget, time_av)
    m.solve(pulp.PULP_CBC_CMD(msg=False))
    sol = tbl.copy()
    sol["selected"] = [bool(x[i].value()) for i in tbl.index]
    obj = pulp.value(m.objective)
    return sol, obj, m

# ----------------- 3. LP-Relaxation Dual -------------------
def lp_dual(model):
    relax = model.copy()
    for v in relax.variables(): v.cat = pulp.LpContinuous
    relax.solve(pulp.PULP_CBC_CMD(msg=False))
    dualB = relax.constraints["budget"].pi
    dualT = relax.constraints["time"].pi
    return dualB, dualT

# ----------------- 4. Break-point Scanner ------------------
def breakpoint_scan(axis="budget",
                    base_B=BUDGET_DEFAULT, base_T=TIME_DEFAULT):
    """
    축소 → 증가 방향으로 훑으며
    • DamageReduced 값이 변하는 지점(break-point)만 리턴
    """
    step = STEP_BUDGET if axis=="budget" else STEP_TIME
    # 충분히 낮은 곳까지 감
    low = 0
    # 충분히 높은 곳까지
    high = base_B*2 if axis=="budget" else base_T*2

    pts = []
    prev_obj = None
    for val in range(low, high + step, step):
        B, T = (val, base_T) if axis=="budget" else (base_B, val)
        _, obj, _ = solve(B, T)
        if obj != prev_obj:              # break-point
            pts.append({"Budget":B, "Time":T, "DamageReduced":obj})
            prev_obj = obj
    return pd.DataFrame(pts)

def add_marginal(df, axis="budget"):
    col = "Budget" if axis=="budget" else "Time"
    df = df.sort_values(col).reset_index(drop=True)
    df["Δ"+col] = df[col].diff().fillna(np.nan)
    df["MarginalEfficiency"] = df["DamageReduced"].diff() / df["Δ"+col]
    return df

# -----------------------------------------------
# 6. 투자 효율 시각화 (Budget vs Time 한눈에)
# -----------------------------------------------
def nice_dual_plot(bpB, bpT):
    """
    bpB : break-point DataFrame(예산)
    bpT : break-point DataFrame(시간)
    두 곡선을 하나의 Figure에 겹쳐 그리고,
    각 break-point마다 누적 Damage 값을 라벨링.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(7,4))

    # ── [1] Budget 축 (아래 x-축) ───────────────────────
    ax.step(bpB["Budget"], bpB["DamageReduced"],
            where='post', linewidth=2,
            label="누적 감소 – 예산")

    # 예산 라벨
    for x, y in zip(bpB["Budget"], bpB["DamageReduced"]):
        ax.text(x, y, f"{int(y)}", va='bottom', ha='center',
                fontsize=8, color='tab:blue')

    # ── [2] Time 축 (위쪽 x-축) ───────────────────────
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())

    # bpT 의 Time 값을 Budget 스케일로 선형 사상
    x_t_proj = np.interp(bpT["Time"],
                         (bpT["Time"].min(), bpT["Time"].max()),
                         ax.get_xlim())

    # 위쪽 눈금 & 레이블
    ax_top.set_xticks(x_t_proj)
    ax_top.set_xticklabels(bpT["Time"])
    ax_top.set_xlabel("Available Time (h)")

    # 시간 곡선 (점선)
    ax.step(x_t_proj, bpT["DamageReduced"],
            where='post', linestyle='--', linewidth=2,
            label="누적 감소 – 시간", dashes=(5,3))

    # 시간 라벨 – 주황색으로 구분
    for x, y in zip(x_t_proj, bpT["DamageReduced"]):
        ax.text(x, y, f"{int(y)}", va='bottom', ha='center',
                fontsize=8, color='tab:orange')

    # ──[3] 공통 서식────────────────────────────────────
    ax.set_xlabel("Budget (₩)")
    ax.set_ylabel("Total Damage Reduced (pts)")
    ax.set_title("누적 브랜드-데미지 감소 vs 예산·가용시간")
    ax.grid(True, linestyle=':')
    ax.legend()
    plt.tight_layout()
    plt.show()

# ----------------- 5. Main Run ----------------------------
if __name__ == "__main__":
    # 5-A. Base MILP
    sol_tbl, best_obj, base_model = solve()
    picked = (sol_tbl.query("selected")
                        .sort_values("damage", ascending=False)
                        .reset_index(drop=True))
    print("\n=== 실행 우선순위 ===")
    print(picked[["keyword","damage","cost","time"]])

    # 5-B. LP dual (upper-bound)
    dB, dT = lp_dual(base_model)
    print(f"\n[LP-Relaxation dual] 예산 1원 ↑ → ≤ {dB:.3f}점, "
          f"시간 1h ↑ → ≤ {dT:.3f}점")

    # 5-C. Budget break-points
    bpB = add_marginal(breakpoint_scan("budget"), "budget")
    print("\n--- 예산 한계효율 (break-points) ---")
    print(bpB[["Budget","DamageReduced","MarginalEfficiency"]])

    # 5-D. Time break-points
    bpT = add_marginal(breakpoint_scan("time"), "time")
    print("\n--- 시간 한계효율 (break-points) ---")
    print(bpT[["Time","DamageReduced","MarginalEfficiency"]])

    nice_dual_plot(bpB, bpT)

