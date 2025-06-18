# ----------------- 0. Imports -----------------------------
import pandas as pd, numpy as np, ast, math, functools
import pulp, matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from pathlib import Path
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm
import matplotlib.pyplot as plt, matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

# ── ❶ 내 폰트 파일 경로 지정 ───────────────────────────────
MY_FONT_PATH = Path(
    r"C:\Users\papag\OneDrive\desktop\Business\ORM\App\MVP_gurobi\fonts\GamjaFlower-Regular.ttf"
)  # ← 필요 시 변경

# ── ❷ Matplotlib 폰트 등록 ───────────────────────────────
fm.fontManager.addfont(str(MY_FONT_PATH))
font_prop = fm.FontProperties(fname=str(MY_FONT_PATH))
plt.rcParams["font.family"] = font_prop.get_name()
plt.rcParams["axes.unicode_minus"] = False

# ❶ 퍼플(#5F3BFF) → 화이트 → 그린(#00C49A) 커스텀 컬러맵
_brand_cmap = LinearSegmentedColormap.from_list(
    "brand_pg", ["#5F3BFF", "#FFFFFF", "#00C49A"], N=256)


# ----------------- 1. Load data ---------------------------
CSV_PATH = "total_brand_reviews_df.csv"
df = pd.read_csv(CSV_PATH)

# 1-A. (상품, 키워드) ↔ damage ------------------------------
km_rows = []
for _, r in df[["prd_name", "real_keywords_dmg_dict"]].dropna().iterrows():
    for kw, dmg in ast.literal_eval(r["real_keywords_dmg_dict"]).items():
        km_rows.append({"prd_name": r["prd_name"], "keyword": kw, "damage": dmg})
kw_damage = pd.DataFrame(km_rows).groupby(["prd_name", "keyword"], as_index=False).sum()

# 1-B. (상품, 키워드) ↔ cost / time -------------------------
plan_rows = []
for _, r in df[["prd_name", "keyword_plan_info"]].dropna().iterrows():
    for kw, info in ast.literal_eval(r["keyword_plan_info"]).items():
        plan_rows.append({
            "prd_name": r["prd_name"],
            "keyword": kw,
            "cost": pd.to_numeric(str(info.get("cost", "")).replace(",", ""), errors="coerce"),
            "time": pd.to_numeric(str(info.get("time", "")).replace(",", ""), errors="coerce"),
        })
kw_plan = (pd.DataFrame(plan_rows)
           .groupby(["prd_name", "keyword"])
           .agg({"cost": "max", "time": "max"})
           .fillna(0)
           .reset_index())

# 1-C. 최종 테이블 ------------------------------------------
tbl = (kw_damage.merge(kw_plan, on=["prd_name", "keyword"], how="left")
                  .fillna({"cost": 0, "time": 0})
                  .query("cost > 0 | time > 0")        # 둘 다 0 인 항목은 제외
                  .reset_index(drop=True))

TOTAL_DAMAGE = tbl["damage"].sum()

# ----------------- 2. 파라미터 ------------------------------
def _nice_round(x, base=10_000):
    return int(math.ceil(x / base) * base)

def _derive_dynamic_params(t: pd.DataFrame):
    n = len(t)
    median_c = t["cost"].median()
    step_B = _nice_round(max(10_000, median_c / 4), 10_000)
    B_def = _nice_round(min(t["cost"].sum()*0.3, median_c*max(10, n*0.1)), step_B)

    positive_times = t.loc[t["time"] > 0, "time"].astype(int)
    step_T = max(1, functools.reduce(math.gcd, positive_times) if len(positive_times) else 1)
    median_t = t["time"].median()
    T_def = int(min(t["time"].sum()*0.3, median_t*max(10, n*0.1)))

    return B_def, T_def, step_B, step_T

BDEF, TDEF, STEP_B, STEP_T = _derive_dynamic_params(tbl)
print(f"[AUTO] 기본 예산  : {BDEF:,.0f} 원")
print(f"[AUTO] 기본 시간  : {TDEF:,.0f} h")
print(f"[AUTO] 예산 단위 : {STEP_B:,.0f} 원")
print(f"[AUTO] 시간 단위 : {STEP_T} h")

# ----------------- 3. 컬럼 rename ---------------------------
COL_MAIN = {"prd_name": "상품", "keyword": "문제 키워드", "damage": "위험 점수",
            "cost": "필요 예산(원)", "time": "필요 시간(시간)", "selected": "이번에 처리"}
COL_BP_B = {"Budget": "투입 예산(원)", "RiskScore": "브랜드 평판 위험점수",
            "MarginalEfficiency": f"추가 {STEP_B//10_000:d} 만원당 위험 감소(점)"}
COL_BP_T = {"Time": "투입 시간(시간)", "RiskScore": "브랜드 평판 위험점수",
            "MarginalEfficiency": f"추가 {STEP_T} 시간당 위험 감소(점)"}

# ----------------- 4. 내부 단위 ------------------------------
tbl["cost_unit"] = np.ceil(tbl["cost"] / STEP_B).astype(int)
tbl["time_unit"] = np.ceil(tbl["time"] / STEP_T).astype(int)

# ----------------- 5. MILP ---------------------------------
def _build_model(bu, tu):
    m = pulp.LpProblem("BrandRisk", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", tbl.index, cat="Binary")
    m += pulp.lpSum(tbl.loc[i, "damage"] * x[i] for i in tbl.index)
    m += pulp.lpSum(tbl.loc[i, "cost_unit"] * x[i] for i in tbl.index) <= bu, "budget"
    m += pulp.lpSum(tbl.loc[i, "time_unit"] * x[i] for i in tbl.index) <= tu, "time"
    return m, x

def solve(budget=None, time_av=None, *, step_B=STEP_B, step_T=STEP_T):
    """
    budget, time_av 가 None ⇒ 기본값(BDEF, TDEF) 사용
    0은 **실제로 0** 으로 처리한다.
    """
    bu = int(((BDEF if budget is None else budget)   ) // step_B)
    tu = int(((TDEF if time_av is None else time_av) ) // step_T)
    m, x = _build_model(bu, tu)
    m.solve(pulp.PULP_CBC_CMD(msg=False))
    sol = tbl.copy()
    sol["selected"] = [bool(x[i].value()) for i in tbl.index]
    return sol, pulp.value(m.objective)   # 제거된 damage

# ----------------- 6. Break-points -------------------------
def _scan(axis, step_B, step_T):
    """
    axis ∈ {'budget','time'}
    스캔하는 축만 0~High 로, 다른 축은 사실상 제약이 없도록 '최대치' 로 둔다.
    """
    total_cost_units = tbl["cost_unit"].sum()
    total_time_units = tbl["time_unit"].sum()

    if axis == "budget":
        high   = total_cost_units * step_B
        step   = step_B
        fixedT = total_time_units * step_T   # 충분히 큰 시간
    else:
        high   = total_time_units * step_T
        step   = step_T
        fixedB = total_cost_units * step_B   # 충분히 큰 예산

    pts, prev = [], None
    for val in range(0, int(high + step), int(step)):
        if axis == "budget":
            B, T = val, fixedT
        else:
            B, T = fixedB, val
        _, dmg_removed = solve(B, T, step_B=step_B, step_T=step_T)
        risk_now = TOTAL_DAMAGE - dmg_removed
        if risk_now != prev:
            pts.append({"Budget": B, "Time": T, "RiskScore": risk_now})
            prev = risk_now
    return pd.DataFrame(pts)

def _add_marginal(df, axis):
    col = "Budget" if axis == "budget" else "Time"
    df = df.sort_values(col).reset_index(drop=True)
    df["Δ" + col] = df[col].diff()
    df["RiskDecrease"] = df["RiskScore"].shift(1) - df["RiskScore"]
    df["MarginalEfficiency"] = df["RiskDecrease"] / df["Δ" + col]
    return df

# ----------------- 7. Plot --------------------------------
def _dual_plot(bpB, bpT):
    import numpy as np
    fig, ax = plt.subplots(figsize=(10, 5), dpi=110)

    # ── ❶ 예산 축 ──────────────────────────────────────────
    ax.step(bpB["투입 예산(원)"], bpB["브랜드 평판 위험점수"],
            where="post", lw=2.2, label="예산을 늘렸을 때")
    for x, y in zip(bpB["투입 예산(원)"], bpB["브랜드 평판 위험점수"]):
        ax.annotate(f"{int(y)}", (x, y),
                    textcoords="offset points", xytext=(0, 6),
                    ha="center", va="bottom",
                    fontsize=7, fontweight="bold", color="tab:blue",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=.8, ec="none"))

    # ── ❷ 시간 축(상단) ────────────────────────────────────
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    xproj = np.interp(
        bpT["투입 시간(시간)"],
        (bpT["투입 시간(시간)"].min(), bpT["투입 시간(시간)"].max()),
        ax.get_xlim(),
    )
    ax2.set_xticks(xproj)
    ax2.set_xticklabels(bpT["투입 시간(시간)"].astype(int))
    ax2.set_xlabel("투입 시간 (h)")

    ax.step(xproj, bpT["브랜드 평판 위험점수"],
            where="post", ls="--", lw=2.2, label="시간을 늘렸을 때", dashes=(6, 3))
    for x_, y in zip(xproj, bpT["브랜드 평판 위험점수"]):
        ax.annotate(f"{int(y)}", (x_, y),
                    textcoords="offset points", xytext=(0, -10),
                    ha="center", va="top",
                    fontsize=7, fontweight="bold", color="tab:orange",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=.8, ec="none"))

    # ── ❸ 축 설정 & 스타일 ─────────────────────────────────
    comma_fmt = FuncFormatter(lambda x, pos: f"{x:,.0f}")
    ax.xaxis.set_major_formatter(comma_fmt)
    ax.set_xlabel("투입 예산 (₩)")
    ax.set_ylabel("브랜드 평판 위험점수")
    ax.set_xlim(left=-STEP_B)                     # 0 지점 여백
    ax.grid(True, ls=":", alpha=.6)
    ax.legend()
    plt.tight_layout()
    return fig

# ----------------- 7-A. 위험표(surface) 만들기 -----------------
def _build_risk_surface(step_B=None, step_T=None,
                        max_B=None, max_T=None, progress=True):
    """
    그리드 해상도를 (기본*multiplier) 로 축소해 연산량을 줄인다.
    """
    # ── ❶ STEP 자동 확대: 기본 STEP 의 k배
    step_B = step_B or STEP_B * 3      # ← multiplier 조정
    step_T = step_T or STEP_T * 4

    max_B = max_B or tbl["cost"].sum()
    max_T = max_T or tbl["time"].sum()

    B_vals = np.arange(0, max_B + step_B, step_B, dtype=int)
    T_vals = np.arange(0, max_T + step_T, step_T, dtype=int)
    Z = np.zeros((len(T_vals), len(B_vals)))

    iterator = tqdm(list(enumerate(T_vals)), disable=not progress,
                    desc="💡 building risk surface", unit="rows")

    for i, T in iterator:
        for j, B in enumerate(B_vals):
            _, dmg_removed = solve(B, T, step_B=STEP_B, step_T=STEP_T)
            Z[i, j] = TOTAL_DAMAGE - dmg_removed

    return B_vals, T_vals, Z

# ----------------- 7-B. Contour + Heat-map -------------------
def plot_heatmap_contour(B_vals, T_vals, Z,
                         user_B=None, user_T=None,
                         fig_size=(9, 7), dpi=130):
    """
    Budget × Time 리스크 맵 — 앱 UI 친화형.
    user_B, user_T: 입력창에서 사용자가 넣은 숫자 → 포인트로 표시.
    """
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)

    # ── Heat-map ─────────────────────────────────────────────
    im = ax.imshow(Z, origin="lower",
                   extent=[B_vals.min(), B_vals.max(),
                           T_vals.min(), T_vals.max()],
                   aspect="auto", cmap=_brand_cmap)

    # ── Contour + 숫자라벨 (화이트 스트로크) ────────────────
    levels = sorted({int(v) for v in np.linspace(Z.min(), Z.max(), 15)})
    CS = ax.contour(B_vals, T_vals, Z, levels=levels,
                    colors="black", linewidths=0.7, alpha=.8)

    fmt = {lev: f"{lev:d}" for lev in levels}
    txts = ax.clabel(CS, levels=levels, fmt=fmt, fontsize=8, inline=True)
    # 흰색 외곽선 효과
    for t in txts:
        t.set_path_effects([pe.Stroke(linewidth=2.5, foreground="white"),
                            pe.Normal()])

    # ── 사용자 입력 지점 표시 ────────────────────────────────
    if user_B is not None and user_T is not None:
        ax.scatter(user_B, user_T, s=80, c="#1E88E5", marker="o",
                   edgecolors="white", linewidths=1.2, zorder=5)
        ax.annotate("현재 입력값",
                    (user_B, user_T), xytext=(10, -15),
                    textcoords="offset points",
                    color="#1E88E5", fontsize=9, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#1E88E5"))

    # ── 축 포맷 ─────────────────────────────────────────────
    comma_fmt = FuncFormatter(lambda x, p: f"{x:,.0f}")
    ax.xaxis.set_major_formatter(comma_fmt)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, p: f"{y:.0f}"))

    ax.set_xlabel("투입 예산 (₩)")
    ax.set_ylabel("투입 시간 (h)")
    ax.set_title("Budget × Time 별 예상 브랜드 위험점수", pad=15, fontsize=14,
                 fontweight="bold")

    # ── Color-bar ───────────────────────────────────────────
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Risk Score", rotation=270, labelpad=18)

    plt.tight_layout()
    return fig


# ----------------- 8. 외부 API -----------------------------
def run_optimizer(budget=None, time_av=None):
    sol, _ = solve(budget, time_av)
    picked = (sol.query("selected")
                   .sort_values("damage", ascending=False)
                   .reset_index(drop=True)
                   .rename(columns=COL_MAIN)
                   [["상품", "문제 키워드", "위험 점수", "필요 예산(원)", "필요 시간(시간)"]])
    
    # ① 위험표 계산 (필요 시 max_B, max_T 줄여서 속도 조절)
    B_vals, T_vals, Z = _build_risk_surface()

    # 사용자 폼 값 (예: 2 500 000원, 200h)
    fig = plot_heatmap_contour(B_vals, T_vals, Z,
                               user_B=budget, user_T=time_av)

    return picked, fig

# ------------------------- End -----------------------------
if __name__ == "__main__":
    picked, fig = run_optimizer(2_500_000, 100)
    print("\n[우선 처리 리스트]")
    print(picked.to_string(index=False))

    plt.show()

    input("\n엔터를 누르면 프로그램을 종료합니다 ▶ ")
