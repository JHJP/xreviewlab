"""
optimizer_gurobi_robust.py
― Robust MILP for review-damage mitigation
―――――――――――――――――――――――――――――――――
Public function
---------------
optimize(df_or_path,
         budget,
         hours_available,
         alpha_uncertainty=0.15,     # 15 % down-side uncertainty
         allowed_methods=None)
    -> rec_df, sens_dict
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Sequence, Optional

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

df              = pd.read_csv("reviews_sample_15.csv")
# ─────────────────────────────────────────
# 0. CONSTANTS
# ─────────────────────────────────────────
METHODS: List[str] = [
    "text_reply", "chat_followup", "phone_call", "discount_coupon", "vip_apology"
]
METHOD_COST: Dict[str, float] = {
    "text_reply":      1_000,
    "chat_followup":   5_000,
    "phone_call":     15_000,
    "discount_coupon":25_000,
    "vip_apology":    50_000,
}
CALL_HOURS_PER = 0.5
CHAT_HOURS_PER = 0.25

# ─────────────────────────────────────────
# 1. DAMAGE SCORE  (기존 그대로)
# ─────────────────────────────────────────
def compute_damage(df: pd.DataFrame) -> np.ndarray:
    emotion  = df.get("high_emotional_language", 0).fillna(0).astype(int)
    warn_imp = df.get("indirect_warning",        0).fillna(0).astype(int)

    length = df.get("review_length",         0).fillna(0).astype(float)
    photo  = df.get("log_compliment_photos", 0).fillna(0).astype(float)

    dmg = 0.002*length + 0.219*photo + 0.135*warn_imp + 0.177*emotion - 0.068*warn_imp*emotion

    return dmg

# ─────────────────────────────────────────
# 2. NOMINAL α MATRIX
# ─────────────────────────────────────────
def build_alpha_nominal(n_reviews: int) -> np.ndarray:
    α = np.zeros((n_reviews, len(METHODS)))
    α[:, METHODS.index("text_reply")]      = 0.10
    α[:, METHODS.index("chat_followup")]   = 0.20
    α[:, METHODS.index("phone_call")]      = 0.30
    α[:, METHODS.index("discount_coupon")] = 0.40
    α[:, METHODS.index("vip_apology")]     = 0.60
    return α

# ─────────────────────────────────────────
# 3. ROBUST MILP
# ─────────────────────────────────────────
def build_milp_robust(
    damage: np.ndarray,
    α_nom: np.ndarray,
    α_uncertainty: float,          # e.g. 0.15  →  ±15 %
    budget: float,
    hours_avail: float,
    allowed_methods: Optional[Sequence[str]] = None,
) -> gp.Model:

    allowed_set = set(allowed_methods or METHODS)
    n_r, n_m = α_nom.shape

    # ― 3.1  Worst-case (lower-bound) α ――――――――――――――
    α_min = np.maximum(α_nom * (1 - α_uncertainty), 0.0)

    m = gp.Model("MinNetDamage_Robust")
    m.Params.OutputFlag = 0

    # ― 3.2  Decision variables ―――――――――――――――――――
    x = m.addVars(n_r, n_m, vtype=GRB.BINARY, name="x")

    # Disable methods not allowed
    for idx, meth in enumerate(METHODS):
        if meth not in allowed_set:
            for r in range(n_r):
                x[r, idx].ub = 0

    # ― 3.3  Objective (worst-case α = α_min) ―――――――
    total_dmg = damage.sum()
    mitigated = gp.quicksum(α_min[r, m_] * damage[r] * x[r, m_]
                            for r in range(n_r) for m_ in range(n_m))
    m.setObjective(total_dmg - mitigated, GRB.MINIMIZE)

    # ― 3.4  Constraints ―――――――――――――――――――――――――――
    # (a) One action per review
    for r in range(n_r):
        m.addConstr(x.sum(r, "*") <= 1, name=f"Single_{r}")

    # (b) Budget
    m.addConstr(
        gp.quicksum(METHOD_COST[METHODS[m_]] * x[r, m_]
                    for r in range(n_r) for m_ in range(n_m))
        <= budget, name="Budget"
    )

    # (c) Direct-response hours
    call_idx = METHODS.index("phone_call")
    chat_idx = METHODS.index("chat_followup")
    total_hours = (CALL_HOURS_PER * gp.quicksum(x[r, call_idx] for r in range(n_r))
                   + CHAT_HOURS_PER * gp.quicksum(x[r, chat_idx] for r in range(n_r)))
    m.addConstr(total_hours <= hours_avail, name="DirectResponseHours")

    return m

def solve_milp(model: gp.Model):
    model.optimize()
    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi status = {model.Status}")
    # Collect chosen actions
    sol = {(int(v.varName.split("[")[1].split(",")[0]),
            int(v.varName.split(",")[1].rstrip("]"))): int(v.X + 0.5)
           for v in model.getVars() if v.VarName.startswith("x[")}
    return sol, model.ObjVal

# ─────────────────────────────────────────
# 4.  ROBUST LP-SENSITIVITY  ← ① 새 함수
# ─────────────────────────────────────────
def robust_lp_sensitivity(robust_m: gp.Model):
    """Relaxed LP 기준 shadow price·RHS range 리턴 (constraint 이름 없거나 GurobiError 발생 시 0 반환)"""
    lp = robust_m.relax()
    lp.optimize()
    try:
        budget_con = lp.getConstrByName("Budget")
    except Exception:
        budget_con = None
    try:
        time_con   = lp.getConstrByName("DirectResponseHours")
    except Exception:
        time_con = None
    return {
        "shadow_prices": {
            "budget": getattr(budget_con, "Pi", 0.0) if budget_con is not None else 0.0,
            "time":   getattr(time_con,   "Pi", 0.0) if time_con is not None else 0.0,
        },
        "rhs_ranges": {
            "budget": (
                getattr(budget_con, "SARHSLow", 0.0) if budget_con is not None else 0.0,
                getattr(budget_con, "SARHSUp",  0.0) if budget_con is not None else 0.0,
            ),
            "time": (
                getattr(time_con, "SARHSLow", 0.0) if time_con is not None else 0.0,
                getattr(time_con, "SARHSUp",  0.0) if time_con is not None else 0.0,
            ),
        },
    }


# ─────────────────────────────────────────
# 5. SIMPLE “SENSITIVITY” : Δα 스캔 (선택) 
# ─────────────────────────────────────────
def alpha_sensitivity_scan(
    damage: np.ndarray,
    α_nom: np.ndarray,
    budget: float,
    hours_avail: float,
    allowed_methods: Sequence[str] | None,
    scan_grid: Sequence[float] = (0.00, 0.05, 0.10, 0.15, 0.20)
):
    """Δ(%)를 바꾸며 Robust 해를 다시 푼 뒤 목적값을 기록."""
    rows = []
    for Δ in scan_grid:
        m = build_milp_robust(damage, α_nom, Δ,
                              budget, hours_avail, allowed_methods)
        _, obj = solve_milp(m)
        rows.append({"alpha_uncertainty": Δ, "worst_case_obj": obj})
    return pd.DataFrame(rows)

# ─────────────────────────────────────────
# 6. PUBLIC API
# ─────────────────────────────────────────
def optimize(
    df_reviews: pd.DataFrame | str,
    budget: float,
    hours_available: float,
    alpha_uncertainty: float = 0.15,
    allowed_methods: Optional[Sequence[str]] = None,
    want_alpha_scan: bool = False
):
    df = pd.read_csv(df_reviews) if isinstance(df_reviews, str) else df_reviews.copy()
    if df.empty:
        raise ValueError("No reviews supplied.")

    # 5.1  Pre-processing
    damage = compute_damage(df)
    α_nom  = build_alpha_nominal(len(df))

    # 5.2  Robust MILP
    milp = build_milp_robust(damage, α_nom, alpha_uncertainty,
                             budget, hours_available, allowed_methods)
    sol, obj_val = solve_milp(milp)

    # 5.2b  Nominal MILP (Δ=0) for shadow price insight
    milp_nom = build_milp_robust(damage, α_nom, 0.0,
                                 budget, hours_available, allowed_methods)
    shadow_nom = robust_lp_sensitivity(milp_nom)

    # slack 값도 추출 (nominal 기준)
    lp_nom = milp_nom.relax()
    lp_nom.optimize()
    try:
        budget_con = lp_nom.getConstrByName("Budget")
        budget_slack = getattr(budget_con, "Slack", None)
    except Exception:
        budget_slack = None
    try:
        time_con = lp_nom.getConstrByName("DirectResponseHours")
        time_slack = getattr(time_con, "Slack", None)
    except Exception:
        time_slack = None

    # 5.3  Optional α-scan
    sens_alpha = None
    if want_alpha_scan:
        sens_alpha = alpha_sensitivity_scan(
            damage, α_nom, budget, hours_available,
            allowed_methods, scan_grid=(0.0, 0.05, 0.10, 0.15, 0.20, 0.30)
        )

    # 5.4  Result assembly (for UI)
    ids   = df.get("review_id", pd.Series(range(len(df))))
    texts = df.get("review_text", df.get("text", pd.Series([""]*len(df))))

    rows = []
    for r in range(len(df)):
        chosen = next((METHODS[m] for m in range(len(METHODS))
                       if sol.get((r, m), 0) == 1), None)
        if chosen:
            rows.append(dict(
                review_id    = int(ids.iloc[r]),
                risk_score   = float(damage[r]),  
                method       = chosen,
                review_text  = str(texts.iloc[r])[:1_000],
            ))

    rec_df = pd.DataFrame(rows)
    # 날짜 컬럼 보장: review_id 기준 merge, 없으면 순서대로 복사
    if not rec_df.empty:
        for col in ['date', '날짜', 'DATE']:
            if col in df.columns:
                if 'review_id' in df.columns and 'review_id' in rec_df.columns:
                    # review_id 기준 merge
                    date_map = df[['review_id', col]].copy()
                    date_map = date_map.rename(columns={col: 'date'})
                    rec_df = rec_df.merge(date_map, on='review_id', how='left')
                else:
                    # 순서대로 복사 (길이 맞을 때만)
                    if len(rec_df) == len(df):
                        rec_df['date'] = df[col].values
                    else:
                        rec_df['date'] = None
                break
    # 위험도 0~100으로 평준화
    if not rec_df.empty:
        min_risk, max_risk = rec_df['risk_score'].min(), rec_df['risk_score'].max()
        if min_risk == max_risk:
            rec_df['risk_score_norm'] = 100
        else:
            rec_df['risk_score_norm'] = ((rec_df['risk_score'] - min_risk) / (max_risk - min_risk) * 100).round(1)

    sens_dict = {
        "obj_worst_case":   obj_val,
        "alpha_uncertainty":alpha_uncertainty,
        "budget_slack": budget_slack,
        "time_slack": time_slack,
    }
    if sens_alpha is not None:
        sens_dict["alpha_scan"] = sens_alpha.to_dict("records")

    sens_dict["shadow_prices"] = robust_lp_sensitivity(milp)
    sens_dict["shadow_prices_nominal"] = shadow_nom

    return rec_df, sens_dict

# ─────────────────────────────────────────
# 6. CLI TEST (optional)
# ─────────────────────────────────────────
if __name__ == "__main__":
    import argparse, json, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=50_000)
    ap.add_argument("--hours",  type=float, default=5)
    ap.add_argument("--alpha_uncertainty", type=float, default=0.15)
    ap.add_argument("--alpha_scan", action="store_true")
    args = ap.parse_args()

    try:
        rec, sens = optimize(
            df, args.budget, args.hours,
            alpha_uncertainty=args.alpha_uncertainty,
            want_alpha_scan=args.alpha_scan
        )
        # print(json.dumps(rec.to_dict("records"), ensure_ascii=False, indent=2))
        print("\n# Robust-sensitivity")
        print(json.dumps(sens, indent=2))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
