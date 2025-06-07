"""
optimizer_gurobi_robust.py
― Robust MILP for review-damage mitigation
―――――――――――――――――――――――――――――――――
This version implements *Budgeted Uncertainty* (Γ-robustness) in the
objective following Bertsimas & Sim (2004) [1].  
When `gamma_budget` is **None** the model reverts to the (old)
“all-coefficients worst-case” behaviour; choose a smaller Γ to allow only
a limited number of α-coefficients to deviate adversely, which yields
meaningful dual values (shadow prices).

Public function
---------------
optimize(df_or_path,
         budget,
         hours_available,
         alpha_uncertainty=0.15,   # ±15 % down-side
         gamma_budget=None,        # Γ (≤ #coeffs). None → full worst-case
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
# helper: constraint lookup that never crashes
# ─────────────────────────────────────────
def _find_constr(model: gp.Model, cname: str):
    """
    Robust version of getConstrByName:
    - tries fast hashing first
    - falls back to linear search if the hash map is missing
    """
    try:
        c = model.getConstrByName(cname)
        return c
    except gp.GurobiError as e:
        if "No constraint names available to index" not in str(e):
            raise                              # 다른 오류면 그대로 리Raise
        # 해시가 없을 때는 모든 제약을 순회 검색
        for c in model.getConstrs():
            if c.ConstrName == cname:
                return c
        return None
# ─────────────────────────────────────────
# 0. CONSTANTS
# ─────────────────────────────────────────
METHODS: List[str] = [
    "text_reply", "chat_followup", "phone_call",
    "discount_coupon", "vip_apology"
]
METHOD_COST: Dict[str, float] = {
    "text_reply":       1_000,
    "chat_followup":    5_000,
    "phone_call":      15_000,
    "discount_coupon": 25_000,
    "vip_apology":     50_000,
}
CALL_HOURS_PER = 0.5
CHAT_HOURS_PER = 0.25

# ─────────────────────────────────────────
# 1. DAMAGE SCORE  (unchanged)
# ─────────────────────────────────────────
def compute_damage(df: pd.DataFrame) -> np.ndarray:
    emotion  = df.get("high_emotional_language", 0).fillna(0).astype(int)
    warn_imp = df.get("indirect_warning",        0).fillna(0).astype(int)

    length = df.get("review_length",         0).fillna(0).astype(float)
    photo  = df.get("log_compliment_photos", 0).fillna(0).astype(float)

    dmg = 0.002*length + 0.219*photo + 0.135*warn_imp + 0.177*emotion - 0.068*warn_imp*emotion
    return dmg.to_numpy()

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
# 3. ROBUST MILP  (Γ-budgeted objective)
# ─────────────────────────────────────────
def build_milp_robust(
    damage: np.ndarray,
    α_nom: np.ndarray,
    α_uncertainty: float,               # e.g. 0.15 → ±15 %
    budget: float,
    hours_avail: float,
    allowed_methods: Optional[Sequence[str]] = None,
    gamma_budget: Optional[float] = None,  # Γ (None → full worst-case)
) -> gp.Model:

    allowed_set = set(allowed_methods or METHODS)
    n_r, n_m = α_nom.shape
    K = n_r * n_m                      # total # coefficients
    Γ = K if gamma_budget is None else float(gamma_budget)

    # Nominal mitigation coefficients  c_{r,m} = α_nom * damage
    c = α_nom * damage.reshape(-1, 1)           # shape (n_r, n_m)
    Δ = α_uncertainty * c                       # max downward deviation

    m = gp.Model("MinNetDamage_Robust")
    m.Params.OutputFlag = 0

    # ― 3.1  Decision variables ――――――――――――――――――――――――――――――――――――――――
    x = m.addVars(n_r, n_m, vtype=GRB.BINARY, name="x")   # mitigation actions
    θ = m.addVar(lb=0.0, name="theta")                    # master deviation
    p = m.addVars(n_r, n_m, lb=0.0, name="p")             # individual parts

    # Disable methods not allowed
    for idx, meth in enumerate(METHODS):
        if meth not in allowed_set:
            for r in range(n_r):
                x[r, idx].ub = 0

    # ― 3.2  Uncertainty-set constraints (Bertsimas–Sim) ――――――――――――――――――
    for r in range(n_r):
        for m_ in range(n_m):
            if Δ[r, m_] > 0:
                m.addConstr(
                    θ + p[r, m_] >= Δ[r, m_] * x[r, m_],
                    name=f"Unc_{r}_{m_}"
                )
            else:  # zero-deviation entries → tighten p to 0
                m.addConstr(p[r, m_] == 0, name=f"UncZero_{r}_{m_}")

    # ― 3.3  Objective function (Γ-robust) ――――――――――――――――――――――――――――――
    total_dmg = damage.sum()
    mitig_nom = gp.quicksum(c[r, m_] * x[r, m_] for r in range(n_r) for m_ in range(n_m))
    extra_worst = Γ * θ + gp.quicksum(p[r, m_] for r in range(n_r) for m_ in range(n_m))
    m.setObjective(total_dmg - mitig_nom + extra_worst, GRB.MINIMIZE)

    # ― 3.4  Operational constraints ――――――――――――――――――――――――――――――――――
    # (a) One action per review
    for r in range(n_r):
        m.addConstr(x.sum(r, "*") <= 1, name=f"Single_{r}")

    # (b) Budget
    m.addConstr(
        gp.quicksum(METHOD_COST[METHODS[m_]] * x[r, m_]
                    for r in range(n_r) for m_ in range(n_m))
        <= budget,
        name="Budget"
    )

    # (c) Direct-response hours
    call_idx = METHODS.index("phone_call")
    chat_idx = METHODS.index("chat_followup")
    total_hours = (
        CALL_HOURS_PER * gp.quicksum(x[r, call_idx] for r in range(n_r))
        + CHAT_HOURS_PER * gp.quicksum(x[r, chat_idx] for r in range(n_r))
    )
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
# 4.  ROBUST LP-SENSITIVITY  (unchanged)
# ─────────────────────────────────────────
def robust_lp_sensitivity(robust_m: gp.Model):
    lp = robust_m.relax()
    lp.optimize()

    budget_con = _find_constr(lp, "Budget")
    time_con   = _find_constr(lp, "DirectResponseHours")

    return {
        "shadow_prices": {
            "budget": getattr(budget_con, "Pi", 0.0) if budget_con else 0.0,
            "time":   getattr(time_con,   "Pi", 0.0) if time_con   else 0.0,
        },
        "rhs_ranges": {
            "budget": (
                getattr(budget_con, "SARHSLow", 0.0) if budget_con else 0.0,
                getattr(budget_con, "SARHSUp",  0.0) if budget_con else 0.0,
            ),
            "time": (
                getattr(time_con, "SARHSLow", 0.0) if time_con else 0.0,
                getattr(time_con, "SARHSUp",  0.0) if time_con else 0.0,
            ),
        },
    }

# ─────────────────────────────────────────
# 5. SIMPLE α-UNCERTAINTY SCAN  (unchanged)
# ─────────────────────────────────────────
def alpha_sensitivity_scan(
    damage: np.ndarray,
    α_nom: np.ndarray,
    budget: float,
    hours_avail: float,
    allowed_methods: Sequence[str] | None,
    scan_grid: Sequence[float] = (0.00, 0.05, 0.10, 0.15, 0.20),
    gamma_budget: Optional[float] = None,
):
    """Solve the robust MILP for several Δ% values and record objective."""
    rows = []
    for Δ in scan_grid:
        m = build_milp_robust(
            damage, α_nom, Δ, budget, hours_avail,
            allowed_methods, gamma_budget=gamma_budget
        )
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
    gamma_budget: Optional[float] = None,
    allowed_methods: Optional[Sequence[str]] = None,
    want_alpha_scan: bool = False
):
    df = pd.read_csv(df_reviews) if isinstance(df_reviews, str) else df_reviews.copy()
    if df.empty:
        raise ValueError("No reviews supplied.")

    # 6.1  Pre-processing
    damage = compute_damage(df)
    α_nom  = build_alpha_nominal(len(df))

    # 6.2  Robust MILP
    milp = build_milp_robust(
        damage, α_nom, alpha_uncertainty, budget, hours_available,
        allowed_methods, gamma_budget
    )
    sol, obj_val = solve_milp(milp)

    # 6.2b  Nominal MILP (Δ=0, Γ full) for dual insight
    milp_nom = build_milp_robust(
        damage, α_nom, 0.0, budget, hours_available,
        allowed_methods, gamma_budget=None
    )
    shadow_nom = robust_lp_sensitivity(milp_nom)

    # slack 값도 추출 (nominal 기준)
    lp_nom = milp_nom.relax()
    lp_nom.optimize()
    budget_con = _find_constr(lp_nom, "Budget")
    time_con   = _find_constr(lp_nom, "DirectResponseHours")

    budget_slack = getattr(budget_con, "Slack", None) if budget_con else None
    time_slack   = getattr(time_con,   "Slack", None) if time_con   else None

    # 6.3  Optional α-scan
    sens_alpha = None
    if want_alpha_scan:
        sens_alpha = alpha_sensitivity_scan(
            damage, α_nom, budget, hours_available,
            allowed_methods, scan_grid=(0.0, 0.05, 0.10, 0.15, 0.20, 0.30),
            gamma_budget=gamma_budget
        )

    # 6.4  Assemble recommendation dataframe
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

    # date column handling (unchanged)
    if not rec_df.empty:
        for col in ['date', '날짜', 'DATE']:
            if col in df.columns:
                if 'review_id' in df.columns and 'review_id' in rec_df.columns:
                    date_map = df[['review_id', col]].rename(columns={col: 'date'})
                    rec_df = rec_df.merge(date_map, on='review_id', how='left')
                elif len(rec_df) == len(df):
                    rec_df['date'] = df[col].values
                else:
                    rec_df['date'] = None
                break

    # risk-score normalisation
    if not rec_df.empty:
        min_risk, max_risk = rec_df['risk_score'].min(), rec_df['risk_score'].max()
        rec_df['risk_score_norm'] = 100 if min_risk == max_risk else (
            (rec_df['risk_score'] - min_risk) / (max_risk - min_risk) * 100
        ).round(1)

    sens_dict = {
        "obj_worst_case":    obj_val,
        "alpha_uncertainty": alpha_uncertainty,
        "gamma_budget":      gamma_budget,
        "budget_slack":      budget_slack,
        "time_slack":        time_slack,
        "shadow_prices":     robust_lp_sensitivity(milp),
        "shadow_prices_nominal": shadow_nom,
    }
    if sens_alpha is not None:
        sens_dict["alpha_scan"] = sens_alpha.to_dict("records")

    return rec_df, sens_dict

# ─────────────────────────────────────────
# 7. CLI TEST (optional)
# ─────────────────────────────────────────
if __name__ == "__main__":
    import argparse, json, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=50_000)
    ap.add_argument("--hours",  type=float, default=5)
    ap.add_argument("--alpha_uncertainty", type=float, default=0.15)
    ap.add_argument("--gamma_budget", type=float, default=5)
    ap.add_argument("--alpha_scan", action="store_true")
    args = ap.parse_args()

    try:
        rec, sens = optimize(
            df, args.budget, args.hours,
            alpha_uncertainty=args.alpha_uncertainty,
            gamma_budget=args.gamma_budget,
            want_alpha_scan=args.alpha_scan
        )
        # print(json.dumps(rec.to_dict("records"), ensure_ascii=False, indent=2))
        print("\n# Robust-sensitivity")
        print(json.dumps(sens, indent=2))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
