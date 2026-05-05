"""
LendingClub Loan Default Predictor
Python Shiny app — loads calibrated LR and XGBoost models trained in NB04.
"""

import json
import pickle
from pathlib import Path

import numpy as np
from shiny import App, reactive, render, ui

# ---------------------------------------------------------------------------
# Load artifacts
# ---------------------------------------------------------------------------
MODEL_DIR = Path(__file__).parent / "models"

with open(MODEL_DIR / "thresholds.json") as f:
    THRESHOLDS = json.load(f)

with open(MODEL_DIR / "medians.json") as f:
    MEDIANS = json.load(f)

with open(MODEL_DIR / "preprocessing.pkl", "rb") as f:
    preproc = pickle.load(f)

SCALER = preproc["scaler"]
STATE_ENCODING = preproc["state_encoding"]
NZV_COLS = preproc["nzv_cols"]
DROP_CORR = preproc["drop_corr"]
FEATURE_NAMES = preproc["feature_names"]

MODELS = {}
for tag, label in [("lr_cal", "Logistic Regression"),
                   ("xgb_cal", "XGBoost")]:
    with open(MODEL_DIR / f"{tag}.pkl", "rb") as f:
        MODELS[label] = pickle.load(f)

# ---------------------------------------------------------------------------
# Constants for UI controls
# ---------------------------------------------------------------------------
US_STATES = sorted(STATE_ENCODING.keys())

PURPOSE_OPTIONS = {
    "debt_consolidation": "Debt Consolidation",
    "credit_card": "Credit Card",
    "home_improvement": "Home Improvement",
    "major_purchase": "Major Purchase",
    "medical": "Medical",
    "small_business": "Small Business",
    "moving": "Moving",
    "vacation": "Vacation",
    "other": "Other",
}

HOME_OPTIONS = {"MORTGAGE": "Mortgage", "RENT": "Rent", "OWN": "Own"}

VERIFICATION_OPTIONS = {
    "Not Verified": "Not Verified",
    "Source Verified": "Source Verified",
    "Verified": "Verified",
}

SUB_GRADE_LABELS = {}
for i, letter in enumerate("ABCDEFG"):
    for num in range(1, 6):
        val = i * 5 + num
        SUB_GRADE_LABELS[str(val)] = f"{letter}{num}"

# ---------------------------------------------------------------------------
# Preprocessing helper
# ---------------------------------------------------------------------------

def build_feature_vector(
    loan_amnt, term, sub_grade, emp_length, annual_inc, addr_state,
    dti, fico_avg, revol_bal, revol_util, credit_age_months,
    home_ownership, purpose, verification_status,
):
    """Build a scaled feature vector matching FEATURE_NAMES order."""
    row = dict(MEDIANS)

    # --- User-tunable features ---
    row["loan_amnt"] = loan_amnt
    row["term"] = term
    row["sub_grade"] = sub_grade
    row["emp_length"] = emp_length
    row["annual_inc"] = annual_inc
    row["dti"] = dti
    row["fico_avg"] = fico_avg
    row["revol_bal"] = revol_bal
    row["revol_util"] = revol_util
    row["credit_age_months"] = credit_age_months

    # Target-encode addr_state
    global_mean = np.mean(list(STATE_ENCODING.values()))
    row["addr_state"] = STATE_ENCODING.get(addr_state, global_mean)

    # Derived features
    row["loan_to_income"] = loan_amnt / max(annual_inc, 1)
    row["revol_bal_to_income"] = revol_bal / max(annual_inc, 1)
    row["open_acc_ratio"] = (
        row["open_acc"] / max(row["total_acc"], 1)
    )
    row["log_annual_inc"] = np.log1p(annual_inc)
    row["log_revol_bal"] = np.log1p(revol_bal)
    row["log_tot_cur_bal"] = np.log1p(row.get("avg_cur_bal", 0) * row.get("open_acc", 1))

    # One-hot: home_ownership
    row["home_ownership_OWN"] = int(home_ownership == "OWN")
    row["home_ownership_RENT"] = int(home_ownership == "RENT")

    # One-hot: purpose
    for key in PURPOSE_OPTIONS:
        col = f"purpose_{key}"
        if col in row:
            row[col] = int(purpose == key)

    # One-hot: verification_status
    row["verification_status_Source Verified"] = int(
        verification_status == "Source Verified"
    )
    row["verification_status_Verified"] = int(
        verification_status == "Verified"
    )

    # Assemble in FEATURE_NAMES order
    vec = np.array([[row[f] for f in FEATURE_NAMES]], dtype=np.float64)

    # Scale
    vec_scaled = SCALER.transform(vec)
    return vec_scaled


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

app_ui = ui.page_fillable(
    ui.tags.style("""
        :root { --primary-color: #2563eb; }
        body { background: #f1f5f9; font-family: 'Segoe UI', system-ui, sans-serif; }
        .app-title {
            text-align: center; padding: 1.2rem 0 0.6rem;
            font-size: 1.6rem; font-weight: 700; color: #1e293b;
        }
        .app-subtitle {
            text-align: center; color: #64748b; margin-bottom: 1.2rem;
            font-size: 0.95rem;
        }
        .box-card {
            background: #fff; border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            padding: 1.5rem; height: 100%;
        }
        .box-title {
            font-size: 1.1rem; font-weight: 600; color: #1e293b;
            margin-bottom: 1rem; padding-bottom: 0.5rem;
            border-bottom: 2px solid #e2e8f0;
        }
        .result-badge {
            display: inline-block; padding: 0.5rem 1.5rem;
            border-radius: 999px; font-weight: 700; font-size: 1.1rem;
            letter-spacing: 0.02em;
        }
        .badge-approved { background: #dcfce7; color: #166534; }
        .badge-denied   { background: #fee2e2; color: #991b1b; }
        .gauge-container { text-align: center; margin: 1rem 0; }
        .gauge-value {
            font-size: 3rem; font-weight: 800; line-height: 1;
        }
        .gauge-label { color: #64748b; font-size: 0.85rem; margin-top: 0.25rem; }
        .prob-bar-wrap { margin: 0.6rem 0; }
        .prob-bar-label {
            display: flex; justify-content: space-between;
            font-size: 0.82rem; color: #475569; margin-bottom: 2px;
        }
        .prob-bar-bg {
            height: 22px; background: #e2e8f0; border-radius: 6px;
            overflow: hidden;
        }
        .prob-bar-fill {
            height: 100%; border-radius: 6px;
            transition: width 0.4s ease;
        }
        .model-select-row {
            display: flex; gap: 0.5rem; flex-wrap: wrap;
            margin-bottom: 1rem;
        }
    """),

    ui.div(
        ui.div("LendingClub Loan Default Predictor", class_="app-title"),
        ui.div(
            "Adjust borrower parameters and see real-time default probability "
            "from 2 calibrated models (Logistic Regression, XGBoost).",
            class_="app-subtitle",
        ),
    ),

    ui.layout_columns(
        # ---- LEFT BOX: Parameters ----
        ui.div(
            ui.div("Borrower Parameters", class_="box-title"),

            ui.layout_columns(
                ui.input_numeric("loan_amnt", "Loan Amount ($)", value=12000,
                                 min=500, max=40000, step=500),
                ui.input_select("term", "Term",
                                choices={"36": "36 months", "60": "60 months"},
                                selected="36"),
                col_widths=(8, 4),
            ),

            ui.layout_columns(
                ui.input_numeric("annual_inc", "Annual Income ($)", value=65000,
                                 min=0, max=500000, step=1000),
                ui.input_numeric("emp_length", "Employment (yrs)", value=6,
                                 min=0, max=10, step=1),
                col_widths=(8, 4),
            ),

            ui.layout_columns(
                ui.input_slider("fico_avg", "FICO Score", min=620, max=850,
                                value=692, step=5),
                col_widths=(12,),
            ),

            ui.layout_columns(
                ui.input_numeric("dti", "Debt-to-Income (%)", value=17.6,
                                 min=0, max=60, step=0.5),
                ui.input_numeric("revol_bal", "Revolving Balance ($)",
                                 value=11134, min=0, max=200000, step=500),
                col_widths=(6, 6),
            ),

            ui.layout_columns(
                ui.input_numeric("revol_util", "Revolving Utilization (%)",
                                 value=52.2, min=0, max=150, step=1),
                ui.input_numeric("credit_age", "Credit Age (months)",
                                 value=177, min=12, max=600, step=6),
                col_widths=(6, 6),
            ),

            ui.layout_columns(
                ui.input_select("purpose", "Purpose", choices=PURPOSE_OPTIONS,
                                selected="debt_consolidation"),
                ui.input_select("home", "Home Ownership",
                                choices=HOME_OPTIONS, selected="MORTGAGE"),
                col_widths=(6, 6),
            ),

            ui.layout_columns(
                ui.input_select("state", "State",
                                choices={s: s for s in US_STATES},
                                selected="CA"),
                ui.input_select("verification", "Income Verification",
                                choices=VERIFICATION_OPTIONS,
                                selected="Not Verified"),
                col_widths=(6, 6),
            ),

            ui.layout_columns(
                ui.input_select("sub_grade", "Sub-Grade",
                                choices=SUB_GRADE_LABELS, selected="11"),
                col_widths=(6,),
            ),

            class_="box-card",
        ),

        # ---- RIGHT BOX: Results ----
        ui.div(
            ui.div("Prediction Results", class_="box-title"),

            ui.input_select(
                "model_choice", "Model",
                choices={
                    "XGBoost": "XGBoost (recommended)",
                    "Logistic Regression": "Logistic Regression",
                },
                selected="XGBoost",
            ),

            ui.output_ui("result_panel"),

            class_="box-card",
        ),

        col_widths=(7, 5),
    ),

    padding="1.5rem",
)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

def server(input, output, session):

    @reactive.calc
    def predictions():
        vec = build_feature_vector(
            loan_amnt=input.loan_amnt(),
            term=float(input.term()),
            sub_grade=float(input.sub_grade()),
            emp_length=input.emp_length(),
            annual_inc=input.annual_inc(),
            addr_state=input.state(),
            dti=input.dti(),
            fico_avg=input.fico_avg(),
            revol_bal=input.revol_bal(),
            revol_util=input.revol_util(),
            credit_age_months=input.credit_age(),
            home_ownership=input.home(),
            purpose=input.purpose(),
            verification_status=input.verification(),
        )
        results = {}
        for name, model in MODELS.items():
            prob = float(model.predict_proba(vec)[0, 1])
            thresh = THRESHOLDS.get(name, 0.5)
            results[name] = {"prob": prob, "threshold": thresh,
                             "default": prob >= thresh}
        return results

    @render.ui
    def result_panel():
        preds = predictions()
        chosen = input.model_choice()
        p = preds[chosen]
        prob_pct = p["prob"] * 100

        # Color gradient: green (0%) -> yellow (25%) -> red (75%+)
        if prob_pct < 25:
            color = "#16a34a"
        elif prob_pct < 50:
            color = "#ca8a04"
        else:
            color = "#dc2626"

        badge_cls = "badge-denied" if p["default"] else "badge-approved"
        badge_txt = "HIGH RISK - LIKELY DEFAULT" if p["default"] else "LOW RISK - LIKELY REPAID"

        # Build bars for all models
        bars_html = ""
        for name in ["XGBoost", "Logistic Regression"]:
            mp = preds[name]
            mp_pct = mp["prob"] * 100
            bar_color = "#dc2626" if mp["default"] else "#16a34a"
            is_selected = " font-weight:700;" if name == chosen else ""
            bars_html += f"""
            <div class="prob-bar-wrap">
              <div class="prob-bar-label">
                <span style="{is_selected}">{name}</span>
                <span style="{is_selected}">{mp_pct:.1f}%</span>
              </div>
              <div class="prob-bar-bg">
                <div class="prob-bar-fill"
                     style="width:{min(mp_pct, 100):.1f}%;background:{bar_color};"></div>
              </div>
            </div>"""

        threshold_pct = p["threshold"] * 100

        if p["default"]:
            explain = (
                f"The predicted probability ({prob_pct:.1f}%) exceeds the "
                f"decision threshold ({threshold_pct:.1f}%). "
                "This threshold is intentionally low because missing a default "
                "is far costlier than a false alarm — our model is tuned to "
                "prioritize catching risky loans (F2 score optimization)."
            )
        else:
            explain = (
                f"The predicted probability ({prob_pct:.1f}%) is below the "
                f"decision threshold ({threshold_pct:.1f}%), indicating "
                "acceptable risk for this borrower profile."
            )

        return ui.HTML(f"""
            <div class="gauge-container">
              <div class="gauge-value" style="color:{color};">{prob_pct:.1f}%</div>
              <div class="gauge-label">Predicted Default Probability</div>
            </div>

            <div style="text-align:center; margin:1rem 0;">
              <span class="result-badge {badge_cls}">{badge_txt}</span>
            </div>

            <div style="text-align:center; color:#64748b; font-size:0.82rem; margin-bottom:0.4rem;">
              Decision threshold: {threshold_pct:.1f}%
            </div>

            <div style="background:#f8fafc; border-radius:8px; padding:0.7rem 1rem;
                        margin:0 0 1.2rem; color:#475569; font-size:0.82rem; line-height:1.5;">
              {explain}
            </div>

            <div style="border-top:1px solid #e2e8f0; padding-top:1rem; margin-top:0.5rem;">
              <div style="font-weight:600; font-size:0.95rem; color:#1e293b; margin-bottom:0.5rem;">
                All Models
              </div>
              {bars_html}
            </div>
        """)


app = App(app_ui, server)
