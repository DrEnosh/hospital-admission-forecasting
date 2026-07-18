"""
Stage 8 -- Interactive HTML Decision-Support Dashboard (Enhanced)
Dataset Type: Synthetic
Date-driven forecasting with real-time interactivity.
Pre-computes forecasts for Jul 2026 - Jun 2027 (12 months) and embeds them.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import pickle
import json
import warnings
warnings.filterwarnings("ignore")

from src.config import (DATA_CLEAN_DIR, MODELS_DIR, FORECASTS_DIR, DASHBOARD_DIR,
                         LOCATION_ORDER, COLORS, WATERMARK)


def generate_extended_forecasts(monthly, n_months=12):
    """Generate 12 months of forecasts (Jul 2026 - Jun 2027) using all available models."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    future_dates = pd.date_range("2026-07-01", periods=n_months, freq="MS")
    all_forecasts = {}

    for loc in LOCATION_ORDER:
        print(f"  Forecasting {loc} ...")
        d = monthly[monthly["Location"] == loc].sort_values("Month").set_index("Month")
        d.index.freq = "MS"
        loc_fc = {"months": [fd.strftime("%Y-%m") for fd in future_dates]}

        # SARIMA
        try:
            with open(os.path.join(MODELS_DIR, f"sarima_{loc}.pkl"), "rb") as f:
                sarima_model = pickle.load(f)
            refit = SARIMAX(d["Admissions"],
                            order=sarima_model.specification["order"],
                            seasonal_order=sarima_model.specification["seasonal_order"],
                            enforce_stationarity=False, enforce_invertibility=False)
            result = refit.fit(disp=False, maxiter=200)
            fc = result.get_forecast(steps=n_months)
            loc_fc["SARIMA"] = [max(0, round(v)) for v in fc.predicted_mean.values]
            ci = fc.conf_int()
            loc_fc["SARIMA_lower"] = [max(0, round(v)) for v in ci.iloc[:, 0].values]
            loc_fc["SARIMA_upper"] = [max(0, round(v)) for v in ci.iloc[:, 1].values]
        except Exception as e:
            print(f"    SARIMA error: {e}")
            loc_fc["SARIMA"] = [0] * n_months

        # ETS
        try:
            with open(os.path.join(MODELS_DIR, f"ets_{loc}.pkl"), "rb") as f:
                ets_model = pickle.load(f)
            refit = ExponentialSmoothing(
                d["Admissions"],
                trend=ets_model.model.trend if hasattr(ets_model.model, 'trend') else "add",
                seasonal=ets_model.model.seasonal if hasattr(ets_model.model, 'seasonal') else "add",
                seasonal_periods=12,
                damped_trend=ets_model.model.damped_trend if hasattr(ets_model.model, 'damped_trend') else False,
                initialization_method="estimated"
            )
            result = refit.fit(optimized=True, use_brute=False)
            pred = result.forecast(n_months)
            loc_fc["ETS"] = [max(0, round(v)) for v in pred.values]
        except Exception as e:
            print(f"    ETS error: {e}")
            loc_fc["ETS"] = [0] * n_months

        # Prophet
        try:
            with open(os.path.join(MODELS_DIR, f"prophet_{loc}.pkl"), "rb") as f:
                prophet_model = pickle.load(f)
            future = pd.DataFrame({"ds": future_dates})
            forecast = prophet_model.predict(future)
            loc_fc["Prophet"] = [max(0, round(v)) for v in forecast["yhat"].values]
            loc_fc["Prophet_lower"] = [max(0, round(v)) for v in forecast["yhat_lower"].values]
            loc_fc["Prophet_upper"] = [max(0, round(v)) for v in forecast["yhat_upper"].values]
        except Exception as e:
            print(f"    Prophet error: {e}")
            loc_fc["Prophet"] = [0] * n_months

        # ML models (iterative forecasting)
        for ml_name, ml_key in [("XGBoost", "xgboost"), ("RandomForest", "rf")]:
            try:
                with open(os.path.join(MODELS_DIR, f"{ml_key}_{loc}.pkl"), "rb") as f:
                    model = pickle.load(f)
                d_flat = monthly[monthly["Location"] == loc].sort_values("Month").reset_index(drop=True)
                hist = d_flat["Admissions"].values.tolist()
                preds = []
                for i, fd in enumerate(future_dates):
                    features = {
                        "Year": fd.year, "MonthNum": fd.month,
                        "Quarter": (fd.month - 1) // 3 + 1,
                        "Month_sin": np.sin(2 * np.pi * fd.month / 12),
                        "Month_cos": np.cos(2 * np.pi * fd.month / 12),
                        "Quarter_sin": np.sin(2 * np.pi * ((fd.month - 1) // 3 + 1) / 4),
                        "Quarter_cos": np.cos(2 * np.pi * ((fd.month - 1) // 3 + 1) / 4),
                        "TimeIndex": len(d_flat) + i,
                        "Lag_1": hist[-1], "Lag_2": hist[-2] if len(hist) >= 2 else hist[-1],
                        "Lag_3": hist[-3] if len(hist) >= 3 else hist[-1],
                        "Lag_6": hist[-6] if len(hist) >= 6 else hist[-1],
                        "Lag_12": hist[-12] if len(hist) >= 12 else hist[-1],
                        "Roll_3": np.mean(hist[-3:]),
                        "Roll_6": np.mean(hist[-6:]) if len(hist) >= 6 else np.mean(hist[-3:]),
                        "Roll_3_std": np.std(hist[-3:]),
                    }
                    X = pd.DataFrame([features])
                    pred = max(0, model.predict(X)[0])
                    preds.append(round(pred))
                    hist.append(pred)
                loc_fc[ml_name] = preds
            except Exception as e:
                print(f"    {ml_name} error: {e}")
                loc_fc[ml_name] = [0] * n_months

        all_forecasts[loc] = loc_fc

    return all_forecasts


def main():
    print("=" * 60)
    print(f"STAGE 8 -- INTERACTIVE DASHBOARD (Enhanced)  |  {WATERMARK}")
    print("=" * 60)

    monthly = pd.read_csv(os.path.join(DATA_CLEAN_DIR, "monthly_admissions.csv"), parse_dates=["Month"])
    daily = pd.read_csv(os.path.join(DATA_CLEAN_DIR, "daily_admissions.csv"), parse_dates=["Admission Date"])
    dept = pd.read_csv(os.path.join(DATA_CLEAN_DIR, "department_monthly.csv"), parse_dates=["Month"])

    # Extended forecasts
    print("\nGenerating 12-month extended forecasts ...")
    ext_forecasts = generate_extended_forecasts(monthly, n_months=12)

    # Monthly data
    monthly_data = {}
    for loc in LOCATION_ORDER:
        d = monthly[monthly["Location"] == loc].sort_values("Month")
        monthly_data[loc] = {
            "months": d["Month"].dt.strftime("%Y-%m").tolist(),
            "admissions": d["Admissions"].tolist()
        }

    # Best models
    best_map = {}
    best_path = os.path.join(FORECASTS_DIR, "best_models.csv")
    if os.path.exists(best_path):
        bdf = pd.read_csv(best_path)
        best_map = dict(zip(bdf["Location"], bdf["BestModel"]))

    # Metrics
    metrics_data = {}
    metrics_path = os.path.join(FORECASTS_DIR, "model_metrics.csv")
    if os.path.exists(metrics_path):
        met = pd.read_csv(metrics_path)
        for loc in LOCATION_ORDER:
            loc_met = met[met["Location"] == loc].to_dict("records")
            metrics_data[loc] = [{k: round(v, 2) if isinstance(v, float) else v for k, v in r.items()} for r in loc_met]

    # Department data
    dept_data = {}
    for loc in LOCATION_ORDER:
        d = dept[dept["Location"] == loc]
        totals = d.groupby("Admitting Department")["Admissions"].sum().sort_values(ascending=False).head(10)
        dept_data[loc] = {"departments": totals.index.tolist(), "values": totals.values.tolist()}

    # Stats
    stats_data = {}
    for loc in LOCATION_ORDER:
        d = daily[daily["Location"] == loc]
        m = monthly[monthly["Location"] == loc]["Admissions"]
        stats_data[loc] = {
            "total": int(d["Admissions"].sum()),
            "avg_daily": round(d["Admissions"].mean(), 1),
            "avg_monthly": round(m.mean(), 1),
            "months": int(len(m)),
            "date_start": str(d["Admission Date"].min().date()),
            "date_end": str(d["Admission Date"].max().date()),
        }

    # Days in each month (for daily rate)
    import calendar
    days_map = {}
    for y in range(2026, 2028):
        for m in range(1, 13):
            days_map[f"{y}-{m:02d}"] = calendar.monthrange(y, m)[1]

    # DOW weights
    dow_df = pd.read_csv(os.path.join(DATA_CLEAN_DIR, "dow_profile.csv"))
    dow_weights = {}
    for loc in LOCATION_ORDER:
        loc_dow = dow_df[dow_df["Location"] == loc]
        total = loc_dow["Admissions"].sum()
        weights = {}
        for _, row in loc_dow.iterrows():
            weights[int(row["DayNum"])] = round((row["Admissions"] / total) * 7.0, 4)
        dow_weights[loc] = weights

    html = _build_html(
        locations_js=json.dumps(LOCATION_ORDER),
        colors_js=json.dumps(COLORS),
        monthly_js=json.dumps(monthly_data),
        forecasts_js=json.dumps(ext_forecasts),
        best_map_js=json.dumps(best_map),
        metrics_js=json.dumps(metrics_data),
        dept_js=json.dumps(dept_data),
        stats_js=json.dumps(stats_data),
        days_js=json.dumps(days_map),
        dow_weights_js=json.dumps(dow_weights),
    )

    output_path = os.path.join(DASHBOARD_DIR, "dashboard.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n[OK] Stage 8 complete -- dashboard saved to {output_path}")


def _build_html(*, locations_js, colors_js, monthly_js, forecasts_js,
                best_map_js, metrics_js, dept_js, stats_js, days_js, dow_weights_js):
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Hospital Admission Forecasting System</title>
<meta name="description" content="Multi-Hospital Admission Forecasting Decision Support System">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
:root{{
  --white:#FFFFFF;--bg:#F4F7FA;--card:#FFFFFF;
  --border:#E2E8F0;--border-hover:#CBD5E1;
  --text:#1E293B;--text-sec:#475569;--text-muted:#94A3B8;
  --primary:#0F766E;--primary-light:#14B8A6;--primary-bg:rgba(20,184,166,0.06);
  --accent:#0284C7;--accent-bg:rgba(2,132,199,0.06);
  --success:#059669;--success-bg:rgba(5,150,105,0.08);
  --warn:#D97706;--warn-bg:rgba(217,119,6,0.06);
  --danger:#DC2626;--danger-bg:rgba(220,38,38,0.06);
  --radius:12px;--radius-lg:16px;
  --shadow-sm:0 1px 3px rgba(0,0,0,0.06);
  --shadow:0 4px 20px rgba(0,0,0,0.06);
  --shadow-lg:0 12px 40px rgba(0,0,0,0.1);
}}
body{{font-family:'Plus Jakarta Sans',system-ui,sans-serif;background:var(--bg);color:var(--text);min-height:100vh}}

/* ════════════════ LOGIN ════════════════ */
.login-overlay{{
  position:fixed;inset:0;z-index:9999;
  background:linear-gradient(135deg,#F0FDFA 0%,#ECFDF5 30%,#EFF6FF 70%,#F0F9FF 100%);
  display:flex;align-items:center;justify-content:center;
}}
.login-card{{
  background:var(--white);border:1px solid var(--border);
  border-radius:var(--radius-lg);padding:3rem;width:100%;max-width:420px;
  box-shadow:var(--shadow-lg);text-align:center;
  animation:loginFade .5s ease;
}}
@keyframes loginFade{{from{{opacity:0;transform:translateY(20px)}}to{{opacity:1;transform:translateY(0)}}}}
.login-logo{{
  width:64px;height:64px;border-radius:14px;margin:0 auto 1.5rem;
  background:linear-gradient(135deg,var(--primary),var(--accent));
  display:flex;align-items:center;justify-content:center;
  box-shadow:0 8px 24px rgba(15,118,110,0.25);
}}
.login-logo svg{{width:32px;height:32px;color:#fff}}
.login-card h1{{font-family:'Outfit',sans-serif;font-size:1.5rem;font-weight:700;color:var(--text);margin-bottom:.3rem}}
.login-card p{{color:var(--text-muted);font-size:.85rem;margin-bottom:2rem}}
.form-group{{text-align:left;margin-bottom:1.2rem}}
.form-group label{{display:block;font-size:.72rem;font-weight:700;color:var(--text-sec);text-transform:uppercase;letter-spacing:.08em;margin-bottom:.4rem}}
.form-group input{{
  width:100%;padding:.75rem 1rem;border:1.5px solid var(--border);border-radius:10px;
  font-family:inherit;font-size:.9rem;color:var(--text);background:var(--bg);outline:none;
  transition:all .2s;
}}
.form-group input:focus{{border-color:var(--primary);box-shadow:0 0 0 3px rgba(20,184,166,0.12)}}
.login-error{{color:var(--danger);font-size:.8rem;margin-bottom:1rem;display:none;font-weight:600}}
.btn-login{{
  width:100%;padding:.85rem;border:none;border-radius:10px;
  background:linear-gradient(135deg,var(--primary),var(--primary-light));
  color:#fff;font-family:inherit;font-size:.95rem;font-weight:700;cursor:pointer;
  transition:all .2s;box-shadow:0 4px 16px rgba(15,118,110,0.3);
}}
.btn-login:hover{{transform:translateY(-1px);box-shadow:0 6px 20px rgba(15,118,110,0.4)}}
.login-footer{{margin-top:1.5rem;color:var(--text-muted);font-size:.7rem}}

/* ════════════════ APP SHELL ════════════════ */
.app{{display:none}}
.topbar{{
  background:var(--white);border-bottom:1px solid var(--border);
  padding:0 2.5rem;height:64px;display:flex;align-items:center;justify-content:space-between;
  position:sticky;top:0;z-index:100;box-shadow:var(--shadow-sm);
}}
.topbar-brand{{display:flex;align-items:center;gap:.8rem}}
.topbar-brand .logo-sm{{
  width:36px;height:36px;border-radius:8px;
  background:linear-gradient(135deg,var(--primary),var(--accent));
  display:flex;align-items:center;justify-content:center;
}}
.topbar-brand .logo-sm svg{{width:18px;height:18px;color:#fff}}
.topbar-brand h2{{font-family:'Outfit',sans-serif;font-size:1.1rem;font-weight:700;color:var(--text)}}
.topbar-right{{display:flex;align-items:center;gap:1.5rem}}
.topbar-right .user-pill{{
  display:flex;align-items:center;gap:.5rem;
  background:var(--bg);border:1px solid var(--border);border-radius:30px;padding:.35rem 1rem .35rem .35rem;
  font-size:.8rem;font-weight:600;color:var(--text-sec);
}}
.topbar-right .user-pill .avatar{{
  width:28px;height:28px;border-radius:50%;
  background:var(--primary);color:#fff;display:flex;align-items:center;justify-content:center;
  font-size:.7rem;font-weight:700;
}}
.topbar-right .badge-synth{{
  background:var(--warn-bg);color:var(--warn);padding:.3rem .8rem;border-radius:20px;
  font-size:.68rem;font-weight:700;text-transform:uppercase;letter-spacing:.05em;
}}
.btn-logout{{
  background:none;border:1px solid var(--border);border-radius:8px;padding:.4rem .8rem;
  font-family:inherit;font-size:.75rem;font-weight:600;color:var(--text-muted);cursor:pointer;
  transition:all .2s;
}}
.btn-logout:hover{{background:var(--danger-bg);color:var(--danger);border-color:var(--danger)}}

/* Container */
.container{{max-width:1280px;margin:0 auto;padding:2rem 2.5rem}}

/* Category Tabs */
.category-bar{{
  display:flex;gap:.5rem;margin-bottom:2rem;
  background:var(--white);border:1px solid var(--border);padding:.4rem;border-radius:var(--radius);
  box-shadow:var(--shadow-sm);max-width:fit-content;
}}
.cat-btn{{
  padding:.7rem 2rem;border:none;border-radius:9px;background:transparent;
  font-family:inherit;font-size:.85rem;font-weight:600;color:var(--text-muted);cursor:pointer;
  transition:all .2s;display:flex;align-items:center;gap:.5rem;
}}
.cat-btn:hover{{color:var(--text);background:var(--bg)}}
.cat-btn.active{{background:var(--primary);color:#fff;box-shadow:0 2px 8px rgba(15,118,110,0.25)}}
.cat-btn .cat-icon{{font-size:1rem}}

/* Controls Row */
.controls{{
  display:flex;gap:1.5rem;align-items:flex-end;flex-wrap:wrap;margin-bottom:2rem;
  background:var(--white);border:1px solid var(--border);border-radius:var(--radius);
  padding:1.5rem 2rem;box-shadow:var(--shadow-sm);
}}
.ctrl-group{{display:flex;flex-direction:column;gap:.4rem;flex:1;min-width:160px}}
.ctrl-group label{{
  font-size:.68rem;font-weight:700;color:var(--text-muted);text-transform:uppercase;letter-spacing:.08em;
  display:flex;align-items:center;gap:.35rem;
}}
.ctrl-group label .lbl-icon{{font-size:.82rem;opacity:.7}}
.ctrl-group select{{
  padding:.7rem 2.2rem .7rem 1rem;border:1.5px solid var(--border);border-radius:9px;
  font-family:inherit;font-size:.85rem;color:var(--text);background:var(--bg);outline:none;
  transition:all .25s ease;cursor:pointer;
  -webkit-appearance:none;-moz-appearance:none;appearance:none;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%2394A3B8' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='6 9 12 15 18 9'/%3E%3C/svg%3E");
  background-repeat:no-repeat;background-position:right .8rem center;
}}
.ctrl-group select:hover{{border-color:var(--border-hover);background-color:#f8fafc}}
.ctrl-group select:focus{{border-color:var(--primary);box-shadow:0 0 0 3px rgba(20,184,166,0.1);background-color:var(--white)}}
.date-row{{display:flex;gap:.5rem}}
.date-row .ctrl-sub{{display:flex;flex-direction:column;gap:.4rem;flex:1;min-width:0}}
.date-row .ctrl-sub label{{font-size:.62rem;font-weight:600;color:var(--text-muted);text-transform:uppercase;letter-spacing:.06em}}
.date-row .ctrl-sub select{{
  padding:.7rem 1.8rem .7rem .7rem;border:1.5px solid var(--border);border-radius:9px;
  font-family:inherit;font-size:.82rem;color:var(--text);background:var(--bg);outline:none;
  transition:all .25s ease;cursor:pointer;
  -webkit-appearance:none;-moz-appearance:none;appearance:none;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='10' viewBox='0 0 24 24' fill='none' stroke='%2394A3B8' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='6 9 12 15 18 9'/%3E%3C/svg%3E");
  background-repeat:no-repeat;background-position:right .55rem center;
}}
.date-row .ctrl-sub select:hover{{border-color:var(--border-hover)}}
.date-row .ctrl-sub select:focus{{border-color:var(--primary);box-shadow:0 0 0 3px rgba(20,184,166,0.1)}}
.btn-predict{{
  padding:.72rem 2rem;border:none;border-radius:9px;
  background:linear-gradient(135deg,var(--primary),var(--primary-light));color:#fff;
  font-family:inherit;font-size:.85rem;font-weight:700;
  cursor:pointer;transition:all .25s ease;box-shadow:0 3px 12px rgba(15,118,110,0.2);
  display:flex;align-items:center;gap:.5rem;white-space:nowrap;
}}
.btn-predict:hover{{transform:translateY(-1px);box-shadow:0 6px 20px rgba(15,118,110,0.35)}}

/* Results */
.results{{animation:resultsIn .3s ease}}
@keyframes resultsIn{{from{{opacity:0;transform:translateY(8px)}}to{{opacity:1;transform:translateY(0)}}}}

/* Summary KPI strip */
.kpi-strip{{
  display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1rem;margin-bottom:2rem;
}}
.kpi-card{{
  background:var(--white);border:1px solid var(--border);border-radius:var(--radius);
  padding:1.2rem 1.5rem;box-shadow:var(--shadow-sm);transition:all .2s;
}}
.kpi-card:hover{{box-shadow:var(--shadow);border-color:var(--border-hover)}}
.kpi-card .kpi-label{{font-size:.68rem;font-weight:700;color:var(--text-muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:.3rem}}
.kpi-card .kpi-value{{font-family:'Outfit',sans-serif;font-size:1.6rem;font-weight:700;color:var(--text)}}
.kpi-card .kpi-sub{{font-size:.75rem;color:var(--text-muted);margin-top:.2rem}}

/* Prediction Table */
.pred-table-wrap{{
  background:var(--white);border:1px solid var(--border);border-radius:var(--radius);
  box-shadow:var(--shadow-sm);overflow:hidden;
}}
.pred-table-header{{
  padding:1.2rem 1.8rem;border-bottom:1px solid var(--border);
  display:flex;justify-content:space-between;align-items:center;
}}
.pred-table-header h3{{font-family:'Outfit',sans-serif;font-size:1rem;font-weight:700;color:var(--text)}}
.pred-table-header .mode-badge{{
  padding:.3rem .8rem;border-radius:20px;font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.04em;
}}
.mode-monthly{{background:var(--primary-bg);color:var(--primary)}}
.mode-weekly{{background:var(--accent-bg);color:var(--accent)}}
.mode-daily{{background:var(--success-bg);color:var(--success)}}
table.pred-tbl{{width:100%;border-collapse:collapse;font-size:.85rem}}
table.pred-tbl th{{
  text-align:left;padding:.85rem 1.8rem;font-size:.7rem;font-weight:700;
  color:var(--text-muted);text-transform:uppercase;letter-spacing:.06em;
  background:var(--bg);border-bottom:1px solid var(--border);
}}
table.pred-tbl td{{
  padding:.9rem 1.8rem;border-bottom:1px solid var(--border);color:var(--text-sec);
  transition:all .15s;
}}
table.pred-tbl tr:last-child td{{border-bottom:none}}
table.pred-tbl tr:hover td{{background:rgba(20,184,166,0.02)}}
.loc-dot{{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:.6rem;vertical-align:middle}}
.accuracy-pill{{
  display:inline-block;padding:.2rem .6rem;border-radius:20px;font-size:.72rem;font-weight:700;
}}
.acc-high{{background:var(--success-bg);color:var(--success)}}
.acc-mid{{background:var(--warn-bg);color:var(--warn)}}
.acc-low{{background:var(--danger-bg);color:var(--danger)}}

/* Footer */
.app-footer{{
  text-align:center;padding:2.5rem;color:var(--text-muted);font-size:.72rem;
  border-top:1px solid var(--border);margin-top:3rem;
}}

@media(max-width:768px){{
  /* Topbar Mobile */
  .topbar{{padding:0 1rem;height:56px}}
  .topbar-brand h2{{font-size:.9rem}}
  .topbar-brand .logo-sm{{width:28px;height:28px}}
  .topbar-brand .logo-sm svg{{width:14px;height:14px}}
  .topbar-right{{gap:.8rem}}
  .topbar-right .user-pill{{display:none}} /* Hide admin badge to save space */
  .topbar-right .badge-synth{{font-size:.6rem;padding:.2rem .5rem}}
  .btn-logout{{padding:.3rem .6rem;font-size:.7rem}}

  /* Container */
  .container{{padding:1rem .8rem}}

  /* Login Card Mobile */
  .login-card{{padding:2rem 1.5rem;max-width:90%;margin:1rem}}
  .login-card h1{{font-size:1.3rem}}
  .login-card p{{font-size:.8rem;margin-bottom:1.5rem}}

  /* Category Bar Mobile */
  .category-bar{{
    display:flex;width:100%;max-width:100%;overflow-x:auto;
    scrollbar-width:none;-ms-overflow-style:none;
    gap:.25rem;border-radius:10px;padding:.25rem;
  }}
  .category-bar::-webkit-scrollbar{{display:none}}
  .cat-btn{{
    flex:1 0 auto;text-align:center;justify-content:center;
    padding:.6rem 1rem;font-size:.78rem;white-space:nowrap;
  }}

  /* Controls Mobile */
  .controls{{padding:1.2rem 1rem;gap:1.2rem}}
  .ctrl-group{{flex:1 1 100%;width:100%;min-width:100%}}
  .date-row{{width:100%}}
  .btn-predict{{width:100%;justify-content:center;padding:.8rem}}

  /* KPIs Mobile */
  .kpi-strip{{grid-template-columns:1fr 1fr;gap:.8rem}}
  .kpi-card{{padding:1rem}}
  .kpi-card .kpi-value{{font-size:1.3rem}}

  /* Table Mobile Card/Flow */
  .pred-table-wrap{{
    overflow-x:hidden;
  }}
  table.pred-tbl{{
    min-width: 100% !important;
  }}
  /* Hide all columns except the 1st (Hospital), 2nd (Predicted Admissions), and last (Accuracy) */
  table.pred-tbl th:not(:nth-child(1)):not(:nth-child(2)):not(:last-child),
  table.pred-tbl td:not(:nth-child(1)):not(:nth-child(2)):not(:last-child) {{
    display:none !important;
  }}
  /* Ensure the remaining columns stretch nicely */
  table.pred-tbl th:nth-child(1), table.pred-tbl td:nth-child(1) {{
    width: 45%;
  }}
  table.pred-tbl th:nth-child(2), table.pred-tbl td:nth-child(2) {{
    width: 30%;
  }}
  table.pred-tbl th:last-child, table.pred-tbl td:last-child {{
    width: 25%;
    text-align: right;
  }}
  .mobile-model-tag{{
    display:block;
    font-size:.65rem;
    color:var(--text-muted);
    margin-left:14px;
    margin-top:2px;
  }}
  .pred-table-header{{padding:1rem}}
  table.pred-tbl th, table.pred-tbl td{{padding:.8rem 1rem;font-size:.8rem}}
}}

@media(min-width:769px){{
  .mobile-model-tag{{display:none}}
}}

@media(max-width:480px){{
  .kpi-strip{{grid-template-columns:1fr}} /* Stack KPIs on very small screens */
}}
</style>
</head>
<body>

<!-- ════════════════ LOGIN SCREEN ════════════════ -->
<div class="login-overlay" id="loginOverlay">
  <div class="login-card">
    <div class="login-logo">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
      </svg>
    </div>
    <h1>Admission Forecasting</h1>
    <p>Decision Support System &mdash; Authorized Access Only</p>
    <div class="login-error" id="loginError">Invalid credentials. Please try again.</div>
    <div class="form-group">
      <label>User ID</label>
      <input type="text" id="loginUser" placeholder="Enter your user ID" autocomplete="off">
    </div>
    <div class="form-group">
      <label>Password</label>
      <input type="password" id="loginPass" placeholder="Enter your password">
    </div>
    <button class="btn-login" onclick="doLogin()">Sign In</button>
    <div class="login-footer">Dataset Type: Synthetic &bull; Multi-Hospital Forecasting Study &copy; 2026</div>
  </div>
</div>

<!-- ════════════════ APP ════════════════ -->
<div class="app" id="appMain">

  <!-- Top Bar -->
  <header class="topbar">
    <div class="topbar-brand">
      <div class="logo-sm">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
        </svg>
      </div>
      <h2>Hospital Admission Forecasting</h2>
    </div>
    <div class="topbar-right">
      <span class="badge-synth">Synthetic Data</span>
      <div class="user-pill">
        <span class="avatar">A</span>
        admin
      </div>
      <button class="btn-logout" onclick="doLogout()">Sign Out</button>
    </div>
  </header>

  <div class="container">

    <!-- Category Tabs -->
    <div class="category-bar">
      <button class="cat-btn active" data-mode="monthly" onclick="setMode('monthly',this)">
        <span class="cat-icon">&#128197;</span> Monthly Forecast
      </button>
      <button class="cat-btn" data-mode="weekly" onclick="setMode('weekly',this)">
        <span class="cat-icon">&#128198;</span> Weekly Forecast
      </button>
      <button class="cat-btn" data-mode="daily" onclick="setMode('daily',this)">
        <span class="cat-icon">&#128336;</span> Daily Forecast
      </button>
    </div>

    <!-- Controls -->
    <div class="controls">
      <div class="ctrl-group" id="ctrlDate">
        <label><span class="lbl-icon">&#128197;</span> <span id="dateLabel">Select Period</span></label>
        <div class="date-row">
          <div class="ctrl-sub">
            <label>Month</label>
            <select id="selMonth">
              <option value="07">July</option><option value="08">August</option>
              <option value="09">September</option><option value="10">October</option>
              <option value="11">November</option><option value="12">December</option>
              <option value="01">January</option><option value="02">February</option>
              <option value="03">March</option><option value="04">April</option>
              <option value="05">May</option><option value="06">June</option>
            </select>
          </div>
          <div class="ctrl-sub">
            <label>Year</label>
            <select id="selYear">
              <option value="2026">2026</option><option value="2027">2027</option>
            </select>
          </div>
          <div class="ctrl-sub" id="dayGroup" style="display:none">
            <label>Day</label>
            <select id="selDay"></select>
          </div>
        </div>
      </div>
      <div class="ctrl-group">
        <label><span class="lbl-icon">&#127973;</span> Hospital</label>
        <select id="inputLocation">
          <option value="all">All Hospitals</option>
        </select>
      </div>
      <div class="ctrl-group">
        <label><span class="lbl-icon">&#9881;</span> Forecasting Model</label>
        <select id="inputModel">
          <option value="best">Best Model (Auto)</option>
          <option value="SARIMA">SARIMA</option>
          <option value="ETS">ETS (Holt-Winters)</option>
          <option value="Prophet">Prophet</option>
          <option value="XGBoost">XGBoost</option>
          <option value="RandomForest">Random Forest</option>
        </select>
      </div>
      <button class="btn-predict" onclick="predict()">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"/></svg>
        Generate Prediction
      </button>
    </div>

    <!-- Results Area -->
    <div id="resultsArea"></div>

  </div>

  <footer class="app-footer">
    <p>Dataset Type: Synthetic &mdash; All data is anonymized and generated for demonstration purposes.</p>
    <p>Multi-Hospital Admission Forecasting and Decision Support Study &copy; 2026</p>
  </footer>

</div>

<script>
const LOCATIONS = {locations_js};
const COLORS = {colors_js};
const monthlyData = {monthly_js};
const fcData = {forecasts_js};
const bestMap = {best_map_js};
const metricsData = {metrics_js};
const deptData = {dept_js};
const statsData = {stats_js};
const daysInMonth = {days_js};
const dowWeights = {dow_weights_js};

const modelNames = {{SARIMA:'SARIMA',ETS:'ETS',Prophet:'Prophet',XGBoost:'XGBoost',RandomForest:'Random Forest'}};
let currentMode = 'monthly';

// Populate selects
const locSel = document.getElementById('inputLocation');
LOCATIONS.forEach(l => {{
  locSel.innerHTML += `<option value="${{l}}">${{l.replace(/_/g,' ')}}</option>`;
}});

// Enter key support on login
document.getElementById('loginPass').addEventListener('keydown', e => {{ if(e.key==='Enter') doLogin(); }});
document.getElementById('loginUser').addEventListener('keydown', e => {{ if(e.key==='Enter') document.getElementById('loginPass').focus(); }});

/* ═══════ LOGIN ═══════ */
function doLogin() {{
  const u = document.getElementById('loginUser').value.trim();
  const p = document.getElementById('loginPass').value;
  if (u === 'admin' && p === '1234') {{
    document.getElementById('loginOverlay').style.display = 'none';
    document.getElementById('appMain').style.display = 'block';
  }} else {{
    document.getElementById('loginError').style.display = 'block';
    document.getElementById('loginPass').value = '';
  }}
}}
function doLogout() {{
  document.getElementById('appMain').style.display = 'none';
  document.getElementById('loginOverlay').style.display = 'flex';
  document.getElementById('loginUser').value = '';
  document.getElementById('loginPass').value = '';
  document.getElementById('loginError').style.display = 'none';
  document.getElementById('resultsArea').innerHTML = '';
}}

/* ═══════ DATE HELPERS ═══════ */
const monthNamesFull = ['','January','February','March','April','May','June','July','August','September','October','November','December'];
function populateDays() {{
  const m = parseInt(document.getElementById('selMonth').value);
  const y = parseInt(document.getElementById('selYear').value);
  const key = `${{y}}-${{String(m).padStart(2,'0')}}`;
  const maxDay = daysInMonth[key] || 30;
  const sel = document.getElementById('selDay');
  const cur = parseInt(sel.value) || 1;
  sel.innerHTML = '';
  for (let d = 1; d <= maxDay; d++) {{
    sel.innerHTML += `<option value="${{d}}" ${{d===Math.min(cur,maxDay)?'selected':''}}>${{d}}</option>`;
  }}
}}
function getDateValue() {{
  const m = document.getElementById('selMonth').value;
  const y = document.getElementById('selYear').value;
  if (currentMode === 'monthly') return `${{y}}-${{m}}`;
  const d = String(document.getElementById('selDay').value).padStart(2,'0');
  return `${{y}}-${{m}}-${{d}}`;
}}
// auto-populate days when month/year changes
document.getElementById('selMonth').addEventListener('change', () => {{ populateDays(); syncYearMonth(); }});
document.getElementById('selYear').addEventListener('change', () => {{ populateDays(); syncYearMonth(); }});
// ensure valid month-year combos (Jul-Dec 2026, Jan-Jun 2027)
function syncYearMonth() {{
  const m = parseInt(document.getElementById('selMonth').value);
  const y = parseInt(document.getElementById('selYear').value);
  if (y === 2026 && m < 7) document.getElementById('selYear').value = '2027';
  if (y === 2027 && m > 6) document.getElementById('selYear').value = '2026';
}}

/* ═══════ MODE SWITCH ═══════ */
function setMode(mode, btn) {{
  currentMode = mode;
  document.querySelectorAll('.cat-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  const dayGroup = document.getElementById('dayGroup');
  const dateLabel = document.getElementById('dateLabel');
  if (mode === 'monthly') {{
    dayGroup.style.display = 'none';
    dateLabel.textContent = 'Select Month';
  }} else if (mode === 'weekly') {{
    dayGroup.style.display = '';
    populateDays();
    dateLabel.textContent = 'Select Week Start';
  }} else {{
    dayGroup.style.display = '';
    populateDays();
    dateLabel.textContent = 'Select Date';
  }}
  document.getElementById('resultsArea').innerHTML = '';
}}

/* ═══════ HELPERS ═══════ */
function fmt(n) {{ return Math.round(n).toLocaleString(); }}
function getAccuracy(loc, model) {{
  const m = (metricsData[loc]||[]).find(r => r.Model === model);
  return m ? (100 - m.MAPE) : null;
}}
function accPill(acc) {{
  if (acc === null) return '<span class="accuracy-pill acc-mid">N/A</span>';
  const cls = acc >= 90 ? 'acc-high' : acc >= 80 ? 'acc-mid' : 'acc-low';
  return `<span class="accuracy-pill ${{cls}}">${{acc.toFixed(1)}}%</span>`;
}}
function getDOW(dateStr) {{
  const p = dateStr.split('-');
  const d = new Date(p[0], p[1]-1, p[2]);
  let dn = d.getDay()-1; if(dn===-1) dn=6;
  return dn;
}}
const dayNames = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'];

/* ═══════ PREDICT ═══════ */
function predict() {{
  const dateVal = getDateValue();
  const locVal = document.getElementById('inputLocation').value;
  const modelVal = document.getElementById('inputModel').value;
  if (!dateVal) return;

  const locs = locVal === 'all' ? LOCATIONS : [locVal];

  if (currentMode === 'monthly') predictMonthly(dateVal, locs, modelVal);
  else if (currentMode === 'weekly') predictWeekly(dateVal, locs, modelVal);
  else predictDaily(dateVal, locs, modelVal);
}}

/* ─── MONTHLY ─── */
function predictMonthly(monthVal, locs, modelVal) {{
  const days = daysInMonth[monthVal] || 30;
  let total = 0;
  const rows = [];
  locs.forEach(loc => {{
    const fc = fcData[loc]; if(!fc) return;
    const idx = fc.months.indexOf(monthVal); if(idx===-1) return;
    const model = modelVal==='best' ? (bestMap[loc]||'SARIMA') : modelVal;
    const val = (fc[model] && fc[model][idx]!==undefined) ? fc[model][idx] : 0;
    const acc = getAccuracy(loc, model);
    total += val;
    rows.push({{loc, val, daily: Math.round(val/days), model, acc}});
  }});
  renderResults('monthly', monthVal, rows, total, days, locs.length);
}}

/* ─── WEEKLY ─── */
function predictWeekly(startDate, locs, modelVal) {{
  const monthVal = startDate.substring(0,7);
  const days = daysInMonth[monthVal] || 30;
  let total = 0;
  const rows = [];
  // compute 7-day forecast
  locs.forEach(loc => {{
    const fc = fcData[loc]; if(!fc) return;
    const idx = fc.months.indexOf(monthVal); if(idx===-1) return;
    const model = modelVal==='best' ? (bestMap[loc]||'SARIMA') : modelVal;
    const monthlyVal = (fc[model] && fc[model][idx]!==undefined) ? fc[model][idx] : 0;
    const baseDaily = monthlyVal / days;
    let weekTotal = 0;
    const startDow = getDOW(startDate);
    for (let i = 0; i < 7; i++) {{
      const dow = (startDow + i) % 7;
      const w = (dowWeights[loc] && dowWeights[loc][dow]!==undefined) ? dowWeights[loc][dow] : 1.0;
      weekTotal += baseDaily * w;
    }}
    weekTotal = Math.round(weekTotal);
    const acc = getAccuracy(loc, model);
    total += weekTotal;
    rows.push({{loc, val: weekTotal, daily: Math.round(weekTotal/7), model, acc}});
  }});
  // compute end date string
  const p = startDate.split('-');
  const sd = new Date(p[0],p[1]-1,p[2]);
  const ed = new Date(sd); ed.setDate(ed.getDate()+6);
  const label = startDate + ' to ' + ed.toISOString().substring(0,10);
  renderResults('weekly', label, rows, total, 7, locs.length);
}}

/* ─── DAILY ─── */
function predictDaily(dateVal, locs, modelVal) {{
  const monthVal = dateVal.substring(0,7);
  const days = daysInMonth[monthVal] || 30;
  const dow = getDOW(dateVal);
  let total = 0;
  const rows = [];
  locs.forEach(loc => {{
    const fc = fcData[loc]; if(!fc) return;
    const idx = fc.months.indexOf(monthVal); if(idx===-1) return;
    const model = modelVal==='best' ? (bestMap[loc]||'SARIMA') : modelVal;
    const monthlyVal = (fc[model] && fc[model][idx]!==undefined) ? fc[model][idx] : 0;
    const w = (dowWeights[loc] && dowWeights[loc][dow]!==undefined) ? dowWeights[loc][dow] : 1.0;
    const val = Math.round((monthlyVal / days) * w);
    const acc = getAccuracy(loc, model);
    total += val;
    rows.push({{loc, val, daily: val, model, acc, dow: dayNames[dow], weight: w}});
  }});
  renderResults('daily', dateVal + ' (' + dayNames[dow] + ')', rows, total, 1, locs.length);
}}

/* ─── RENDER ─── */
function renderResults(mode, label, rows, total, periodDays, numLocs) {{
  const area = document.getElementById('resultsArea');
  const modeClass = mode === 'monthly' ? 'mode-monthly' : mode === 'weekly' ? 'mode-weekly' : 'mode-daily';
  const modeLabel = mode.charAt(0).toUpperCase() + mode.slice(1);

  let kpis = `
    <div class="kpi-strip">
      <div class="kpi-card">
        <div class="kpi-label">Total Predicted Admissions</div>
        <div class="kpi-value">${{fmt(total)}}</div>
        <div class="kpi-sub">${{modeLabel}} forecast</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Average per Hospital</div>
        <div class="kpi-value">${{fmt(Math.round(total/numLocs))}}</div>
        <div class="kpi-sub">Across ${{numLocs}} hospital${{numLocs>1?'s':''}}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Average per Day</div>
        <div class="kpi-value">${{fmt(Math.round(total/periodDays))}}</div>
        <div class="kpi-sub">${{periodDays}} day${{periodDays>1?'s':''}} in period</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Prediction Period</div>
        <div class="kpi-value" style="font-size:1rem;line-height:1.4">${{label}}</div>
        <div class="kpi-sub">${{modeLabel}} view</div>
      </div>
    </div>`;

  // Extra columns for daily mode
  const dowCol = mode === 'daily';
  let tableRows = '';
  rows.forEach(r => {{
    const modelTag = modelNames[r.model]||r.model;
    tableRows += `<tr>
      <td>
        <span class="loc-dot" style="background:${{COLORS[r.loc]}}"></span>
        ${{r.loc.replace(/_/g,' ')}}
        <span class="mobile-model-tag">${{modelTag}}</span>
      </td>
      <td style="font-weight:700;color:var(--text)">${{fmt(r.val)}}</td>
      ${{dowCol ? `<td>${{r.dow}}</td><td>${{r.weight.toFixed(2)}}x</td>` : `<td>${{fmt(r.daily)}}</td>`}}
      <td>${{modelTag}}</td>
      <td>${{accPill(r.acc)}}</td>
    </tr>`;
  }});

  let table = `
    <div class="pred-table-wrap">
      <div class="pred-table-header">
        <h3>Admission Predictions by Hospital</h3>
        <span class="mode-badge ${{modeClass}}">${{modeLabel}}</span>
      </div>
      <table class="pred-tbl">
        <thead><tr>
          <th>Hospital</th>
          <th>Predicted Admissions</th>
          ${{dowCol ? '<th>Day of Week</th><th>DOW Adj.</th>' : '<th>Avg / Day</th>'}}
          <th>Model Used</th>
          <th>Accuracy</th>
        </tr></thead>
        <tbody>${{tableRows}}</tbody>
      </table>
    </div>`;

  area.innerHTML = `<div class="results">${{kpis}}${{table}}</div>`;
}}
</script>
</body>
</html>'''


if __name__ == "__main__":
    main()

