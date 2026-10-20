"""
Supply Chain Demand Forecasting Dashboard
==========================================
Two-tab Streamlit dashboard:
  Tab 1 — Key EDA Findings
  Tab 2 — LightGBM Forecasting Model (interactive dropdowns)

Run:  streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings, os

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Supply Chain Demand Forecasting",
    page_icon="📦",
    layout="wide",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px; border-radius: 12px; color: white; text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,.1);
    }
    .metric-card h3 { margin: 0; font-size: 14px; opacity: .85; }
    .metric-card h1 { margin: 5px 0 0; font-size: 32px; }
    .insight-box {
        background: #f0f2f6; border-left: 4px solid #667eea;
        padding: 15px; border-radius: 0 8px 8px 0; margin: 10px 0;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        background-color: #667eea; color: white; border-radius: 8px 8px 0 0;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# DATA LOADING & FEATURE ENGINEERING  (cached)
# ═══════════════════════════════════════════════
@st.cache_data(show_spinner="Loading data & training model …")
def load_and_train():
    """Load data, engineer features, train LightGBM, return everything."""

    # ---------- load ----------
    # Resolve path relative to this script's location (works on Windows & Linux)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "supply_chain_demand_forecasting_v3.xlsx")
    if not os.path.exists(data_path):
        st.error(f"Data file not found at: {data_path}\n\nMake sure 'supply_chain_demand_forecasting_v3.xlsx' is in the same folder as dashboard.py.")
        st.stop()
    df = pd.read_excel(data_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Store Id", "Product Id", "Date"]).reset_index(drop=True)

    # ---------- feature engineering ----------
    df["quarter"]     = df["Date"].dt.quarter
    df["is_year_end"] = df["Month"].isin([11, 12]).astype(int)
    df["is_q1"]       = (df["quarter"] == 1).astype(int)
    df["month_sin"]   = np.sin(2 * np.pi * df["Month"] / 12)
    df["month_cos"]   = np.cos(2 * np.pi * df["Month"] / 12)
    df["week_sin"]    = np.sin(2 * np.pi * df["Week Of Year"] / 52)
    df["week_cos"]    = np.cos(2 * np.pi * df["Week Of Year"] / 52)

    grp = df.groupby(["Store Id", "Product Id"])
    df["lag_1w"]  = grp["Sales Quantity"].shift(1)
    df["lag_2w"]  = grp["Sales Quantity"].shift(2)
    df["lag_4w"]  = grp["Sales Quantity"].shift(4)
    df["lag_52w"] = grp["Sales Quantity"].shift(52)

    df["rolling_mean_4w"]  = grp["Sales Quantity"].transform(lambda x: x.shift(1).rolling(4).mean())
    df["rolling_mean_12w"] = grp["Sales Quantity"].transform(lambda x: x.shift(1).rolling(12).mean())
    df["rolling_std_4w"]   = grp["Sales Quantity"].transform(lambda x: x.shift(1).rolling(4).std())
    df["rolling_std_12w"]  = grp["Sales Quantity"].transform(lambda x: x.shift(1).rolling(12).std())
    df["rolling_max_4w"]   = grp["Sales Quantity"].transform(lambda x: x.shift(1).rolling(4).max())
    df["ewm_4w"]           = grp["Sales Quantity"].transform(lambda x: x.shift(1).ewm(span=4).mean())

    df["lag_promo_1w"]    = grp["Promotion Flag"].shift(1)
    df["promo_x_holiday"] = df["Promotion Flag"] * df["Holiday Flag"]

    cat_avg = df.groupby("Category")["Unit Price"].transform("mean")
    df["price_vs_avg"] = df["Unit Price"] / cat_avg

    df["inventory_coverage"] = df["Inventory Start"] / df["Sales Quantity"].replace(0, 1)
    df["lag_stockout_1w"]    = grp["Stockout Flag"].shift(1)

    # ---------- feature list ----------
    features = [
        "Year", "Month", "Week Of Year", "quarter", "is_year_end", "is_q1",
        "month_sin", "month_cos", "week_sin", "week_cos",
        "lag_1w", "lag_2w", "lag_4w", "lag_52w",
        "rolling_mean_4w", "rolling_mean_12w", "rolling_std_4w", "rolling_std_12w",
        "rolling_max_4w", "ewm_4w",
        "Temperature F", "Precipitation Mm", "Holiday Flag", "Promotion Flag",
        "lag_promo_1w", "promo_x_holiday",
        "Unit Price", "price_vs_avg",
        "inventory_coverage", "lag_stockout_1w",
        "Category", "Region", "Store Id", "Product Id",
    ]
    target = "Sales Quantity"

    model_df = df[["Date"] + features + [target, "Product Name", "Baseline Forecast"]].dropna().copy()

    # encode categoricals
    cat_cols = ["Category", "Region", "Store Id", "Product Id"]
    for c in cat_cols:
        model_df[c] = model_df[c].astype("category")

    # ---------- time-based split ----------
    train = model_df[model_df["Date"] < "2023-07-01"]
    val   = model_df[(model_df["Date"] >= "2023-07-01") & (model_df["Date"] < "2023-11-01")]
    test  = model_df[model_df["Date"] >= "2023-11-01"]

    X_train, y_train = train[features], train[target]
    X_val,   y_val   = val[features],   val[target]
    X_test,  y_test  = test[features],  test[target]

    # ---------- train LightGBM ----------
    lgb_model = lgb.LGBMRegressor(
        objective="regression", metric="mae", boosting_type="gbdt",
        n_estimators=1000, num_leaves=63, learning_rate=0.05,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
        min_child_samples=20, verbose=-1, random_state=42,
    )
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )

    # predictions
    test = test.copy()
    test["Predicted"] = np.clip(lgb_model.predict(X_test), 0, None)
    test["Residual"]  = test[target] - test["Predicted"]

    # feature importance
    fi = pd.DataFrame({
        "Feature": features,
        "Importance": lgb_model.feature_importances_,
    }).sort_values("Importance", ascending=False)

    # metrics
    metrics = {
        "MAE":  mean_absolute_error(y_test, test["Predicted"]),
        "RMSE": np.sqrt(mean_squared_error(y_test, test["Predicted"])),
        "R2":   r2_score(y_test, test["Predicted"]),
        "MAPE": np.mean(np.abs((y_test - test["Predicted"]) / y_test.replace(0, 1))) * 100,
        "Baseline_MAE": mean_absolute_error(y_test, test["Baseline Forecast"]),
    }

    # Keep the full engineered df for future prediction lag lookups
    return df, model_df, test, lgb_model, features, fi, metrics


df_raw, model_df, test_df, model, feature_list, feat_imp, metrics = load_and_train()

# ─────────────────────────────────────────────
# Helper: predict future weeks for a store-product
# ─────────────────────────────────────────────
def predict_future(store_id, product_id, weeks_ahead, promo_flags, holiday_flags, temp, precip):
    """Generate future weekly predictions by rolling forward one week at a time."""
    sub = model_df[
        (model_df["Store Id"] == store_id) & (model_df["Product Id"] == product_id)
    ].sort_values("Date").copy()

    if sub.empty:
        return pd.DataFrame()

    last_date = sub["Date"].max()
    product_info = df_raw[df_raw["Product Id"] == product_id].iloc[0]
    category     = product_info["Category"]
    region       = product_info["Region"]
    unit_price   = product_info["Unit Price"]
    product_name = product_info["Product Name"]

    # Category average price for price_vs_avg
    cat_avg_price = df_raw[df_raw["Category"] == category]["Unit Price"].mean()

    # Historical sales for lags/rolling (use raw Sales Quantity column)
    hist_sales = sub["Sales Quantity"].tolist()
    hist_promo = sub["Promotion Flag"].tolist()
    hist_stockout = sub["lag_stockout_1w"].tolist()

    results = []
    for w in range(weeks_ahead):
        future_date = last_date + pd.Timedelta(weeks=w + 1)
        month   = future_date.month
        week_oy = future_date.isocalendar()[1]
        year    = future_date.year
        quarter = (month - 1) // 3 + 1
        promo   = promo_flags[w] if w < len(promo_flags) else 0
        holiday = holiday_flags[w] if w < len(holiday_flags) else 0

        # Lag features from extended history
        lag_1w  = hist_sales[-1] if len(hist_sales) >= 1 else 0
        lag_2w  = hist_sales[-2] if len(hist_sales) >= 2 else 0
        lag_4w  = hist_sales[-4] if len(hist_sales) >= 4 else 0
        lag_52w = hist_sales[-52] if len(hist_sales) >= 52 else hist_sales[0]

        recent = hist_sales[-1:]  # shifted by 1
        rm4  = np.mean(hist_sales[-4:])  if len(hist_sales) >= 4  else np.mean(hist_sales)
        rm12 = np.mean(hist_sales[-12:]) if len(hist_sales) >= 12 else np.mean(hist_sales)
        rs4  = np.std(hist_sales[-4:])   if len(hist_sales) >= 4  else 0
        rs12 = np.std(hist_sales[-12:])  if len(hist_sales) >= 12 else 0
        rmax4 = max(hist_sales[-4:])     if len(hist_sales) >= 4  else max(hist_sales)

        # EWM approximation
        alpha = 2 / (4 + 1)
        ewm_val = hist_sales[-1]
        for s in reversed(hist_sales[-4:]):
            ewm_val = alpha * s + (1 - alpha) * ewm_val

        lag_promo_1w = hist_promo[-1] if len(hist_promo) >= 1 else 0
        lag_stockout = hist_stockout[-1] if len(hist_stockout) >= 1 else 0

        row = {
            "Year": year, "Month": month, "Week Of Year": week_oy,
            "quarter": quarter,
            "is_year_end": int(month in [11, 12]),
            "is_q1": int(quarter == 1),
            "month_sin": np.sin(2 * np.pi * month / 12),
            "month_cos": np.cos(2 * np.pi * month / 12),
            "week_sin": np.sin(2 * np.pi * week_oy / 52),
            "week_cos": np.cos(2 * np.pi * week_oy / 52),
            "lag_1w": lag_1w, "lag_2w": lag_2w, "lag_4w": lag_4w, "lag_52w": lag_52w,
            "rolling_mean_4w": rm4, "rolling_mean_12w": rm12,
            "rolling_std_4w": rs4, "rolling_std_12w": rs12,
            "rolling_max_4w": rmax4, "ewm_4w": ewm_val,
            "Temperature F": temp, "Precipitation Mm": precip,
            "Holiday Flag": holiday, "Promotion Flag": promo,
            "lag_promo_1w": lag_promo_1w, "promo_x_holiday": promo * holiday,
            "Unit Price": unit_price,
            "price_vs_avg": unit_price / cat_avg_price if cat_avg_price else 1,
            "inventory_coverage": sub["inventory_coverage"].iloc[-1],
            "lag_stockout_1w": lag_stockout,
            "Category": category, "Region": region,
            "Store Id": store_id, "Product Id": product_id,
        }

        row_df = pd.DataFrame([row])
        for c in ["Category", "Region", "Store Id", "Product Id"]:
            row_df[c] = row_df[c].astype("category")

        pred = max(0, model.predict(row_df[feature_list])[0])
        hist_sales.append(pred)
        hist_promo.append(promo)
        hist_stockout.append(0)

        results.append({
            "Date": future_date, "Week": w + 1,
            "Predicted Sales": round(pred, 1),
            "Promotion": "Yes" if promo else "No",
            "Holiday": "Yes" if holiday else "No",
            "Product Name": product_name, "Category": category,
        })

    return pd.DataFrame(results)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("## 📦 Supply Chain Demand Forecasting Dashboard")
st.markdown("Interactive exploration of 26,250 weekly demand records across 10 stores, 25 products, and 5 regions (Jan 2022 – Dec 2023).")

tab_eda, tab_forecast = st.tabs(["EDA Findings", "Forecasting Model"])


# ═══════════════════════════════════════════════
# TAB 1 — EDA
# ═══════════════════════════════════════════════
with tab_eda:

    # --- KPI row ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(df_raw):,}")
    c2.metric("Products", df_raw["Product Id"].nunique())
    c3.metric("Stores", df_raw["Store Id"].nunique())
    c4.metric("Avg Weekly Sales", f"{df_raw['Sales Quantity'].mean():.0f} units")

    st.divider()

    # --- 1. Target distribution ---
    st.subheader("1. Sales Quantity Distribution")
    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.histogram(df_raw, x="Sales Quantity", nbins=60,
                           color_discrete_sequence=["#667eea"],
                           title="Raw Distribution (skewness ≈ 2.0)")
        fig.update_layout(height=350, margin=dict(t=40, b=30))
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        fig = px.histogram(df_raw, x=np.log1p(df_raw["Sales Quantity"]), nbins=60,
                           color_discrete_sequence=["#764ba2"],
                           title="Log-Transformed Distribution",
                           labels={"x": "log1p(Sales Quantity)"})
        fig.update_layout(height=350, margin=dict(t=40, b=30))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    <b>Insight:</b> Raw sales are heavily right-skewed (skewness ≈ 2.0). Most weeks see 20–100 units,
    but occasional spikes reach 500+. The log transformation normalises the distribution significantly.
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # --- 2. Weekly sales trend ---
    st.subheader("2. Weekly Sales Trend")
    weekly = df_raw.groupby("Date")["Sales Quantity"].sum().reset_index()
    fig = px.line(weekly, x="Date", y="Sales Quantity",
                  title="Total Weekly Sales Across All Products",
                  color_discrete_sequence=["#667eea"])
    fig.update_layout(height=350, margin=dict(t=40, b=30))
    st.plotly_chart(fig, use_container_width=True)

    # --- 3. Sales by category ---
    st.subheader("3. Weekly Sales by Category")
    cat_weekly = df_raw.groupby(["Date", "Category"])["Sales Quantity"].sum().reset_index()
    fig = px.line(cat_weekly, x="Date", y="Sales Quantity", color="Category",
                  title="Category-Level Weekly Trends",
                  color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(height=400, margin=dict(t=40, b=30))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    <b>Insight:</b> Demand fluctuates around a stable mean with no strong upward/downward drift.
    Category-level trends are visually distinct — Electronics and Sports show higher variance.
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # --- 4. Seasonality heatmap ---
    st.subheader("4. Seasonality Heatmap")
    heat_data = df_raw.groupby(["Category", "Week Of Year"])["Sales Quantity"].mean().reset_index()
    heat_pivot = heat_data.pivot(index="Category", columns="Week Of Year", values="Sales Quantity")
    fig = px.imshow(heat_pivot, aspect="auto",
                    color_continuous_scale="YlOrRd",
                    title="Avg Sales by Week of Year & Category",
                    labels=dict(x="Week of Year", y="Category", color="Avg Sales"))
    fig.update_layout(height=350, margin=dict(t=40, b=30))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --- 5. Promotion & holiday impact ---
    st.subheader("5. Promotion & Holiday Impact")
    col_p, col_h = st.columns(2)

    with col_p:
        promo_avg = df_raw.groupby(["Category", "Promotion Flag"])["Sales Quantity"].mean().reset_index()
        promo_avg["Promotion Flag"] = promo_avg["Promotion Flag"].map({0: "No Promo", 1: "Promo"})
        fig = px.bar(promo_avg, x="Category", y="Sales Quantity", color="Promotion Flag",
                     barmode="group", title="Promotion Effect by Category",
                     color_discrete_sequence=["#a0aec0", "#667eea"])
        fig.update_layout(height=380, margin=dict(t=40, b=30))
        st.plotly_chart(fig, use_container_width=True)

    with col_h:
        hol_avg = df_raw.groupby(["Category", "Holiday Flag"])["Sales Quantity"].mean().reset_index()
        hol_avg["Holiday Flag"] = hol_avg["Holiday Flag"].map({0: "Non-Holiday", 1: "Holiday"})
        fig = px.bar(hol_avg, x="Category", y="Sales Quantity", color="Holiday Flag",
                     barmode="group", title="Holiday Effect by Category",
                     color_discrete_sequence=["#a0aec0", "#e53e3e"])
        fig.update_layout(height=380, margin=dict(t=40, b=30))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    <b>Insight:</b> Promotions drive a ~54% sales lift and holidays drive a ~39% lift — both strong,
    actionable signals. The effect is consistent across all five categories.
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # --- 6. Regional sales distribution ---
    st.subheader("6. Sales by Region & Store")
    col_r, col_s = st.columns(2)
    with col_r:
        fig = px.box(df_raw, x="Region", y="Sales Quantity",
                     color="Region", title="Sales Distribution by Region",
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(height=400, showlegend=False, margin=dict(t=40, b=30))
        st.plotly_chart(fig, use_container_width=True)
    with col_s:
        fig = px.box(df_raw, x="Store Id", y="Sales Quantity",
                     color="Store Id", title="Sales Distribution by Store",
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(height=400, showlegend=False, margin=dict(t=40, b=30))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --- 7. Correlation heatmap ---
    st.subheader("7. Feature Correlation Matrix")
    num_cols = ["Sales Quantity", "Unit Price", "Competitor Price",
                "Inventory Start", "Inventory End", "Temperature F",
                "Precipitation Mm", "Baseline Forecast"]
    corr = df_raw[num_cols].corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                    title="Pearson Correlation — Numeric Features", aspect="auto")
    fig.update_layout(height=500, margin=dict(t=40, b=30))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    <b>Insight:</b> Weather variables (Temperature, Precipitation) show near-zero correlation with sales (r ≈ 0.05–0.06).
    The Baseline Forecast has moderate positive correlation but leaves substantial room for improvement.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# TAB 2 — FORECASTING MODEL
# ═══════════════════════════════════════════════
with tab_forecast:

    # --- Model performance overview ---
    st.subheader("Model Performance — LightGBM vs Baseline")

    m1, m2, m3, m4 = st.columns(4)
    mae_improve = (1 - metrics["MAE"] / metrics["Baseline_MAE"]) * 100
    m1.metric("LightGBM MAE", f"{metrics['MAE']:.2f}", delta=f"{mae_improve:.1f}% vs baseline", delta_color="normal")
    m2.metric("RMSE", f"{metrics['RMSE']:.2f}")
    m3.metric("R² Score", f"{metrics['R2']:.3f}")
    m4.metric("MAPE", f"{metrics['MAPE']:.1f}%")

    st.divider()

    # --- Feature importance ---
    st.subheader("Feature Importance (Top 15)")
    top15 = feat_imp.head(15)
    fig = px.bar(top15, x="Importance", y="Feature", orientation="h",
                 color="Importance", color_continuous_scale="Viridis",
                 title="LightGBM Feature Importance")
    fig.update_layout(height=450, yaxis=dict(autorange="reversed"), margin=dict(t=40, b=30))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --- Interactive forecasting section ---
    st.subheader("Interactive Forecast Explorer")
    st.markdown("Use the dropdowns below to filter predictions and compare **Actual vs Predicted** demand.")

    # Dropdowns
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)

    categories = ["All"] + sorted(test_df["Category"].astype(str).unique().tolist())
    regions    = ["All"] + sorted(test_df["Region"].astype(str).unique().tolist())
    stores     = ["All"] + sorted(test_df["Store Id"].astype(str).unique().tolist())
    products   = ["All"] + sorted(test_df["Product Id"].astype(str).unique().tolist())

    with col_f1:
        sel_cat = st.selectbox("Category", categories)
    with col_f2:
        sel_reg = st.selectbox("Region", regions)
    with col_f3:
        sel_store = st.selectbox("Store", stores)
    with col_f4:
        sel_prod = st.selectbox("Product", products)

    # Filter
    filtered = test_df.copy()
    if sel_cat != "All":
        filtered = filtered[filtered["Category"].astype(str) == sel_cat]
    if sel_reg != "All":
        filtered = filtered[filtered["Region"].astype(str) == sel_reg]
    if sel_store != "All":
        filtered = filtered[filtered["Store Id"].astype(str) == sel_store]
    if sel_prod != "All":
        filtered = filtered[filtered["Product Id"].astype(str) == sel_prod]

    if filtered.empty:
        st.warning("No data matches the selected filters. Please adjust your selections.")
    else:
        # Filtered metrics
        f_mae  = mean_absolute_error(filtered["Sales Quantity"], filtered["Predicted"])
        f_rmse = np.sqrt(mean_squared_error(filtered["Sales Quantity"], filtered["Predicted"]))
        f_r2   = r2_score(filtered["Sales Quantity"], filtered["Predicted"]) if len(filtered) > 1 else 0
        f_mape = np.mean(np.abs(filtered["Residual"]) / filtered["Sales Quantity"].replace(0, 1)) * 100

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Filtered MAE", f"{f_mae:.2f}")
        mc2.metric("Filtered RMSE", f"{f_rmse:.2f}")
        mc3.metric("Filtered R²", f"{f_r2:.3f}")
        mc4.metric("Records", f"{len(filtered):,}")

        # --- Actual vs Predicted time series ---
        ts = filtered.groupby("Date").agg(
            Actual=("Sales Quantity", "sum"),
            Predicted=("Predicted", "sum"),
            Baseline=("Baseline Forecast", "sum"),
        ).reset_index()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts["Date"], y=ts["Actual"], name="Actual",
                                 line=dict(color="#2d3748", width=2.5)))
        fig.add_trace(go.Scatter(x=ts["Date"], y=ts["Predicted"], name="LightGBM Predicted",
                                 line=dict(color="#667eea", width=2, dash="dash")))
        fig.add_trace(go.Scatter(x=ts["Date"], y=ts["Baseline"], name="Baseline Forecast",
                                 line=dict(color="#e53e3e", width=1.5, dash="dot")))
        fig.update_layout(title="Actual vs Predicted Demand (Test Period: Nov–Dec 2023)",
                          xaxis_title="Date", yaxis_title="Sales Quantity",
                          height=450, legend=dict(orientation="h", yanchor="bottom", y=1.02),
                          margin=dict(t=60, b=30))
        st.plotly_chart(fig, use_container_width=True)

        # --- Scatter: predicted vs actual ---
        col_sc, col_res = st.columns(2)
        with col_sc:
            fig = px.scatter(filtered, x="Sales Quantity", y="Predicted",
                             color="Category" if sel_cat == "All" else None,
                             title="Predicted vs Actual",
                             color_discrete_sequence=px.colors.qualitative.Set2,
                             opacity=0.6)
            max_val = max(filtered["Sales Quantity"].max(), filtered["Predicted"].max())
            fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val],
                                     mode="lines", line=dict(dash="dash", color="gray"),
                                     name="Perfect Prediction"))
            fig.update_layout(height=400, margin=dict(t=40, b=30))
            st.plotly_chart(fig, use_container_width=True)

        with col_res:
            fig = px.histogram(filtered, x="Residual", nbins=40,
                               color_discrete_sequence=["#667eea"],
                               title="Residual Distribution (Actual − Predicted)")
            fig.add_vline(x=0, line_dash="dash", line_color="red")
            fig.update_layout(height=400, margin=dict(t=40, b=30))
            st.plotly_chart(fig, use_container_width=True)

        # --- Data table ---
        with st.expander("View Filtered Forecast Data"):
            display_cols = ["Date", "Store Id", "Region", "Product Id", "Product Name",
                            "Category", "Sales Quantity", "Predicted", "Baseline Forecast", "Residual"]
            st.dataframe(
                filtered[display_cols].sort_values("Date").reset_index(drop=True),
                use_container_width=True, height=400,
            )

    # ═══════════════════════════════════════════════
    # FUTURE DEMAND PREDICTOR
    # ═══════════════════════════════════════════════
    st.divider()
    st.subheader("🔮 Future Demand Predictor")
    st.markdown("Select a store-product combination and scenario to forecast demand for upcoming weeks.")

    fp_col1, fp_col2, fp_col3, fp_col4 = st.columns(4)

    all_stores   = sorted(df_raw["Store Id"].unique().tolist())
    all_products = sorted(df_raw["Product Id"].unique().tolist())
    product_names = df_raw.drop_duplicates("Product Id").set_index("Product Id")["Product Name"].to_dict()
    product_labels = [f"{pid} — {product_names.get(pid, '')}" for pid in all_products]

    with fp_col1:
        fp_store = st.selectbox("Store", all_stores, key="fp_store")
    with fp_col2:
        fp_prod_label = st.selectbox("Product", product_labels, key="fp_prod")
        fp_prod = fp_prod_label.split(" — ")[0]
    with fp_col3:
        fp_weeks = st.slider("Weeks to Forecast", min_value=1, max_value=12, value=4, key="fp_weeks")
    with fp_col4:
        fp_temp  = st.number_input("Avg Temperature (°F)", value=55.0, step=5.0, key="fp_temp")
        fp_precip = st.number_input("Avg Precipitation (mm)", value=50.0, step=10.0, key="fp_precip")

    # Promotion & holiday toggles per week
    st.markdown("**Set promotion & holiday flags per week:**")
    promo_cols = st.columns(min(fp_weeks, 12))
    promo_flags = []
    holiday_flags = []
    for w in range(fp_weeks):
        with promo_cols[w % len(promo_cols)]:
            st.markdown(f"**Wk {w+1}**")
            promo_flags.append(1 if st.checkbox("Promo", key=f"promo_{w}") else 0)
            holiday_flags.append(1 if st.checkbox("Holiday", key=f"hol_{w}") else 0)

    if st.button("Generate Forecast", type="primary", use_container_width=True):
        future_df = predict_future(fp_store, fp_prod, fp_weeks, promo_flags, holiday_flags, fp_temp, fp_precip)

        if future_df.empty:
            st.error("No historical data found for this store-product combination.")
        else:
            # Show metrics
            fp_m1, fp_m2, fp_m3 = st.columns(3)
            fp_m1.metric("Avg Weekly Forecast", f"{future_df['Predicted Sales'].mean():.0f} units")
            fp_m2.metric("Peak Week", f"{future_df['Predicted Sales'].max():.0f} units")
            fp_m3.metric("Total Forecast", f"{future_df['Predicted Sales'].sum():.0f} units")

            # Historical + future chart
            hist_sub = df_raw[
                (df_raw["Store Id"] == fp_store) & (df_raw["Product Id"] == fp_prod)
            ].groupby("Date")["Sales Quantity"].sum().reset_index().tail(26)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist_sub["Date"], y=hist_sub["Sales Quantity"],
                name="Historical Actual", line=dict(color="#2d3748", width=2),
            ))
            fig.add_trace(go.Scatter(
                x=future_df["Date"], y=future_df["Predicted Sales"],
                name="Future Forecast", line=dict(color="#667eea", width=2.5, dash="dash"),
                mode="lines+markers", marker=dict(size=8),
            ))
            # Shade forecast region
            fig.add_vrect(
                x0=future_df["Date"].min(), x1=future_df["Date"].max(),
                fillcolor="#667eea", opacity=0.07, line_width=0,
                annotation_text="Forecast", annotation_position="top left",
            )
            fig.update_layout(
                title=f"Demand Forecast — {fp_store} / {fp_prod} ({product_names.get(fp_prod, '')})",
                xaxis_title="Date", yaxis_title="Sales Quantity",
                height=450, legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(t=60, b=30),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Forecast table
            st.dataframe(
                future_df[["Date", "Week", "Predicted Sales", "Promotion", "Holiday"]],
                use_container_width=True, hide_index=True,
            )

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.divider()
st.caption("Built with Streamlit & LightGBM | Data: Supply Chain Demand Forecasting v3 | Test Period: Nov–Dec 2023")
