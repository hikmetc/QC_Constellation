# patient_qc_app.py
# Developed by Hikmet Can Çubukçu – user-defined TEa added 2025-07-13
# Revised 2026-01-25 for full compatibility with Rust Desktop App v1.0.8

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import statistics

##############################################################################
# -------------------------------- HELPERS --------------------------------- #
##############################################################################

# Custom Box-Cox implementation to match Rust Desktop App (Golden Section Search)
def custom_boxcox_rust_style(data):
    # Clean data (ensure numpy array)
    data = np.array(data, dtype=float)
    n = float(len(data))
    sum_ln_x = np.sum(np.log(data))

    def cost_function(l):
        if abs(l) < 1e-9:
            y = np.log(data)
        else:
            y = (np.power(data, l) - 1.0) / l
        
        mean_y = np.mean(y)
        # Using population variance (N) or Sample (N-1)?
        # Rust `stats.rs`: let variance = (transformed_sq_sum / n) - (mean * mean);
        # This is Population Variance (div by N).
        # numpy var default is population (ddof=0).
        variance = np.var(y)
        
        if variance <= 0.0:
            return float('inf')
        
        # Negative Log-Likelihood
        return (n / 2.0) * np.log(variance) - (l - 1.0) * sum_ln_x

    # Golden Section Search [-5.0, 5.0]
    a, b = -5.0, 5.0
    tol = 1e-4
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    resphi = 2.0 - phi

    c = a + resphi * (b - a)
    d = b - resphi * (b - a)
    fc = cost_function(c)
    fd = cost_function(d)

    for _ in range(100):
        if abs(b - a) < tol:
            break
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = a + resphi * (b - a)
            fc = cost_function(c)
        else:
            a = c
            c = d
            fc = fd
            d = b - resphi * (b - a)
            fd = cost_function(d)
    
    optimal_lambda = (a + b) / 2.0

    # Transform with optimal lambda
    if abs(optimal_lambda) < 1e-9:
        fitted = np.log(data)
    else:
        fitted = (np.power(data, optimal_lambda) - 1.0) / optimal_lambda
    
    return pd.Series(fitted), optimal_lambda


def get_uploaded_data():
    with st.sidebar:
        with open("./template/Na_normal.xlsx", "rb") as f:
            st.download_button("Click to Download Template File",
                               data=f.read(),
                               file_name="Na_normal.xlsx")
        up = st.file_uploader("#### **Upload your .xlsx (Excel) or .csv file:**",
                              type=["csv", "xlsx"])
    if up is None:
        return None, None, None
    try:
        df = pd.read_excel(up)
    except Exception:
        df = pd.read_csv(up, sep=None, engine="python")

    col_res = st.sidebar.selectbox("**Select patient results column**", df.columns)
    col_day = st.sidebar.selectbox("**Select day/batch column**", df.columns, key="daycol")

    res = df[col_res].dropna().reset_index(drop=True)
    day = df[col_day].iloc[res.index].reset_index(drop=True)
    return res, day, df


def truncate_series(s, lo, hi):
    return s[(s >= lo) & (s <= hi)].reset_index(drop=True)


def box_cox_if_needed(s, do_it):
    if not do_it:
        return s
    fitted, _ = custom_boxcox_rust_style(s)
    return fitted


def perf_df(value_err, value_0, anped, mnped):
    return pd.DataFrame(
        {"Metric": ["Sensitivity (TPR)", "Specificity", "False Positive Rate",
                    "Youden Index", "ANPed", "MNPed"],
         "Value": [value_err, 1 - value_0, value_0,
                   value_err - value_0, anped, mnped]}
    )


def plot_anped(errs, anpeds, mnpeds, title):
    fig = go.Figure()
    pos = errs >= 0
    fig.add_trace(go.Scatter(x=errs[pos], y=anpeds[pos],
                             mode="lines+markers", name="ANPed (+)"))
    fig.add_trace(go.Scatter(x=errs[~pos], y=anpeds[~pos],
                             mode="lines+markers", name="ANPed (−)"))
    fig.add_trace(go.Scatter(x=errs[pos], y=mnpeds[pos],
                             mode="lines+markers", name="MNPed (+)"))
    fig.add_trace(go.Scatter(x=errs[~pos], y=mnpeds[~pos],
                             mode="lines+markers", name="MNPed (−)"))
    fig.add_vline(x=0, line=dict(color="red", dash="dash"))
    fig.update_layout(title=title, xaxis_title="Error Rate (%)",
                      yaxis_title="ANPed / MNPed",
                      title_font=dict(color="#cc0000"))
    st.plotly_chart(fig, use_container_width=True)


##############################################################################
# -----------------------------  CALCULATIONS ------------------------------ #
##############################################################################
def run_ewma(raw, day, lo, hi, transform, block,
             LCL, UCL, target_mean, prev_sd,
             TEa, error_pt=10):
    data = truncate_series(raw, lo, hi)
    day = day.iloc[data.index].reset_index(drop=True)
    data = box_cox_if_needed(data, transform == "Box-Cox")

    alerts_0 = alerts_err = 0
    eD_err = []

    for err_flag in (0, 1):
        for d in range(1, day.nunique() + 1):
            sub = data[day == d].reset_index(drop=True)
            if sub.empty:
                continue
            if err_flag and len(sub) > error_pt:
                sub.iloc[error_pt:] *= 1 + TEa / 100
            
            # Initialize with target_mean matches Rust Desktop App v1.0.8 logic
            alpha = 2.0 / (block + 1.0)
            ewma_list = []
            curr_ewma = target_mean
            
            # The loop exactly matches Rust `prev_ewma = (1-a)*prev + a*val`
            for val in sub:
                curr_ewma = (1.0 - alpha) * curr_ewma + alpha * val
                ewma_list.append(curr_ewma)
            
            ewma = pd.Series(ewma_list)
            
            out_hi = ewma >= UCL
            out_lo = ewma <= LCL
            if out_hi.any() or out_lo.any():
                if err_flag:
                    alerts_err += 1
                else:
                    alerts_0 += 1
            if err_flag:
                idx_hi = out_hi[out_hi].index.min()
                idx_lo = out_lo[out_lo].index.min()
                first = min([x for x in (idx_hi, idx_lo) if pd.notna(x)], default=None)
                if first is not None and first + 1 >= error_pt:
                    eD_err.append(first + 1 - error_pt)

    FPR = alerts_0 / day.nunique()
    TPR = alerts_err / day.nunique()
    ANPed = statistics.mean(eD_err) if eD_err else float("nan")
    MNPed = statistics.median(eD_err) if eD_err else float("nan")
    return TPR, FPR, ANPed, MNPed


def run_cusum(raw, day, lo, hi, transform, h,
              target_mean, prev_sd, k, TEa, error_pt=10):
    data = truncate_series(raw, lo, hi)
    day = day.iloc[data.index].reset_index(drop=True)
    data = box_cox_if_needed(data, transform == "Box-Cox")

    mu, sd = target_mean, prev_sd
    alerts_0 = alerts_err = 0
    eD_err = []

    for err_flag in (0, 1):
        for d in range(1, day.nunique() + 1):
            sub = data[day == d].reset_index(drop=True)
            if sub.empty:
                continue
            if err_flag and len(sub) > error_pt:
                sub.iloc[error_pt:] *= 1 + TEa / 100
            
            Cp = Cm = 0
            hit = False
            for idx, x in enumerate(sub):
                z = (x - mu) / sd
                # Matches Rust logic exactly: max(0, prev + z - k)
                Cp = max(0, Cp + z - k)
                Cm = max(0, Cm - z - k)
                if Cp >= h or Cm >= h:
                    hit = True
                    if err_flag and idx + 1 >= error_pt:
                        eD_err.append(idx + 1 - error_pt)
                    break
            if hit:
                if err_flag:
                    alerts_err += 1
                else:
                    alerts_0 += 1

    FPR = alerts_0 / day.nunique()
    TPR = alerts_err / day.nunique()
    ANPed = statistics.mean(eD_err) if eD_err else float("nan")
    MNPed = statistics.median(eD_err) if eD_err else float("nan")
    return TPR, FPR, ANPed, MNPed


##############################################################################
# -------------------------------  UI LAYOUT  ------------------------------ #
##############################################################################
st.set_page_config(page_title="Patient-based QC optimiser", layout="wide")
st.markdown("#### **:blue[Verify Moving Average Charts for Patient-Based QC]**")
st.write("---")

raw_res, raw_day, _ = get_uploaded_data()
if raw_res is None:
    st.info("Please upload data first. For large files prefer .csv.", icon="ℹ️")
    st.stop()

tab_ewma, tab_cusum = st.tabs(["📉 **EWMA** (user defined)",
                               "📈 **CUSUM** (user defined)"])

# ------------------------------------------------------------------------- #
#                                   EWMA                                    #
# ------------------------------------------------------------------------- #
with tab_ewma:
    with st.form("ewma_form"):
        st.subheader("EWMA parameters")
        c1, c2, c3 = st.columns(3)

        lo_tl = c1.number_input("Lower truncation limit",
                                value=float(raw_res.min()), format="%.8f", step=0.00000001)
        hi_tl = c1.number_input("Upper truncation limit",
                                value=float(raw_res.max()), format="%.8f", step=0.00000001)

        transform = c2.radio("Transformation", ("Raw Data", "Box-Cox"))

        # Initialize with reasonably high precision
        mean_val = float(raw_res.mean())
        std_val = float(raw_res.std())

        LCL = c2.number_input("Lower control limit (absolute)",
                              value=mean_val - 2 * std_val, format="%.8f", step=0.00000001)
        UCL = c2.number_input("Upper control limit (absolute)",
                              value=mean_val + 2 * std_val, format="%.8f", step=0.00000001)

        block = c3.number_input("EWMA block size (span)",
                                value=20, min_value=2, step=1)

        target_mean = c3.number_input("Target mean",
                                      value=mean_val, format="%.8f", step=0.00000001)
        prev_sd = c3.number_input("Previous SD", 
                                  value=std_val, format="%.8f", step=0.00000001)

        TEa_user = c3.number_input("Allowable error (%)",
                                   value=5.0, min_value=0.0, step=0.000001, format="%.6f")

        calc_ewma = st.form_submit_button("Calculate 🟢")

    if calc_ewma:
        TPR, FPR, ANPed, MNPed = run_ewma(
            raw_res, raw_day, lo_tl, hi_tl, transform,
            block, LCL, UCL, target_mean, prev_sd, TEa_user
        )
        st.markdown("##### **Performance**")
        st.dataframe(perf_df(TPR, FPR, ANPed, MNPed), hide_index=True)

        errs = np.arange(-TEa_user, TEa_user + 0.1 * TEa_user, 0.1 * TEa_user) if TEa_user else np.array([0])
        an_list, mn_list = [], []
        # Filter errs near 0
        errs = [e if abs(e) > 1e-9 else 0.0 for e in errs]
        
        for e in errs:
            _, _, an, mn = run_ewma(
                raw_res, raw_day, lo_tl, hi_tl, transform,
                block, LCL, UCL, target_mean, prev_sd, e
            )
            an_list.append(an)
            mn_list.append(mn)
        plot_anped(np.array(errs), np.array(an_list), np.array(mn_list),
                   f"EWMA – ANPed / MNPed vs error rate (±{TEa_user} %)")

# ------------------------------------------------------------------------- #
#                                   CUSUM                                   #
# ------------------------------------------------------------------------- #
with tab_cusum:
    with st.form("cusum_form"):
        st.subheader("CUSUM parameters")
        c1, c2, c3 = st.columns(3)

        lo_tl = c1.number_input("Lower truncation limit",
                                value=float(raw_res.min()), key="lo_c", format="%.8f", step=0.00000001)
        hi_tl = c1.number_input("Upper truncation limit",
                                value=float(raw_res.max()), key="hi_c", format="%.8f", step=0.00000001)

        transform = c2.radio("Transformation",
                             ("Raw Data", "Box-Cox"), key="tf_c")

        h = c2.number_input("Control limit *h*",
                            value=10.0, format="%.8f", step=0.00000001)

        mean_val = float(raw_res.mean())
        std_val = float(raw_res.std())
        
        target_mean = c3.number_input("Target mean",
                                      value=mean_val, key="tm_c", format="%.8f", step=0.00000001)
        prev_sd = c3.number_input("Previous SD",
                                  value=std_val, key="sd_c", format="%.8f", step=0.00000001)

        k_val = 0.5 # Fixed as per Rust logic

        TEa_user_c = c3.number_input("Allowable error (%)",
                                     value=5.0, min_value=0.0, step=0.000001,
                                     key="tea_c", format="%.6f")

        calc_c = st.form_submit_button("Calculate 🟢")

    if calc_c:
        TPR, FPR, ANPed, MNPed = run_cusum(
            raw_res, raw_day, lo_tl, hi_tl, transform,
            h, target_mean, prev_sd, k_val, TEa_user_c
        )
        st.markdown("##### **Performance**")
        st.dataframe(perf_df(TPR, FPR, ANPed, MNPed), hide_index=True)

        errs = np.arange(-TEa_user_c, TEa_user_c + 0.1 * TEa_user_c, 0.1 * TEa_user_c) if TEa_user_c else np.array([0])
        an_list, mn_list = [], []
        errs = [e if abs(e) > 1e-9 else 0.0 for e in errs]

        for e in errs:
            _, _, an, mn = run_cusum(
                raw_res, raw_day, lo_tl, hi_tl, transform,
                h, target_mean, prev_sd, k_val, e
            )
            an_list.append(an)
            mn_list.append(mn)
        plot_anped(np.array(errs), np.array(an_list), np.array(mn_list),
                   f"CUSUM – ANPed / MNPed vs error rate (±{TEa_user_c} %)")

##############################################################################
st.sidebar.image("./images/QC Constellation icon.png", use_container_width=True)
st.sidebar.info("*Developed by Hikmet Can Çubukçu, MD, MSc, PhD, EuSpLM*  \n<hikmetcancubukcu@gmail.com>")
