# patient_qc_app.py
# Developed by Hikmet Can Çubukçu – user-defined TEa added 2025-07-13
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
import statistics

##############################################################################
# -------------------------------- HELPERS --------------------------------- #
##############################################################################
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
    fitted, _ = stats.boxcox(s)
    return pd.Series(fitted)


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
            ewma = sub.ewm(span=block, adjust=False).mean()
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
                                value=float(raw_res.min()))
        hi_tl = c1.number_input("Upper truncation limit",
                                value=float(raw_res.max()))

        transform = c2.radio("Transformation", ("Raw Data", "Box-Cox"))

        LCL = c2.number_input("Lower control limit (absolute)",
                              value=float(raw_res.mean() - 2 * raw_res.std()))
        UCL = c2.number_input("Upper control limit (absolute)",
                              value=float(raw_res.mean() + 2 * raw_res.std()))

        block = c3.number_input("EWMA block size (span)",
                                value=20, min_value=2, step=1)

        target_mean = c3.number_input("Target mean",
                                      value=float(raw_res.mean()))
        prev_sd = c3.number_input("Previous SD", value=float(raw_res.std()))

        TEa_user = c3.number_input("Allowable error (%)",
                                   value=5.0, min_value=0.0, step=0.1)

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
        for e in errs:
            _, _, an, mn = run_ewma(
                raw_res, raw_day, lo_tl, hi_tl, transform,
                block, LCL, UCL, target_mean, prev_sd, e
            )
            an_list.append(an)
            mn_list.append(mn)
        plot_anped(errs, np.array(an_list), np.array(mn_list),
                   f"EWMA – ANPed / MNPed vs error rate (±{TEa_user} %)")

# ------------------------------------------------------------------------- #
#                                   CUSUM                                   #
# ------------------------------------------------------------------------- #
with tab_cusum:
    with st.form("cusum_form"):
        st.subheader("CUSUM parameters")
        c1, c2, c3 = st.columns(3)

        lo_tl = c1.number_input("Lower truncation limit",
                                value=float(raw_res.min()), key="lo_c")
        hi_tl = c1.number_input("Upper truncation limit",
                                value=float(raw_res.max()), key="hi_c")

        transform = c2.radio("Transformation",
                             ("Raw Data", "Box-Cox"), key="tf_c")

        h = c2.number_input("Control limit *h*",
                            value=10.0, step=0.1)

        target_mean = c3.number_input("Target mean",
                                      value=float(raw_res.mean()), key="tm_c")
        prev_sd = c3.number_input("Previous SD",
                                  value=float(raw_res.std()), key="sd_c")

        k_val = 0.5 # c3.number_input("Reference value *k*", value=0.5, step=0.1)

        TEa_user_c = c3.number_input("Allowable error (%)",
                                     value=5.0, min_value=0.0, step=0.1,
                                     key="tea_c")

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
        for e in errs:
            _, _, an, mn = run_cusum(
                raw_res, raw_day, lo_tl, hi_tl, transform,
                h, target_mean, prev_sd, k_val, e
            )
            an_list.append(an)
            mn_list.append(mn)
        plot_anped(errs, np.array(an_list), np.array(mn_list),
                   f"CUSUM – ANPed / MNPed vs error rate (±{TEa_user_c} %)")

##############################################################################
st.sidebar.image("./images/QC Constellation icon.png")
st.sidebar.info("*Developed by Hikmet Can Çubukçu, MD, MSc, PhD, EuSpLM*  \n<hikmetcancubukcu@gmail.com>")
