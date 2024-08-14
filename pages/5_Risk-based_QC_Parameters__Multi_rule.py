# Developed by Hikmet Can Çubukçu

import streamlit as st
st.set_page_config(layout="wide", page_title="QC Constellation", page_icon="📈")
from datetime import datetime
import math
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as sp
from scipy.stats import norm

with st.sidebar:
    st.image('./images/QC_Constellation_sidebar.png')
    st.info('*Developed by Hikmet Can Çubukçu, MD, MSc, EuSpLM* <hikmetcancubukcu@gmail.com>')
    
st.markdown("### **:blue[Risk-based QC Parameters: Multi Rule Scheme]**")
st.write(" ")
def round_half_up(n, decimals=0):
    multiplier = 10**decimals
    return math.floor(n * multiplier + 0.5) / multiplier

col1, col2 = st.columns([1,1])

# Options for the multiselect
options = ['2-2s', '2-2s/4-1s/R-4s' ,'1-3s/2-2s', '1-3s/2-2s/R-4s', '1-3s/2-2s/4-1s', '1-3s/2-2s/4-1s/R-4s',
                     '1-3s/2-2s/4-1s/R-4s/6x', '1-3s/2-2s/4-1s/R-4s/8x', '1-3s/2-2s/4-1s/R-4s/10x']
# Default selection
default_selection = options.index('1-3s/2-2s')
# Get user selection
selected_qc_rules = col1.selectbox('**:blue[Select/Add QC rules]**', options, index=default_selection)
# Check if the selection is empty
if not selected_qc_rules:
    # Set to default selection if empty and display a warning
    st.error('**At least one QC rule must be selected. Resetting to default single rule {default_selection}.**')
    selected_qc_rules = default_selection

# number of QC measurement
number_of_QCs = col2.number_input('**:blue[Number of QC observation]**', min_value=1 , max_value=6 , step = 1)
# selected multirule with n for selection appropriate data column
selected_multirule_with_n = f'{selected_qc_rules} (N:{number_of_QCs}) Pred'


# Sigma-metric calculation
col11, col22, col33 = st.columns([1,1,1])
total_error_allowable_TEa = col11.number_input('**:blue[TEa (%) or MAU (%)]**', min_value=0.0, max_value=33.0, step = 0.01)
imprecision_input = col22.number_input('**:blue[Imprecision (%CV)]**', min_value=0.001, max_value=33.0, value = 1.0, step = 0.001)
bias_input = col33.number_input('**:blue[Bias (%)]**', min_value=0.0, max_value=33.0, step = 0.01)
sigma_value = (total_error_allowable_TEa-bias_input)/imprecision_input
sigma_value_without_bias = (total_error_allowable_TEa)/imprecision_input

# E(NB) refers to the expected number of patient samples tested between QC events (practical run size).
e_NB = st.number_input('**:blue[Expected number of patient samples between planned QCs (E(NB))]**', min_value=1 , max_value=20000, value = 50 , step = 1)

# severity of harm category
severity_options = ['Negligible', 'Minor', 'Serious', 'Critical', 'Catastrophic']
severity_of_harm_input = st.selectbox('**:blue[Severity of harm category of the measurand]**', severity_options, index=0)



# severity of harm scores
severity_of_harm_scores_dict = {
  "Negligible": 1,
  "Minor": 2,
  "Serious": 3,
  "Critical": 4,
  "Catastrophic": 5
}
severity_of_harm_score = severity_of_harm_scores_dict[severity_of_harm_input]

# Patient risk factor if not custom , the value is calculated as 6 - severity of harm score
patient_risk_factor_choice = st.checkbox(':blue[Calculate MRS based on custom patient risk factor]')

if patient_risk_factor_choice:
    patient_risk_factor_input  = st.number_input('**:blue[Custom patient risk factor]**', min_value=1, max_value=5, value = 1, step = 1)
else:
    patient_risk_factor_input = 6-severity_of_harm_score

bias_as_multiples_of_CV = bias_input/imprecision_input

multiples_of_TEa = [-2.00, -1.95, -1.90, -1.85, -1.80, -1.75, -1.70, -1.65, -1.60, -1.55,
-1.50, -1.45, -1.40, -1.35, -1.30, -1.25, -1.20, -1.15, -1.10, -1.05,
-1.00, -0.95, -0.90, -0.85, -0.80, -0.75, -0.70, -0.65, -0.60, -0.55,
-0.50, -0.45, -0.40, -0.35, -0.30, -0.25, -0.20, -0.15, -0.10, -0.05,
    0.00,  0.05,  0.10,  0.15,  0.20,  0.25,  0.30,  0.35,  0.40,  0.45,
    0.50,  0.55,  0.60,  0.65,  0.70,  0.75,  0.80,  0.85,  0.90,  0.95,
    1.00,  1.05,  1.10,  1.15,  1.20,  1.25,  1.30,  1.35,  1.40,  1.45,
    1.50,  1.55,  1.60,  1.65,  1.70,  1.75,  1.80,  1.85,  1.90,  1.95,
    2.00]
occurance_of_error_as_percentages = [x * total_error_allowable_TEa for x in multiples_of_TEa]
occurance_of_error_as_SDs = [x/imprecision_input for x in occurance_of_error_as_percentages]


# read multi rule data
def load_data(path):
    df = pd.read_excel(path)
    return df
Ped_multirule_data = load_data('./multirule_data/merged_data_multirule_interpolated.xlsx')



# Multirule Adjustment conventional rounding of SD to 1 decimal containing numerals
occurance_of_error_as_SDs_rounded = [round_half_up(x, 2) for x in occurance_of_error_as_SDs]



# selection of Ped of selected multirule and N
Ped_multirule_data_based_on_selected_rule = Ped_multirule_data[['error_degree', selected_multirule_with_n]]

# Probability of error detection for QC rules after an error occurs (P1) list
Ped_for_QC_rule_P1 = [value for x in occurance_of_error_as_SDs_rounded 
                      for value in Ped_multirule_data_based_on_selected_rule[Ped_multirule_data_based_on_selected_rule['error_degree'] == abs(x)][selected_multirule_with_n]]

# Calculate combined Pfr 
probability_of_false_rejection_Pfr_for_QC_rules = Ped_for_QC_rule_P1[40]

# ARLed calculation 1 / P1 , ARLED is the average run length to error detection
ARLed = [1/x for x in Ped_for_QC_rule_P1]

# eNB list
e_NB_list = [e_NB]*len(multiples_of_TEa)

# E(N0) is the expected number of patient test results generated between the time an out-of-control condition occurs and the next QC event.
e_N0 = e_NB/2
e_N0_list = [e_N0] * len(multiples_of_TEa) # E(N0) list with size of multiples_of_TEa

# Mean number of patient samples processed from the start of the out-of-control condition to QC detection E(NP)
# the e_Np formula as a function
def e_Np_calculate_formula(e_NB, Ped):
    return e_NB / 2 + e_NB * (1 / Ped - 1)
# Applying the formula to each value in Ped_for_QC_rule_P1
e_Np_list = [e_Np_calculate_formula(e_NB, x) for x in Ped_for_QC_rule_P1]

# Calculate the Probability of exceeding TEa fraction when in control using the standard normal distribution CDF
p_TEa_in_control = norm.cdf(-sigma_value_without_bias - bias_as_multiples_of_CV) + (1 - norm.cdf(sigma_value_without_bias - bias_as_multiples_of_CV))
p_TEa_in_control_list = [p_TEa_in_control] * len(multiples_of_TEa)

# Probability of exceeding the TEa fraction when error exists
# Define the formula as a function
def probability_exceeding_TEa_inerror_calculate_formula(sigma_value_without_bias, bias_as_multiples_of_CV, error_SD):
    return norm.cdf(-(sigma_value_without_bias + bias_as_multiples_of_CV + error_SD)) + (1 - norm.cdf(sigma_value_without_bias - bias_as_multiples_of_CV - error_SD))

# Applying the formula to each value in occurance_of_error_as_SDs
probability_exceeding_TEa_inerror_list = [probability_exceeding_TEa_inerror_calculate_formula(sigma_value_without_bias, bias_as_multiples_of_CV, x) for x in occurance_of_error_as_SDs]

# Incremental value of the probability of generating a result that exceeds the TEa range
# ΔPE: represents the increase in the probability of producing unacceptable patient outcomes due to the presence of an out-of-control error condition.
# Calculate the difference between the two lists element-wise
delta_PE_list = [a - b for a, b in zip(probability_exceeding_TEa_inerror_list, p_TEa_in_control_list)]

# total expected number of unreliable results [E (Nu)]
e_NU__list = [e_Nq * (exceeding_TEa_inerror - TEa_in_control) for e_Nq, exceeding_TEa_inerror, TEa_in_control in zip(e_Np_list, probability_exceeding_TEa_inerror_list, p_TEa_in_control_list)]

# Percentage of unreliable results % (UnR%)
unreliable_results_percentage_list = [e_NU*100/e_Np for e_NU, e_Np in zip(e_NU__list, e_Np_list)]

# E(Nuf): the Expected Number of unreliable final patient results
e_Nuf_list = [dif*((ARL_ed-1)*e_NB_list_e-(1-Ped_P1)*(e_NB_list_e-e_N0_list_e)) for dif, ARL_ed, Ped_P1, e_N0_list_e, e_NB_list_e in zip(delta_PE_list, ARLed, Ped_for_QC_rule_P1, e_N0_list, e_NB_list)] 

# E(Nuc) Number of correctable unreliable results
e_Nuc_list = [(Ped_for_QC_rule_P1_f*e_N0_list_f+(1-Ped_for_QC_rule_P1_f)*e_NB_list_f)*delta_PE_list_f for Ped_for_QC_rule_P1_f, e_N0_list_f, e_NB_list_f, delta_PE_list_f in zip(Ped_for_QC_rule_P1, e_N0_list, e_NB_list, delta_PE_list)]

# Max E(Nuf) 
max_eNuf_rounded = round(max(e_Nuf_list), 2)
max_eNuf = max(e_Nuf_list)

# index of Max E(Nuf)
index_of_max_enuf = e_Nuf_list.index(max_eNuf) 

# Systematic error at Max(Enuf)
systematic_error_at_MaxENuf = occurance_of_error_as_percentages[index_of_max_enuf]





# PLOTS
# PLOT 1
# occurance_of_error_as_percentages on X axis,  and line plots of e_Nuf_list, e_Nuc_list
# Create subplots with shared X-axis
fig111 = sp.make_subplots(specs=[[{"secondary_y": True}]])

# Add a line plot for e_Nuf_list
fig111.add_trace(go.Scatter(x=occurance_of_error_as_percentages, y=e_Nuf_list, mode='lines', name='E(Nuf)'))
# Add a line plot for e_Nuc_list
fig111.add_trace(go.Scatter(x=occurance_of_error_as_percentages, y=e_Nuc_list, mode='lines', name='E(Nuc)'), secondary_y=True)

# Generate 5 ticks for each y-axis
left_y_ticks_1 = np.linspace(start=math.floor(min(e_Nuc_list)), stop=max(e_Nuc_list), num=5)
right_y_ticks_1 = np.linspace(start=math.floor(min(e_Nuf_list)), stop=max(e_Nuf_list), num=5)
# Update the y-axes with the generated ticks
fig111.update_yaxes(title_text = 'E(Nuf)', tickvals=right_y_ticks_1, secondary_y=False)
fig111.update_yaxes(title_text = 'E(Nuc)',tickvals=left_y_ticks_1, secondary_y=True)
# Set the title and labels
fig111.update_layout(
    title='Line Plot of e_Nuf_list and e_Nuc_list vs. occurance_of_error_as_percentages',
    xaxis_title='Systematic Error (%)')

# Set title color
fig111.update_layout(title=dict(text='E(Nuf) value vs Systematic Error(%) Plot', font=dict(color='#cc0000')))
# Show the plot
st.plotly_chart(fig111, use_container_width=True)
st.markdown("""**E(Nuf):** The expected number of unreliable final results produced before the last accepted QC. 
            **E(Nuc):** Expected number of unreliable correctable patient results achieved between the last 
            accepted QC event and the QC rule rejection.""")

# PLOT 2
# expected number of QC events to detect a systematic error E(QCE) AKA # ARLed calculation 1 / P1 , ARLED is the average run length to error detection
# Create a Plotly figure   
# Create subplots with shared X-axis
fig114 = sp.make_subplots(specs=[[{"secondary_y": True}]])

# Add a line plot for E(QCE) on the left Y-axis
fig114.add_trace(go.Scatter(x=occurance_of_error_as_percentages, y=ARLed, mode='lines', name='E(QCE)'))

# Add a line plot for U(nR)% on the right Y-axis
fig114.add_trace(go.Scatter(x=occurance_of_error_as_percentages, y=unreliable_results_percentage_list, mode='lines', name='UnR (%)'), secondary_y=True)

# Set the title and labels
fig114.update_layout(
    title='Line Plot of E(QCE) and U(nR)% vs. occurance_of_error_as_percentages',
    xaxis_title='Systematic Error (%)')

# Generate 5 ticks for each y-axis
left_y_ticks = np.linspace(start=min(ARLed), stop=max(ARLed), num=5)
right_y_ticks = np.linspace(start=math.floor(min(unreliable_results_percentage_list)), stop=max(unreliable_results_percentage_list), num=5)

# Update the y-axes with the generated ticks
fig114.update_yaxes(title_text='E(QCE)', tickvals=left_y_ticks, secondary_y=False)
fig114.update_yaxes(title_text='UnR (%)',tickvals=right_y_ticks, secondary_y=True)

# Set title color
fig114.update_layout(title=dict(text='E(QCE) value and U(nR)% vs. Systematic Error(%) Plot', font=dict(color='#cc0000')))

# Show the plot
st.plotly_chart(fig114, use_container_width=True)
st.markdown("**E(QCE):** Expected number of QC events to detect a systematic error, **UnR (%):** Percentage of unreliable results.")

# Calculate Max Run Size
max_run_size_result = (e_NB / max_eNuf) * (patient_risk_factor_input)

# Apply the conditions
if max_run_size_result > 10000:
    max_run_size_result = ">10000"
elif max_run_size_result < 1:
    max_run_size_result = "<1"
else:
    max_run_size_result = math.floor(max_run_size_result)

st.markdown(" ")
st.markdown("##### **:blue[Risk based QC Parameters]**")

st.markdown(f"""   
                    | *:green[Parameter]* | *:green[Value]* |
                    | ----------- | ----------- |
                    | **Probability of false rejection (%)** | **{round_half_up(probability_of_false_rejection_Pfr_for_QC_rules*100,2)}** |
                    | **Sigma-metric value** | **{round_half_up(sigma_value,2)}** |
                    | **Systematic error (%) at MaxE(Nuf)** | **{abs(round_half_up(systematic_error_at_MaxENuf,2))}** |
                    | **MaxE(Nuf)** | **{round_half_up(max_eNuf,2)}** |
                    | **Max run size** | **{max_run_size_result}** |
                        
                """)


st.write(" ")

st.markdown("##### **:blue[Recommendations]**")
if max_eNuf >= patient_risk_factor_input:
    st.error(f'**Unacceptable MaxE(Nuf). Reduce number of patient samples between planned QCs to {max_run_size_result} to achieve acceptable MaxE(Nuf). You may also try another multirule with higher number of QC measurement**')
else:
    if type(max_run_size_result) is str or e_NB < max_run_size_result:
        st.info(f'**Number of patient samples between planned QCs are plausible to reach acceptable MaxE(Nuf). You can increase number of patient samples between planned QCs up to {max_run_size_result}**')
    else:
        st.info(f'**Number of patient samples between planned QCs are plausible to reach acceptable MaxE(Nuf).**')

if probability_of_false_rejection_Pfr_for_QC_rules > 0.5 and number_of_QCs > 1:
    st.error(f'**Unnacceptable Pfr ({round_half_up(probability_of_false_rejection_Pfr_for_QC_rules*100,2)}%) Try to another multirule or reduce number of QC measurements to achieve acceptable Pfr (<5%)**')
elif probability_of_false_rejection_Pfr_for_QC_rules > 0.5 and number_of_QCs == 1:
    st.error("**Try another multirule to reduce Pfr to acceptable levels (<5%)**")
else:
    st.success("**Pfr values are within acceptable range (<5%)**")

st.markdown("---")



# Risk Management Index Section
st.markdown("##### **:blue[Risk Management Index]**")
# input probability of harm given an unacceptable result
prob_conditional_patient_harm = st.number_input('**:blue[Probability of harm given an unacceptable result (%)]**', help = 'Enter the probability (%) that an erroneous test result will lead to an inappropriate medical decision or action that result in patient harm',min_value = 0, max_value= 100, value=5) # input probability of harm given an unacceptable result
prob_conditional_patient_harm_as_fraction = prob_conditional_patient_harm/100
acceptable_prob_harm_dict = {
  "Negligible": 0.01,
  "Minor": 0.001,
  "Serious": 0.0001,
  "Critical": 0.00001,
  "Catastrophic": 0.000001
}
acceptable_prob_harm = acceptable_prob_harm_dict[severity_of_harm_input]

# Prob harm and related calculations
# mean number of days between failures
number_of_days_bw_system_failure = st.number_input('**:blue[The mean number of days between instrument failures]**',min_value = 1, max_value= 730, value=90) # input probability of harm given an unacceptable result
# mean number of patient samples per day
number_of_samples_each_day = st.number_input('**:blue[The mean number of patients tested per day]**',min_value = 1, max_value= 100000, value=100) # input probability of harm given an unacceptable result
#mean patients between failures
MPBF = number_of_days_bw_system_failure*number_of_samples_each_day
# probability of erroneous results
prob_erroneous_results = [pe0 + enuf/(MPBF+anped) for pe0, enuf, anped in zip(p_TEa_in_control_list, e_Nuf_list, ARLed)]
# probality of harm
prob_harm = [pe*prob_conditional_patient_harm_as_fraction for pe in prob_erroneous_results]
max_prob_harm = max(prob_harm) # max prob harm
index_of_max_prob_harm = prob_harm.index(max_prob_harm)
SE_at_max_prob_harm = occurance_of_error_as_percentages[index_of_max_prob_harm]
# Risk management index
RMI = np.mean(prob_harm)/acceptable_prob_harm

st.markdown(f"""   
                    | *:green[Parameter]* | *:green[Value]* |
                    | ----------- | ----------- |
                    | **Probability of harm** | **{format(np.mean(prob_harm), '.12f')}** |
                    | **Acceptable probability of harm** | **{format(acceptable_prob_harm, '.6f')}** |
                    | **Risk management index** | **{format(RMI, '.6f')}** |
                    | **Systematic error (%) at Max RMI** | **{round_half_up(SE_at_max_prob_harm,2)}** |
                    | **Systematic error (as multiples of SD) at Max RMI** | **{round_half_up(SE_at_max_prob_harm,2)/imprecision_input}** |                        
                """)
st.write(" ")
if RMI > 1:
    st.warning(f'**Risk management index ({round_half_up(RMI,5)}) is higher than 1**')
else:
    st.success(f'**Risk management index ({round_half_up(RMI,5)}) is in the acceptable range (≤1)**')

st.markdown("---")





# Further QC parameters
st.markdown("##### **:blue[Further QC Parameters based on Systematic error at MaxE(Nuf)]**")

# Assign SE at Max Enuf 
SE_of_concern_maxenuf = round_half_up(abs(systematic_error_at_MaxENuf/imprecision_input),2)

custom_se_choice = st.checkbox('**:blue[Calculate based on custom systematic error]**')

if custom_se_choice:
    SE_of_concern  = st.number_input('**:blue[Systematic error (as SD) of concern]**', min_value=0.0, max_value=33.0, value = SE_of_concern_maxenuf, step = 0.01)
else:
    SE_of_concern = SE_of_concern_maxenuf


# Probability of error detection for QC rules after an error occurs (P1) list
Ped_P1_value_draft = Ped_multirule_data_based_on_selected_rule[Ped_multirule_data_based_on_selected_rule['error_degree'] == SE_of_concern][selected_multirule_with_n].iloc[0]

# Ped value corresponds to index of SE
Ped_P1_value = Ped_P1_value_draft

# expected number of QC events to detect a systematic error E(QCE) AKA ARLed
e_QCE = 1/Ped_P1_value #index of

# E(N0)
e_N0 = e_NB/2

# Mean number of patient samples processed from the start of the out-of-control condition to QC detection E(NP)
e_Nq = e_NB/2 + e_NB*(1/Ped_P1_value-1)

# Probability of exceeding TEa fraction when in control
p_TEa_exceed_in_control = norm.cdf(-sigma_value_without_bias - bias_as_multiples_of_CV) + (1 - norm.cdf(sigma_value_without_bias - bias_as_multiples_of_CV))

# Probability of exceeding the TEa fraction when error exists
p_TEa_exceed_out_control = norm.cdf(-sigma_value_without_bias - bias_as_multiples_of_CV - SE_of_concern) + (1 - norm.cdf(sigma_value_without_bias - bias_as_multiples_of_CV - SE_of_concern))

# Incremental value of the probability of generating a result that exceeds the TEa range
delta_p_exceed_TEa = p_TEa_exceed_out_control - p_TEa_exceed_in_control

# Number of unreliable results produced E(NU)
e_NU_sv = e_Nq*delta_p_exceed_TEa

# Percentage of unreliable results % (UnR%)
UnR_percentage = e_NU_sv/e_Nq*100

# Number of unreliable results for which reports have been issued E(Nuf)
e_Nuf_sv = delta_p_exceed_TEa*((e_QCE-1)*e_NB-(1-Ped_P1_value)*(e_NB-e_N0))

# Number of correctable unreliable results E(Nuc)
e_Nuc_sv = (Ped_P1_value*e_N0+(1-Ped_P1_value)*e_NB)*delta_p_exceed_TEa

st.markdown(f"""   
                    | *:green[Parameter]* | *:green[Value]* |
                    | ----------- | ----------- |
                    | **Probability of error detection** | **{round(Ped_P1_value, 3)}** |
                    | **Expected number of QC events to detect a systematic error** | **{round(e_QCE, 2)}** |
                    | **Average expected number of patient samples processed from the onset of the out-of-control condition to QC detection** | **{round(e_Nq)}** |
                    | **Probability of generating a result that exceeds the systematic error of {SE_of_concern} (as multiples of SD) during in-control state** | **{ '<0.000001' if p_TEa_exceed_in_control < 0.000001 else format(p_TEa_exceed_in_control, '.6f')}** |
                    | **Probability of generating a result that exceeds the systematic error of {SE_of_concern} (as multiples of SD) when an error exists** | **{round(p_TEa_exceed_out_control, 4)}** |
                    | **Increase in the probability of generating a result that exceeds the systematic error of {SE_of_concern} due to the out-of-control condition** | **{round(delta_p_exceed_TEa, 4)}** |
                    | **Expected number of unreliable results (E(Nu))** | **{round(e_NU_sv, 2)}** |
                    | **Percentage of unreliable results % (UnR%)** | **{round(UnR_percentage, 2)}** |
                    | **Number of unreliable results produced between the onset of the out-of-control condition and the last accepted QC event (E(Nuf))** | **{round(e_Nuf_sv, 2)}** |
                    | **Number of correctable unreliable results between the last accepted QC event and the QC event detected the out-of-control condition (E(Nuc))** | **{round(e_Nuc_sv, 2)}** |
                        
                """)


# POWER GRAPH
SE_list = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
sigma_list = [se + 1.65 for se in SE_list]
ped_list = []

combined_Ped_se = Ped_for_QC_rule_P1
combined_Pfr_se = Ped_for_QC_rule_P1[40]
    
# ped_list for power graph
ped_list = combined_Ped_se



# POWER CURVE
# Prepare the higher X-axis data
# Prepare the higher X-axis data
sigma_values = [x + 1.65 for x in occurance_of_error_as_SDs_rounded]

# Creating the plot with secondary x-axis
fig223 = sp.make_subplots(specs=[[{"secondary_y": False}]])

# Adding SE_list vs ped_list trace
fig223.add_trace(
    go.Scatter(x=occurance_of_error_as_SDs_rounded, y=Ped_for_QC_rule_P1, name="SE values", mode='lines+markers')
)

# Generate tick values for the x-axis at 0.5 intervals
max_x_value = max(occurance_of_error_as_SDs_rounded)
x_ticks = [x * 0.5 for x in range(int((max_x_value + 0.5) * 2))]

# Setting up the primary x-axis (lower x-axis for SE_list)
fig223.update_xaxes(title_text="Systematic error as multiples of SD", tickvals=x_ticks, row=1, col=1)

# Adding and setting up the secondary x-axis (upper x-axis for sigma_list)
fig223.update_layout(
    xaxis2=dict(
        title="Sigma values",
        overlaying='x',
        side='top',
        anchor="free",
        position=1,  # Adjust position as needed
        tickvals=Ped_for_QC_rule_P1,
        ticktext=sigma_values
    ),
    yaxis_title="Ped", showlegend=False
)

# Limit the X-axis range
fig223.update_xaxes(range=[0, max_x_value])

# Adding a vertical line at the specified occurance_of_SE_input_value
fig223.add_vline(
    x=SE_of_concern,
    line_width=2,
    line_dash="dash",
    line_color="red",
    annotation_text=f"Systematic error of concern: {SE_of_concern}",  # Add annotation with value
    annotation_position="top",
)
fig223.update_layout(title=dict(text='Power Function Plot', font=dict(color='#cc0000')))

st.plotly_chart(fig223, use_container_width=True)

