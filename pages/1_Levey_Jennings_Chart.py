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
    with open('./template/template_IQC.xlsx', "rb") as template_file:
        template_byte = template_file.read()
    # download template excel file
    st.download_button(label="Click to Download Template File",
                        data=template_byte,
                        file_name="template_IQC.xlsx",
                        mime='application/octet-stream')
      # upload file
    uploaded_file = st.file_uploader('#### **Upload your .xlsx (Excel) or .csv file:**', type=['csv','xlsx'], accept_multiple_files=False)
    
    def process_file(file):
        # data of analyte selection
        try:
            uploaded_file = pd.read_excel(file)
        except:
            uploaded_file = pd.read_csv(file, sep=None, engine='python')
        analyte_name_box = st.selectbox("**Select IQC result Column**", tuple(uploaded_file.columns))
        analyte_data = uploaded_file[analyte_name_box]
        analyte_data = analyte_data.dropna(axis=0).reset_index()
        analyte_data = analyte_data[analyte_name_box]
        return analyte_data, analyte_name_box

    # column name (data) selection
    if uploaded_file is not None:
        # data of analyte selection
        analyte_data, analyte_name_box = process_file(uploaded_file)
    st.image('./images/QC_Constellation_sidebar.png')
    st.info('*Developed by Hikmet Can Çubukçu, MD, MSc, EuSpLM* <hikmetcancubukcu@gmail.com>')
    

st.markdown("### **:blue[Levey-Jennings Chart]**")
st.markdown(' ')

# Enter the number of rows for the dataframe
number_of_rows = st.number_input('**:blue[Enter Number of Rows of Your Data]**', min_value=3, max_value=999999999999999)

# Initialize an empty dataframe with the specified number of rows
df = pd.DataFrame(
    [{"IQC results": None, "include": True} for _ in range(number_of_rows)]
)

# Use st.data_editor to create an editable dataframe
edited_df = st.data_editor(
    df,
    column_config={
        "IQC results": st.column_config.NumberColumn(
            "IQC results",
            help="IQC results",
            min_value=0,
            max_value=999999999999999999999999999999999999999,
            # step=1,
            format="%g",
        ),
        "include": st.column_config.CheckboxColumn(
            "include",
            help="Select to include",
            default=True,
        ),
    },
    hide_index=True, num_rows="dynamic"
)  # An editable dataframe
edited_df = pd.DataFrame(edited_df)
# data selection
data_select = st.radio("**:blue[Select the data to be plotted]**",
    ["From entered data table","Uploaded data"])

if data_select == "Uploaded data":
    try:
        data = analyte_data
    except NameError as error:
        print("NameError occurred:", error)
        st.error("Data wasn't uploaded")
else:
    edited_df = edited_df[edited_df['include']==True] # select where include == True
    data = edited_df['IQC results']

APC_select = st.radio("**:blue[Source of mean and standard deviation for L-J Control Chart]**",
    ["From the entered/uploaded data","Custom"])
if APC_select == "Custom":
    st.markdown('**:blue[Enter custom mean/target and standard deviation]**')
    col1, col2 = st.columns([1,1])
    mean_input = col1.number_input('**Mean**',step = 0.00001)
    SD_input = col2.number_input('**Standard Deviation**',step = 0.00001)
    st.write(f'Custom mean: {mean_input}, Custom SD: {SD_input} ')
    mean= mean_input
    std_dev = SD_input
else:
    if data_select == "Uploaded data":
        if uploaded_file is not None:
            mean = np.mean(data)
            std_dev = np.std(data)
        else:
            st.error("Data wasn't uploaded")
    else:
        mean = np.mean(data)
        std_dev = np.std(data)

try:
    # Calculate control limits
    upper_limit_3sd = mean + 3 * std_dev
    lower_limit_3sd = mean - 3 * std_dev
    upper_limit_2sd = mean + 2 * std_dev
    lower_limit_2sd = mean - 2 * std_dev
    upper_limit_1sd = mean + 1 * std_dev
    lower_limit_1sd = mean - 1 * std_dev
    
    # Select rules   
    st.markdown('**:blue[Select your IQC rules]**')
    col1, col2, col3, col4 ,col5, col6 = st.columns([1,1,1,1,1,1])
    rule_1_2s = col1.checkbox('**1-2s**')
    rule_1_3s = col2.checkbox('**1-3s**')
    rule_2_2s = col3.checkbox('**2-2s**')
    rule_R_4s = col4.checkbox('**R-4s**')
    rule_4_1s = col5.checkbox('**4-1s**')
    rule_10x = col6.checkbox('**10x**')


    # Create a dataframe for the plotly express function
    df = pd.DataFrame({'Data': data, 'Mean': mean, '+3SD': upper_limit_3sd, '-3SD': lower_limit_3sd,
                    '+2SD': upper_limit_2sd, '-2SD': lower_limit_2sd, 
                    '+1SD':upper_limit_1sd, '-1SD':lower_limit_1sd})

    # Create a Shewhart Chart using Plotly
    fig = go.Figure()

    # Scatter plot for the data points
    fig.add_trace(go.Scatter(x=df.index, y=df['Data'], mode='markers', name='Data'))

    # Line plot for upper and lower control limits
    fig.add_trace(go.Scatter(x=df.index, y=df['+3SD'], mode='lines', line=dict(color='red'), name='+3SD'))
    fig.add_trace(go.Scatter(x=df.index, y=df['-3SD'], mode='lines', line=dict(color='red'), name='-3SD'))
    fig.add_trace(go.Scatter(x=df.index, y=df['+2SD'], mode='lines', line=dict(color='blue'), name='+2SD'))
    fig.add_trace(go.Scatter(x=df.index, y=df['-2SD'], mode='lines', line=dict(color='blue'), name='-2SD'))
    fig.add_trace(go.Scatter(x=df.index, y=df['+1SD'], mode='lines', line=dict(color='lightblue'), name='+1SD'))
    fig.add_trace(go.Scatter(x=df.index, y=df['-1SD'], mode='lines', line=dict(color='lightblue'), name='-1SD'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Mean'], mode='lines', line=dict(color='lightgreen'), name='Mean'))
            
    out_of_control_traces = []
    # Highlight points outside control limits
    if rule_1_3s:
        df['Out of Control 1-3s'] = ((df['Data'] >= upper_limit_3sd) | (df['Data'] <= lower_limit_3sd))
        out_of_control_points_1_3s = df[df['Out of Control 1-3s']]
        fig.add_trace(go.Scatter(x=out_of_control_points_1_3s.index, y=out_of_control_points_1_3s['Data'],
                                mode='markers', marker=dict(color='red'), showlegend=False, 
                                name='1-3s',text='Out of Control (1-3s)'))
            
    if rule_1_2s:
        df['Out of Control 1-2s'] = (df['Data'] >= upper_limit_2sd) | (df['Data'] <= lower_limit_2sd)
        out_of_control_points_1_2s = df[df['Out of Control 1-2s']]
        fig.add_trace(go.Scatter(x=out_of_control_points_1_2s.index, y=out_of_control_points_1_2s['Data'],
                                mode='markers', marker=dict(color='red'), showlegend=False,
                                name='1-2s',text='Out of Control (1-2s)'))

    if rule_2_2s:
        # Identify points outside the 2-SD limit
        df['Out of Control 2-2s'] = ((((df['Data'] >= upper_limit_2sd) & (df['Data'].shift(1) >= upper_limit_2sd)) |
                                    ((df['Data'] >= upper_limit_2sd) & (df['Data'].shift(-1) >= upper_limit_2sd))) |
                                    (((df['Data'] <= lower_limit_2sd) & (df['Data'].shift(1) <= lower_limit_2sd)) |
                                    ((df['Data'] <= lower_limit_2sd) & (df['Data'].shift(-1) <= lower_limit_2sd))))
        # Extract only the points that meet the 2-2s rule
        out_of_control_2_2s_points = df[df['Out of Control 2-2s']]

        # Scatter plot for the out of control points
        fig.add_trace(go.Scatter(x=out_of_control_2_2s_points.index, y=out_of_control_2_2s_points['Data'],
                                mode='markers', marker=dict(color='red'), 
                                showlegend=False, name='2-2s', text='Out of Control (2-2s)'))

    if rule_R_4s:
        df['Out of Control R-4s'] = ((((df['Data'] >= upper_limit_2sd) & (df['Data'].shift(1) <= lower_limit_2sd))|
                                    ((df['Data'] >= upper_limit_2sd) & (df['Data'].shift(-1) <= lower_limit_2sd))) |
                                    (((df['Data'] <= lower_limit_2sd) & (df['Data'].shift(1) >= upper_limit_2sd))|
                                    ((df['Data'] <= lower_limit_2sd) & (df['Data'].shift(-1) >= upper_limit_2sd))))
                
        out_of_control_R_4s_points = df[df['Out of Control R-4s']]
        # Identify points outside
        fig.add_trace(go.Scatter(x=out_of_control_R_4s_points.index, y=out_of_control_R_4s_points['Data'],
                                mode='markers', marker=dict(color='red'),showlegend=False,
                                name='R-4s', text='Out of Control (R-4s)'))

    if rule_4_1s:
        df['Out of Control 4-1s'] = ((((df['Data'] >= upper_limit_1sd) & (df['Data'].shift(1) >= upper_limit_1sd) & (df['Data'].shift(2) >= upper_limit_1sd) & (df['Data'].shift(3) >= upper_limit_1sd)) |
                                    ((df['Data'] >= upper_limit_1sd) & (df['Data'].shift(-1) >= upper_limit_1sd) & (df['Data'].shift(-2) >= upper_limit_1sd) & (df['Data'].shift(-3) >= upper_limit_1sd)) |
                                    ((df['Data'].shift(1) >= upper_limit_1sd) & (df['Data'] >= upper_limit_1sd) & (df['Data'].shift(-1) >= upper_limit_1sd) & (df['Data'].shift(-2) >= upper_limit_1sd)) |
                                ((df['Data'].shift(-1) >= upper_limit_1sd) & (df['Data'] >= upper_limit_1sd) & (df['Data'].shift(1) >= upper_limit_1sd) & (df['Data'].shift(2) >= upper_limit_1sd))) |
                                    (((df['Data'] <= lower_limit_1sd) & (df['Data'].shift(1) <= lower_limit_1sd) & (df['Data'].shift(2) <= lower_limit_1sd) & (df['Data'].shift(3) <= lower_limit_1sd)) |
                                    ((df['Data'].shift(1) <= lower_limit_1sd) & (df['Data'] <= lower_limit_1sd) & (df['Data'].shift(-1) <= lower_limit_1sd) & (df['Data'].shift(-2) <= lower_limit_1sd))|
                                    ((df['Data'] <= lower_limit_1sd) & (df['Data'].shift(-1) <= lower_limit_1sd) & (df['Data'].shift(-2) <= lower_limit_1sd) & (df['Data'].shift(-3) <= lower_limit_1sd)) |
                                    ((df['Data'].shift(-1) <= lower_limit_1sd) & (df['Data'] <= lower_limit_1sd) & (df['Data'].shift(1) <= lower_limit_1sd) & (df['Data'].shift(2) <= lower_limit_1sd))))
                
        # Extract only the points that meet the 4-1s rule
        out_of_control_4_1s_points = df[df['Out of Control 4-1s']]
        fig.add_trace(go.Scatter(x=out_of_control_4_1s_points.index, y=out_of_control_4_1s_points['Data'],
                                mode='markers', marker=dict(color='red'), showlegend=False, 
                                name='4-1s',text='Out of Control (4-1s)'))     

    if rule_10x:
        # Identify points shifted
        df['Out of Control 10x'] = ((((df['Data'] >= mean) & (df['Data'].shift(1) > mean) & (df['Data'].shift(2) > mean) & (df['Data'].shift(3) > mean) & (df['Data'].shift(4) > mean) 
                                    & (df['Data'].shift(5) > mean) & (df['Data'].shift(6) > mean) & (df['Data'].shift(7) > mean)  & (df['Data'].shift(8) > mean)  & (df['Data'].shift(9) > mean)) |
                                    ((df['Data'].shift(-1) > mean) & (df['Data'] > mean) & (df['Data'].shift(1) > mean) & (df['Data'].shift(2) > mean) & (df['Data'].shift(3) > mean) 
                                    & (df['Data'].shift(4) > mean) & (df['Data'].shift(5) > mean) & (df['Data'].shift(6) > mean) & (df['Data'].shift(7) > mean)  & (df['Data'].shift(8) > mean)) |
                                    ((df['Data'].shift(-2) > mean) & (df['Data'].shift(-1) > mean) & (df['Data'] > mean) & (df['Data'].shift(1) > mean) & (df['Data'].shift(2) > mean) 
                                    & (df['Data'].shift(3) > mean) & (df['Data'].shift(4) > mean) & (df['Data'].shift(5) > mean) & (df['Data'].shift(6) > mean) & (df['Data'].shift(7) > mean)) |
                                    ((df['Data'].shift(-3) > mean) & (df['Data'].shift(-2) > mean) & (df['Data'].shift(-1) > mean) & (df['Data'] > mean) & (df['Data'].shift(1) > mean) 
                                    & (df['Data'].shift(2) > mean) & (df['Data'].shift(3) > mean) & (df['Data'].shift(4) > mean) & (df['Data'].shift(5) > mean) & (df['Data'].shift(6) > mean)) |
                                    ((df['Data'].shift(-4) > mean) & (df['Data'].shift(-3) > mean) & (df['Data'].shift(-2) > mean) & (df['Data'].shift(-1) > mean) & (df['Data'] > mean) 
                                    & (df['Data'].shift(1) > mean) & (df['Data'].shift(2) > mean) & (df['Data'].shift(3) > mean) & (df['Data'].shift(4) > mean) & (df['Data'].shift(5) > mean)) |
                                    ((df['Data'].shift(1) > mean) & (df['Data'] > mean) & (df['Data'].shift(-1) > mean) & (df['Data'].shift(-2) > mean) & (df['Data'].shift(-3) > mean) 
                                    & (df['Data'].shift(-4) > mean) & (df['Data'].shift(-5) > mean) & (df['Data'].shift(-6) > mean) & (df['Data'].shift(-7) > mean) & (df['Data'].shift(-8) > mean))|
                                    ((df['Data'].shift(2) > mean) & (df['Data'].shift(1) > mean) & (df['Data'] > mean) & (df['Data'].shift(-1) > mean) & (df['Data'].shift(-2) > mean) 
                                    & (df['Data'].shift(-3) > mean) & (df['Data'].shift(-4) > mean) & (df['Data'].shift(-5) > mean) & (df['Data'].shift(-6) > mean) & (df['Data'].shift(-7) > mean))|
                                    ((df['Data'].shift(3) > mean) & (df['Data'].shift(2) > mean) & (df['Data'].shift(1) > mean) & (df['Data'] > mean) & (df['Data'].shift(-1) > mean) 
                                    & (df['Data'].shift(-2) > mean) & (df['Data'].shift(-3) > mean) & (df['Data'].shift(-4) > mean) & (df['Data'].shift(-5) > mean) & (df['Data'].shift(-6) > mean))|
                                    ((df['Data'].shift(4) > mean) & (df['Data'].shift(3) > mean) & (df['Data'].shift(2) > mean) & (df['Data'].shift(1) > mean) & (df['Data'] > mean) 
                                    & (df['Data'].shift(-1) > mean) & (df['Data'].shift(-2) > mean) & (df['Data'].shift(-3) > mean) & (df['Data'].shift(-4) > mean) & (df['Data'].shift(-5) > mean))|
                                    ((df['Data'] > mean) & (df['Data'].shift(-1) > mean) & (df['Data'].shift(-2) > mean) & (df['Data'].shift(-3) > mean) & (df['Data'].shift(-4) > mean) 
                                    & (df['Data'].shift(-5) > mean) & (df['Data'].shift(-6) > mean) & (df['Data'].shift(-7) > mean) & (df['Data'].shift(-8) > mean) & (df['Data'].shift(-9) > mean))) 
                                    |
                                    (((df['Data'] < mean) & (df['Data'].shift(1) < mean) & (df['Data'].shift(2) < mean) & (df['Data'].shift(3) < mean) & (df['Data'].shift(4) < mean) 
                                    & (df['Data'].shift(5) < mean) & (df['Data'].shift(6) < mean) & (df['Data'].shift(7) < mean)  & (df['Data'].shift(8) < mean)  & (df['Data'].shift(9) < mean)) |
                                    ((df['Data'].shift(-1) < mean) & (df['Data'] < mean) & (df['Data'].shift(1) < mean) & (df['Data'].shift(2) < mean) & (df['Data'].shift(3) < mean) 
                                    & (df['Data'].shift(4) < mean) & (df['Data'].shift(5) < mean) & (df['Data'].shift(6) < mean) & (df['Data'].shift(7) < mean)  & (df['Data'].shift(8) < mean)) |
                                    ((df['Data'].shift(-2) < mean) & (df['Data'].shift(-1) < mean) & (df['Data'] < mean) & (df['Data'].shift(1) < mean) & (df['Data'].shift(2) < mean) 
                                    & (df['Data'].shift(3) < mean) & (df['Data'].shift(4) < mean) & (df['Data'].shift(5) < mean) & (df['Data'].shift(6) < mean) & (df['Data'].shift(7) < mean)) |
                                    ((df['Data'].shift(-3) < mean) & (df['Data'].shift(-2) < mean) & (df['Data'].shift(-1) < mean) & (df['Data'] < mean) & (df['Data'].shift(1) < mean) 
                                    & (df['Data'].shift(2) < mean) & (df['Data'].shift(3) < mean) & (df['Data'].shift(4) < mean) & (df['Data'].shift(5) < mean) & (df['Data'].shift(6) < mean)) |
                                    ((df['Data'].shift(-4) < mean) & (df['Data'].shift(-3) < mean) & (df['Data'].shift(-2) < mean) & (df['Data'].shift(-1) < mean) & (df['Data'] < mean) 
                                    & (df['Data'].shift(1) < mean) & (df['Data'].shift(2) < mean) & (df['Data'].shift(3) < mean) & (df['Data'].shift(4) < mean) & (df['Data'].shift(5) < mean)) |
                                    ((df['Data'].shift(1) < mean) & (df['Data'] < mean) & (df['Data'].shift(-1) < mean) & (df['Data'].shift(-2) < mean) & (df['Data'].shift(-3) < mean) 
                                    & (df['Data'].shift(-4) < mean) & (df['Data'].shift(-5) < mean) & (df['Data'].shift(-6) < mean) & (df['Data'].shift(-7) < mean) & (df['Data'].shift(-8) < mean))|
                                    ((df['Data'].shift(2) < mean) & (df['Data'].shift(1) < mean) & (df['Data'] < mean) & (df['Data'].shift(-1) < mean) & (df['Data'].shift(-2) < mean) 
                                    & (df['Data'].shift(-3) < mean) & (df['Data'].shift(-4) < mean) & (df['Data'].shift(-5) < mean) & (df['Data'].shift(-6) < mean) & (df['Data'].shift(-7) < mean))|
                                    ((df['Data'].shift(3) < mean) & (df['Data'].shift(2) < mean) & (df['Data'].shift(1) < mean) & (df['Data'] < mean) & (df['Data'].shift(-1) < mean) 
                                    & (df['Data'].shift(-2) < mean) & (df['Data'].shift(-3) < mean) & (df['Data'].shift(-4) < mean) & (df['Data'].shift(-5) < mean) & (df['Data'].shift(-6) < mean))|
                                    ((df['Data'].shift(4) < mean) & (df['Data'].shift(3) < mean) & (df['Data'].shift(2) < mean) & (df['Data'].shift(1) < mean) & (df['Data'] < mean) 
                                    & (df['Data'].shift(-1) < mean) & (df['Data'].shift(-2) < mean) & (df['Data'].shift(-3) < mean) & (df['Data'].shift(-4) < mean) & (df['Data'].shift(-5) < mean))|
                                    ((df['Data'] < mean) & (df['Data'].shift(-1) < mean) & (df['Data'].shift(-2) < mean) & (df['Data'].shift(-3) < mean) & (df['Data'].shift(-4) < mean) 
                                    & (df['Data'].shift(-5) < mean) & (df['Data'].shift(-6) < mean) & (df['Data'].shift(-7) < mean) & (df['Data'].shift(-8) < mean) & (df['Data'].shift(-9) < mean))))

        # Extract only the points that meet the 10x rule
        out_of_control_10x_points = df[df['Out of Control 10x']]
        fig.add_trace(go.Scatter(x=out_of_control_10x_points.index, y=out_of_control_10x_points['Data'],
                                mode='markers', marker=dict(color='red'), showlegend=False, 
                                name='10x',text='Out of Control (10x)'))

            # Layout settings
    fig.update_layout(title='Levey-Jennings Control Chart',
                xaxis_title='Data Point',
                yaxis_title='Value',
                showlegend=True, title_font=dict(color='#cc0000'))

    # Show the plot
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    
    st.write("---")        
       
    

    # show dataframe with out-of-control results notation
    with st.expander("**:blue[See the details of your data & download your data as .csv file]**"):
        st.dataframe(df)

    if not math.isnan(mean):
        st.markdown("**:blue[Analytical Performance Characteristics (Mean, standard deviation, and CV) of the data]**")
        def round_half_up(n, decimals=0):
            multiplier = 10**decimals
            return math.floor(n * multiplier + 0.5) / multiplier
        try:
            st.markdown(f"""
                        | *:green[Analytical Performance Characteristics]* | *:green[Value]* |
                        | ----------- | ----------- |
                        | **Mean** | **{round_half_up(mean,2)}** |
                        | **Standard Deviation** | **{round_half_up(std_dev,2)}** |
                        | **Coefficient of Variation (CV)** | **{round_half_up((std_dev*100/mean),2)}** |
                        """)
        except ZeroDivisionError as ze:
            st.error('Mean value can not be "0"') 
except NameError as ne:
    if 'data' in str(ne):
        st.info("Please upload your data")
    else:
        # Handle other NameError cases if needed
        print("A NameError occurred, but it's not related to 'data'")
