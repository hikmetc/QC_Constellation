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
    

st.markdown("### **:blue[Moving Averages Charts for Internal Quality Control]**")
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

APC_select = st.radio("**:blue[Source of mean and standard deviation for moving averages charts]**",
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
   

    # Create a dataframe for the plotly express function
    df = pd.DataFrame({'Data': data, 'Mean': mean, '+3SD': upper_limit_3sd, '-3SD': lower_limit_3sd,
                    '+2SD': upper_limit_2sd, '-2SD': lower_limit_2sd, 
                    '+1SD':upper_limit_1sd, '-1SD':lower_limit_1sd})

    # EWMA PLOT
    lambda_value_choice = st.select_slider('**:blue[Select the lambda value (weighting factor) for EWMA chart]**',
                options=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1], value = 0.2)
    
    lambda_value = lambda_value_choice
    if lambda_value == 0.05:
        L = 2.615
    elif lambda_value == 0.1:
        L = 2.814
    elif lambda_value == 0.2:
        L = 2.962
    elif lambda_value == 0.3:
        L = 3.023
    elif lambda_value == 0.4:
        L = 3.054
    elif lambda_value == 0.5:
        L = 3.071
    elif lambda_value == 0.75:
        L = 3.087
    elif lambda_value == 1:
        L = 3.090

    try:
    # Calculate Exponential Weighted Moving Average (EWMA)
        ewma = df['Data'].ewm(alpha= lambda_value, span=None, adjust=False).mean()
    except Exception as e:
        st.error("Your data contains inappropriate type of values. Please check your data.")

    # Calculate UCL and LCL
    results = range(1, len(ewma) + 1)
    UCL_values = []
    LCL_values = []

    for ind in results:
        UCL = mean + L * std_dev * (((lambda_value) * (1 - (1 - lambda_value)**(2 * ind)) / (2 - lambda_value))**(0.5))
        LCL = mean - L * std_dev * (((lambda_value) * (1 - (1 - lambda_value)**(2 * ind)) / (2 - lambda_value))**(0.5))
        
        UCL_values.append(UCL)
        LCL_values.append(LCL)

    # Create a Plotly figure
    fig2 = go.Figure()
    
    # Add EWMA data
    fig2.add_trace(go.Scatter(x=ewma.index, y=ewma, mode='lines', name='EWMA'))
    
    # Add markers for points above UCL
    fig2.add_trace(go.Scatter(x=ewma[ewma > UCL_values].index, y=ewma[ewma >= UCL_values], mode='markers',
                            marker=dict(color='red'), name='Above UCL'))

    # Add markers for points below LCL
    fig2.add_trace(go.Scatter(x=ewma[ewma < LCL_values].index, y=ewma[ewma <= LCL_values], mode='markers',
                            marker=dict(color='blue'), name='Below LCL'))


    # Add UCL and LCL
    fig2.add_trace(go.Scatter(x=ewma.index, y=UCL_values, mode='lines', name='UCL', line=dict(color='red')))
    fig2.add_trace(go.Scatter(x=ewma.index, y=LCL_values, mode='lines', name='LCL', line=dict(color='blue')))

    # Customize the layout
    fig2.update_layout(title=f'Exponentially Weighted Moving Average (EWMA) chart with weighting factor "{lambda_value}"',
                    xaxis_title='Data point',
                    yaxis_title='Value', title_font=dict(color='#cc0000'))
    
    st.plotly_chart(fig2, theme="streamlit", use_container_width=True) 
    df[f'EWMA (lambda={lambda_value}) higher than UCL'] = (ewma >= UCL_values)      
    df[f'EWMA (lambda={lambda_value}) lower than LCL'] = (ewma <= LCL_values)      

    st.write("---")
    
    # CUSUM PLOT
    def plot_cusum(cusum_np_arr, mu, sd, k=0.5, h=5):
        # Drop rows with None values in 'Data' column
        cusum_np_arr = cusum_np_arr.dropna().reset_index(drop=True)
        
        Cp = (cusum_np_arr * 0).copy()
        Cm = Cp.copy()

        for ii in np.arange(len(cusum_np_arr)):
            if ii == 0:
                Cp[ii] = 0
                Cm[ii] = 0
            else:
                Cp[ii] = np.max([0, ((cusum_np_arr[ii] - mu) / sd) - k + Cp[ii - 1]])
                Cm[ii] = np.max([0, -k - ((cusum_np_arr[ii] - mu) / sd) + Cm[ii - 1]])

        Cont_limit_arr = np.array(h * np.ones((len(cusum_np_arr), 1)))
        Cont_lim_df = pd.DataFrame(Cont_limit_arr, columns=["h"])
        cusum_df = pd.DataFrame({'Cp': Cp, 'Cn': Cm})

        # Create figure
        fig = go.Figure()

        # Add trace for Cp and Cn
        fig.add_trace(go.Scatter(x=np.arange(len(cusum_np_arr)), y=cusum_df['Cp'], mode='lines', name='Cp'))
        fig.add_trace(go.Scatter(x=np.arange(len(cusum_np_arr)), y=-cusum_df['Cn'], mode='lines', name='Cn'))

        # Add trace for Cont_limit
        fig.add_trace(go.Scatter(x=np.arange(len(cusum_np_arr)), y=Cont_lim_df['h'], mode='lines', name='UCL', line=dict(color='red')))

        # Add trace for Cont_limit
        fig.add_trace(go.Scatter(x=np.arange(len(cusum_np_arr)), y=-Cont_lim_df['h'], mode='lines', name='LCL', line=dict(color='blue')))

        # Add markers for points above UCL
        fig.add_trace(go.Scatter(x=cusum_df[cusum_df['Cp'] > h].index, y=cusum_df['Cp'][cusum_df['Cp'] > h],
                                mode='markers', marker=dict(color='red'), name='Above UCL'))

        # Add markers for points below LCL
        fig.add_trace(go.Scatter(x=cusum_df[-cusum_df['Cn'] < -h].index, y=-cusum_df['Cn'][-cusum_df['Cn'] < -h],
                                mode='markers', marker=dict(color='blue'), name='Below LCL'))

        # Update layout
        fig.update_layout(
            title="CUSUM Control Chart",
            xaxis_title="Data points",
            yaxis_title="Value",
            #legend=dict(x=0, y=1),
            showlegend=True,title_font=dict(color='#cc0000')
        )

        # Show figure
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    plot_cusum(df['Data'], mean, std_dev)
    
    # This part add cusum results to the dataframe
    cusum_np_arr = df['Data'].dropna().reset_index(drop=True)
    k=0.5
    h=5    
    mu = mean
    sd = std_dev
    Cp = (cusum_np_arr * 0).copy()
    Cm = Cp.copy()

    for ii in np.arange(len(cusum_np_arr)):
        if ii == 0:
            Cp[ii] = 0
            Cm[ii] = 0
        else:
            Cp[ii] = np.max([0, ((cusum_np_arr[ii] - mu) / sd) - k + Cp[ii - 1]])
            Cm[ii] = np.max([0, -k - ((cusum_np_arr[ii] - mu) / sd) + Cm[ii - 1]])

    Cont_limit_arr = np.array(h * np.ones((len(cusum_np_arr), 1)))
    Cont_lim_df = pd.DataFrame(Cont_limit_arr, columns=["h"])
    cusum_df = pd.DataFrame({'Cp': Cp, 'Cn': Cm})

    df[f'CUSUM higher than UCL'] = (Cp >= h)      
    df[f'CUSUM lower than LCL'] = (Cm >= h)      

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
