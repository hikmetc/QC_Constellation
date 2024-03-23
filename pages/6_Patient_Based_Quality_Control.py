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
from scipy import stats


with st.sidebar:
    with open('./template/Na_normal_20_pbqc_plot.xlsx', "rb") as template_file:
        template_byte = template_file.read()
    # download template excel file
    st.download_button(label="Click to Download Template File",
                        data=template_byte,
                        file_name="Na_normal_20_pbqc_plot.xlsx",
                        mime='application/octet-stream')
      # upload file
    uploaded_file = st.file_uploader('#### **Upload your .xlsx (Excel) or .csv file:**', type=['csv','xlsx'], accept_multiple_files=False)
    
    @st.cache_data(experimental_allow_widgets=True)
    def process_file(file):
        # data of analyte selection
        try:
            uploaded_file = pd.read_excel(file)
        except:
            uploaded_file = pd.read_csv(file, sep=None, engine='python')
        analyte_name_box = st.selectbox("**Select patient results column**", tuple(uploaded_file.columns))
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
    

st.markdown("#### **:blue[Moving Averages Charts for Patient Based Quality Control]**")

st.write("---")

# Check if the data is uploaded or not
if uploaded_file is None:
    st.info("**Please firstly upload your data**", icon = "ℹ️")

else:
    try:
        data = analyte_data
    except NameError as error:
        print("NameError occurred:", error)
        st.error("Data wasn't uploaded")
        st.info("Please upload your data")
    
    st.write(" ")
    st.markdown('##### **:blue[Percentiles of the uploaded data]**')
    
    # Specify the percentiles
    percentiles = [1, 2, 3, 4, 5] 
    reversed_percentiles = [95, 96, 97, 98, 99]

    # Calculate the specified percentiles
    percentile_values = np.percentile(data, percentiles)
    reversed_percentile_values = np.percentile(data, reversed_percentiles)
    # Create a DataFrame to store the results
    percentile_table = pd.DataFrame({
        'Percentile Ranges': [f"{lower}-{upper}" for lower, upper in zip(percentiles, reversed(reversed_percentiles))],
        'Value Ranges': [f"{lower_val}-{upper_val}" for lower_val, upper_val in zip(percentile_values, reversed(reversed_percentile_values))]
    })
    st.dataframe(percentile_table, hide_index=True)
    
    st.write(" ") 
    st.markdown('##### **:blue[Distribution of the uploaded data]**')
    st.write(" ") 
    
    # Truncation Limits
    #truncation_limits = st.slider('**:red[Set the truncation limits]**', data.min(), data.max(), (data.min(), data.max()))
    col1, col2 = st.columns([1,1])
    lower_truncation_limit = col1.number_input('**:red[Lower Truncation Limit]**', value = float(data.min()), step = 0.00001, key = 7565)
    upper_truncation_limit = col2.number_input('**:red[Upper Truncation Limit]**', value = float(data.max()), step = 0.00001, key = 564377)
    truncation_limits = (lower_truncation_limit, upper_truncation_limit)
    # truncate data
    data = data[(data >= truncation_limits[0]) & (data <= truncation_limits[1])]
    
    # Box - Cox Transformation of the data
    box_cox_checkbox = st.checkbox('**:green[Box - Cox transformation]**')
    if box_cox_checkbox:
        fitted_data, fitted_lambda = stats.boxcox(data)
        #data = fitted_data
        data = pd.Series(fitted_data)
        #st.write(type(data))
    else:
        data = data

    # Create the Histogram figure
    fig_h = go.Figure()   

    fig_h.add_trace(go.Histogram(
        x=data,
        opacity=0.7,
        autobinx=True
    ))          
    # Customize the layout of the figure
    fig_h.update_layout(
        xaxis=dict(
            title=analyte_name_box,
            title_font=dict(
                size=12
            ),
            tickfont=dict(
                size=11
            ),
        ),
        yaxis=dict(
            title='Count',
            title_font=dict(
                size=12
            ),
            tickfont=dict(
                size=11
            )
        ),
        legend=dict(
            title='Category Intervals',
            xanchor='right',  # Position the legend on the right
            yanchor='top',  # Position the legend on the top
            x=0.98,  # Adjust the x position of the legend
            y=0.98,  # Adjust the y position of the legend
            traceorder='normal',
            font=dict(
                size=11
            )
        ),
        margin=dict(
            t=10,
            r=10,
            b=10,
            l=10
        ),
        height=500,
        width=800, barmode='stack',
    )

    # Show the figure using Streamlit
    st.plotly_chart(fig_h, theme="streamlit", use_container_width=True)

    
    # Mean SD selection
    st.markdown('**:blue[Enter custom mean/target and standard deviation]**')
    col1, col2 = st.columns([1,1])
    mean_input = col1.number_input('**:red[Mean]**', value = np.mean(data), step = 0.00001, format="%.5f")
    SD_input = col2.number_input('**:red[Standard Deviation]**', value = np.std(data), step = 0.00001, format="%.5f")
    st.write(f'**:blue[Mean: ]** **{mean_input}**, **:blue[SD: ]** **{SD_input}**')
    mean= mean_input
    std_dev = SD_input    
    st.write(" ")
    st.write("---")
    try:
        # Calculate control limits
        upper_limit_3sd = mean + 3 * std_dev
        lower_limit_3sd = mean - 3 * std_dev
        upper_limit_2sd = mean + 2 * std_dev
        lower_limit_2sd = mean - 2 * std_dev
        upper_limit_1sd = mean + 1 * std_dev
        lower_limit_1sd = mean - 1 * std_dev
    
        # Reset Index of Data
        data = data.reset_index(drop=True)
        # Create a dataframe for the plotly express function
        df = pd.DataFrame({'Data': data, 'Mean': mean, '+3SD': upper_limit_3sd, '-3SD': lower_limit_3sd,
                        '+2SD': upper_limit_2sd, '-2SD': lower_limit_2sd, 
                        '+1SD':upper_limit_1sd, '-1SD':lower_limit_1sd})

        # EWMA PLOT
        st.markdown('##### **:blue[Exponentially Weighted Moving Average]**')
        block_or_lambda = st.radio("**:blue[Specify block size or weighting factor for EWMA Chart]**",
            ["Block size", "Weighting factor"])
        if block_or_lambda == "Weighting factor":
            lambda_value_choice = st.select_slider('**:red[Select the lambda value (weighting factor) for EWMA chart]**',
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
            # ewma calculation based on specified lambda    
            ewma = df['Data'].ewm(alpha= lambda_value, span=None, adjust=False).mean()
        else:
            block_size = st.number_input('**:red[Enter block size]**', value =100, step = 1)
            ewma = df['Data'].ewm(span=block_size, adjust=False).mean()
            L = 3.000
            lambda_value = 2 / (block_size+1)

        try:
        # save ewma results in df (Exponential Weighted Moving Average (EWMA))
            df['ewma'] = ewma
        except Exception as e:
            st.error("Your data contains inappropriate type of values. Please check your data.")

        # Calculate UCL and LCL
        results = range(1, len(ewma) + 1)
        UCL_values = []
        LCL_values = []

        # UCL and LCL selection
        CL_select = st.radio("**:blue[Set control limits for EWMA Chart]**",
            ["Custom limits", "Custom L factor to change width of conventional UCL and LCL"])
        
        # EWMA plot with Custom limiti
        if CL_select == "Custom limits":
            st.markdown('**:blue[Enter custom control limits]**')
            col1, col2 = st.columns([1,1])
            LCL_input = col1.number_input('**:red[Lower Control Limit (LCL)]**', value =  mean - std_dev,step = 0.000000000001, format="%.12f")
            UCL_input = col2.number_input('**:red[Upper Control Limit (UCL)]**', value =  mean + std_dev, step = 0.000000000001, format="%.12f")
            st.markdown(f'**:blue[Custom LCL:]** **{LCL_input}**, **:blue[Custom UCL:]** **{UCL_input}**')
            UCL_array = np.array(UCL_input * np.ones((len(results), 1)))
            #UCL_array = pd.DataFrame(UCL_array)
            LCL_array = np.array(LCL_input * np.ones((len(results), 1)))
            #LCL_array = pd.DataFrame(LCL_array)
            
            df['UCL (custom)'] = UCL_array
            df['LCL (custom)'] = LCL_array
            
            UCL_values = UCL_array
            LCL_values = LCL_array
            
            # Create a EWMA Plotly figure for custom
            fig2 = go.Figure()
            
            # Add EWMA data
            fig2.add_trace(go.Scatter(x=ewma.index, y=ewma, mode='lines', name='EWMA'))
            
            # add UCL and LCL
            fig2.add_trace(go.Scatter(x=df.index, y=df['UCL (custom)'], mode='lines', name='UCL', line=dict(color='red')))
            fig2.add_trace(go.Scatter(x=df.index, y=df['LCL (custom)'], mode='lines', name='LCL', line=dict(color='blue')))
    
            # Add markers for points above UCL
            df['Out of UCL (custom)'] = (df['ewma'] > UCL_input)
            out_of_UCL = df[df['Out of UCL (custom)']]
            fig2.add_trace(go.Scatter(x=out_of_UCL.index, y=out_of_UCL['ewma'],
                                mode='markers', marker=dict(color='red'), showlegend=False,
                                name='UCL',text='Above of UCL'))
        
            # Add markers for points above UCL
            df['Out of LCL (custom)'] = (df['ewma'] < LCL_input)
            out_of_LCL = df[df['Out of LCL (custom)']]
            fig2.add_trace(go.Scatter(x=out_of_LCL.index, y=out_of_LCL['ewma'],
                                mode='markers', marker=dict(color='blue'), showlegend=False,
                                name='LCL',text='Above of LCL'))
            
            # Customize the layout
            fig2.update_layout(title=f'Exponentially Weighted Moving Average (EWMA) chart',
                            xaxis_title='Data point',
                            yaxis_title='Value', title_font=dict(color='#cc0000'))
            
            st.plotly_chart(fig2, theme="streamlit", use_container_width=True) 
            
            df[f'EWMA (lambda={lambda_value}) higher than UCL'] = (df['ewma'] >= UCL_input)      
            df[f'EWMA (lambda={lambda_value}) lower than LCL'] = (df['ewma'] <= LCL_input)   

        else:
            st.markdown('**:blue[Enter custom L factor]**')
            L_input = st.number_input('**:blue[L factor]**', value = L, step = 0.001, format="%.3f")
            st.markdown(f'**:blue[Custom L factor:]** **{L_input}**')
            L = L_input
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
            fig2.update_layout(title=f'Exponentially Weighted Moving Average (EWMA) chart',
                            xaxis_title='Data point',
                            yaxis_title='Value', title_font=dict(color='#cc0000'))
            
            st.plotly_chart(fig2, theme="streamlit", use_container_width=True) 
            df[f'EWMA (lambda={lambda_value}) higher than UCL'] = (ewma >= UCL_values)      
            df[f'EWMA (lambda={lambda_value}) lower than LCL'] = (ewma <= LCL_values) 
            
        st.write("---")
                
        # CUSUM PLOT
        st.markdown('##### **:blue[Cumulative Sum (CUSUM) Control Chart]**')
        # CUSUM plot function
        def plot_cusum(cusum_np_arr, mu, sd, h, k=0.5):
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
                showlegend=True,title_font=dict(color='#cc0000')
            )

            # Show figure
            st.plotly_chart(fig, theme="streamlit", use_container_width=True)
        
        # CUSUM limit selection
        h = st.number_input('**:red[Control Limit (h)]**', value = 10.0, step = 0.01, format="%.2f", key = 8756)
        plot_cusum(df['Data'], mean, std_dev, h)

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
        df['cusum Cp'] = Cp
        df['cusum Cm'] = Cm
        df[f'CUSUM higher than UCL'] = (Cp >= h)      
        df[f'CUSUM lower than LCL'] = (Cm >= h)      

        
        # show dataframe with out-of-control results notation
        with st.expander("**:blue[See the details of your data & download your data as .csv file]**"):
            st.dataframe(df)

    except NameError as ne:
        if 'data' in str(ne):
            st.info("Please upload your data")
        else:
            # Handle other NameError cases if needed
            print("A NameError occurred, but it's not related to 'data'")
