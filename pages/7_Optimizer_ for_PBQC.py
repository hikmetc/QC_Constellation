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
import statistics
from stqdm import stqdm


with st.sidebar:
    st.image('./images/QC_Constellation_sidebar.png')
    st.info('*Developed by Hikmet Can Çubukçu, MD, MSc, EuSpLM* <hikmetcancubukcu@gmail.com>')
    
    
st.markdown("#### **:blue[Optimize Parameters of Moving Averages Charts for Patient Based Quality Control]**")

st.write("---")

st.info('**Desktop application download link for Windows:** [Optimizer-4-PBQC.exe](https://drive.google.com/file/d/1nNtCXfHDpgUiW2tOfSDqdjexjEHJSW24/view?usp=sharing)')

comment_out = """
with st.sidebar:
    with open('./template/Na_normal.xlsx', "rb") as template_file:
        template_byte = template_file.read()
    # download template excel file
    st.download_button(label="Click to Download Template File",
                        data=template_byte,
                        file_name="Na_normal.xlsx",
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
        
        days_name_box = st.selectbox("**Select day/batch column**", tuple(uploaded_file.columns), key = 222)
        
        analyte_data = uploaded_file[analyte_name_box]
        analyte_data = analyte_data.dropna(axis=0).reset_index()
        analyte_data = analyte_data[analyte_name_box]
        
        day_data = uploaded_file[days_name_box]
        day_data = day_data.dropna(axis=0).reset_index()
        day_data = day_data[days_name_box]
                
        return analyte_data, analyte_name_box, day_data, days_name_box
    

    
    # column name (data) selection
    if uploaded_file is not None:
        # data of analyte selection
        analyte_data, analyte_name_box, day_data, days_name_box = process_file(uploaded_file)
        refreshable_analyte_data = analyte_data
        refreshable_analyte_name_box = analyte_name_box
        refreshable_day_data = day_data
        refreshable_days_name_box = days_name_box
    
    st.image('./images/QC_Constellation_sidebar.png')
    st.info('*Developed by Hikmet Can Çubukçu, MD, MSc, EuSpLM* <hikmetcancubukcu@gmail.com>')
    


st.markdown("#### **:blue[Optimize Parameters of Moving Averages Charts for Patient Based Quality Control]**")

st.write("---")

tab1, tab2 = st.tabs(["📉 **:green[Exponentially Weighted Moving Average (EWMA)]**", "📈 **:blue[Cumulative Sum (CUSUM)]**"])

with tab1:
    # Check if the data is uploaded or not
    if uploaded_file is None:
        st.info('**Desktop application download link for Windows:** [Optimizer-4-PBQC.exe](https://drive.google.com/file/d/1jOCSbMrqBvkCLgFpGjR71Se0j57XrlQE/view?usp=sharing)')
        st.info("**Please firstly upload your data**", icon = "ℹ️")

    else:
    
        try:
            data = refreshable_analyte_data
        except NameError as error:
            print("NameError occurred:", error)
            st.error("Data wasn't uploaded")
            st.info("Please upload your data")
        
        st.write(" ")

        # error addition index set as 10 if bacth size lower than 10 stored (NOT INPUT)
        error_added_point = 10
        st.info(f'**Please be sure the average results for each day/batch be higher than {error_added_point}**', icon = "ℹ️")

        # DATA LOAD
        # Day data - analyte data assignments -  Reset Index of Data
        day_data = refreshable_day_data[data.index] # take day data corresponding respective results
        data = data.reset_index(drop=True)
        day_data = day_data.reset_index(drop=True)

        # data renewal for loop refresh
        renewable_data = data
        renewable_day_data = day_data

        # input widgets
        # input variables
        col1, col2 = st.columns([1,1])
        TEa = col1.number_input('**:red[Allowable error (%)]**', value = 5.0, step = 0.01, format="%.2f", key = 1)
        allowable_FPR = col2.number_input('**:red[Allowable false positive rate (%)]**', value = 10.0, step = 0.1, format="%.1f", key = 2)
        # to be selected as lower than number of patient results of each day or batch
        #max_block_size = st.number_input('**:red[Maximum number of patient results of each day/batch]**', value = 160, step = 1, min_value=10, key = 3)
        max_block_size = 160
        # FPR adjustment (conversion into fraction) for further analysis
        FPR_filter = allowable_FPR/100

        # metric maker function
        def metrics_maker(final_df):
            final_df = final_df.copy()
            final_df['youden'] = final_df['value_error'] - final_df['value_0']
            
            # best parameters's performance metrics
            df_performance_of_best_parameters = {'Metric': ['Sensitivity (True Positive Rate)', 'Specificity', 'False Positive Rate', 
                                                            'Youden Index','ANPed', 'MNPed'], 
                                                'Value': [final_df['value_error'].iloc[0], 1-final_df['value_0'].iloc[0], 
                                                        final_df['value_0'].iloc[0], final_df['youden'].iloc[0], 
                                                        final_df['ANPed_error'].iloc[0], final_df['MNPed_error'].iloc[0]]}

            df_performance_of_best_parameters = pd.DataFrame(df_performance_of_best_parameters)
            return df_performance_of_best_parameters

        # LINE PLOT OF ANPED AND ERROR RATE
        def line_plot_of_ANPed(best_NPed_with_FPR_df, renewable_data, renewable_day_data):
            data = renewable_data
            day_data = renewable_day_data
            best_NPed_with_FPR_df.reset_index(inplace=True)

            split_columns_TL = best_NPed_with_FPR_df['truncation limit'].str.split('-', expand=True)

            split_columns_TL.columns = ['lower truncation limit', 'upper truncation limit']
            # Concatenate the split columns with the original DataFrame
            best_NPed_with_FPR_df = pd.concat([best_NPed_with_FPR_df, split_columns_TL], axis=1)
            # Drop the original column if you no longer need it
            best_NPed_with_FPR_df.drop(['truncation limit','index'], axis=1, inplace=True)
            best_NPed_with_FPR_df['lower truncation limit'] = best_NPed_with_FPR_df['lower truncation limit'].astype(float)
            best_NPed_with_FPR_df['upper truncation limit'] = best_NPed_with_FPR_df['upper truncation limit'].astype(float)

            lower_val = best_NPed_with_FPR_df['lower truncation limit'].loc[0]
            upper_val = best_NPed_with_FPR_df['upper truncation limit'].loc[0]
            transformation =  best_NPed_with_FPR_df['transformation status'].loc[0]
            UCL_input = best_NPed_with_FPR_df['upper control limit'].loc[0]
            LCL_input = best_NPed_with_FPR_df['lower control limit'].loc[0]
            block_size = best_NPed_with_FPR_df['block size'].loc[0]

            ANPed_list = []
            error_rate_list = []
            MNPed_list = []

            # Calculate the total number of iterations based on the parameters
            total_iterations = (
                len(np.arange(-1.0 * TEa ,1.1 * TEa, 0.1 * TEa))*
                day_data.nunique()
            )
            # Create tqdm instance
            pbar = stqdm(total=total_iterations, unit='iteration')

            error_rates = np.arange(-1.0 * TEa ,1.1 * TEa, 0.1 * TEa)

            # Set a tolerance close to machine epsilon for floating-point numbers
            tol = np.finfo(float).eps * 100  # Adjust factor as needed

            # Replace values less than tolerance with zero
            error_rates = np.where(np.abs(error_rates) < tol, 0.0, error_rates)

            for error_rate in error_rates:

                # refresh data
                data = renewable_data
                day_data = renewable_day_data

                truncation_limits = (lower_val, upper_val)
                def truncate(data, truncation_limits):
                    data = data[(data >= truncation_limits[0]) & (data <= truncation_limits[1])]
                    return data

                data = truncate(data, truncation_limits) # truncate data


                # box-cox transformation
                if transformation == "Raw Data":
                    data = data
                else:
                    fitted_data, fitted_lambda = stats.boxcox(data)
                    data = pd.Series(fitted_data)

                # reset index
                data = data.dropna()
                day_data = day_data[data.index]
                data = data.reset_index(drop=True)
                day_data = day_data.reset_index(drop=True)

                mean = np.mean(data)
                std_dev = np.std(data)
                
                # Create a dataframe         
                df = pd.DataFrame({'Day': day_data,'Data': data})

                # Control limit loop for ewma
                # ewma passed before

                alerts = 0
                eD_index_list = [] # list of rejection indexes
                
                # day loop
                for i in range(1, day_data.nunique()+1): 
                    # selecting i th day
                    day_based_df = df[df['Day']==i]

                    # reset index to add error appropriately
                    day_based_df = day_based_df.reset_index(drop=True)                        

                    
                    # Check if the block has enough data points for applying the error
                    if len(day_based_df) > error_added_point:
                        # Apply error after the block_size th index
                        day_based_df.loc[day_based_df.index[error_added_point:], 'Data'] *= (1 + error_rate / 100)
                    else:
                        number_of_odd_batches += 1
                        pass
                    
                    ewma = day_based_df['Data'].ewm(span=block_size, adjust=False).mean()
                    day_based_df['ewma'] = ewma
                    day_based_df['EWMA higher than UCL'] = (day_based_df['ewma'] >= UCL_input)      
                    day_based_df['EWMA lower than LCL'] = (day_based_df['ewma'] <= LCL_input)
                    
                    # alerts for rejection rates
                    if (day_based_df['EWMA higher than UCL'] == True).any() or (day_based_df['EWMA lower than LCL'] == True).any():
                        alerts += 1
                
                    # Find the first index where EWMA values exceed control limits after error_added_point
                    first_higher_than_UCL_index = day_based_df[(day_based_df.index >= error_added_point) & (day_based_df['EWMA higher than UCL'])].index.min()
                    first_lower_than_LCL_index = day_based_df[(day_based_df.index >= error_added_point) & (day_based_df['EWMA lower than LCL'])].index.min()

                    # Skip if index+1 < error_added_point
                    if first_higher_than_UCL_index is not None and first_higher_than_UCL_index + 1 >= error_added_point:
                        eD_index_list.append(first_higher_than_UCL_index + 1 - error_added_point)

                    if first_lower_than_LCL_index is not None and first_lower_than_LCL_index + 1 >= error_added_point:
                        eD_index_list.append(first_lower_than_LCL_index + 1 - error_added_point)
                            
                    pbar.update(1)

                # calculate updated alert
                positive_rate = alerts / day_data.nunique()
                ANPed = statistics.mean(eD_index_list) if eD_index_list else float('nan')
                MNPed = statistics.median(eD_index_list) if eD_index_list else float('nan')
                


                error_rate_list.append(error_rate)
                ANPed_list.append(ANPed)
                MNPed_list.append(MNPed)
                
            # Close tqdm instance
            pbar.close()

            # Create the line plot
            fig = go.Figure()

            # Add trace for ANPed_list on y-axis and error_rate_list on x-axis
            # exlcude error rate zero "0" and corresponding ANPed
            negative_error_rate_list = error_rate_list[:10]
            positive_error_rate_list = error_rate_list[11:]
            negative_ANPed_list = ANPed_list[:10]
            positive_ANPed_list = ANPed_list[11:]
            
            
            # Add trace for MNPed_list on y-axis and error_rate_list on x-axis
            # exlcude error rate zero "0" and corresponding MNPed
            negative_MNPed_list = MNPed_list[:10]
            positive_MNPed_list = MNPed_list[11:]

            fig.add_trace(go.Scatter(
                x=positive_error_rate_list,
                y=positive_ANPed_list,
                mode='lines+markers',
                name='ANPed (Positive Error Rates)'
            ))        
            
            fig.add_trace(go.Scatter(
                x=negative_error_rate_list,
                y=negative_ANPed_list,
                mode='lines+markers',
                name='ANPed (Negative Error Rates)'
            ))

            fig.add_trace(go.Scatter(
                x=positive_error_rate_list,
                y=positive_MNPed_list,
                mode='lines+markers',
                name='MNPed (Positive Error Rates)'
            ))

            fig.add_trace(go.Scatter(
                x=negative_error_rate_list,
                y=negative_MNPed_list,
                mode='lines+markers',
                name='MNPed (Negative Error Rates)'
            ))

            # Add a vertical line at the error rate of 0
            fig.add_vline(x=0, line=dict(color="red", width=1, dash="dash"))
            # Customize the layout
            fig.update_layout(
                title='Line Plot of ANPed and MNPed versus Error Rate',
                xaxis_title='Error Rate',
                yaxis_title='ANPed / MNPed', title_font=dict(color='#cc0000')
            )

            # Show the plot
            return st.plotly_chart(fig, theme="streamlit", use_container_width=True)

        optimize_button = st.button('**:green[Optimize]**', key = 78)
        if optimize_button:
            # Calculate the total number of iterations based on the parameters
            if max_block_size > 160:
                max_block_size_limit = 160
            else:
                max_block_size_limit = max_block_size

            total_iterations_ewma = (
                len(np.arange(0, 1.2, 1.0)) *
                len(np.percentile(data, [0, 1, 2, 3])) *
                2 *
                len(np.arange(0.5, 4.2, 0.5)) *
                len(np.arange(10, max_block_size_limit, 10)) *
                day_data.nunique()
            )
            # Create tqdm instance
            pbar = stqdm(total=total_iterations_ewma, unit='iteration')

            # performance metrics dictionary
            performance_metrics ={}

            # number batches that its size is lower than block size
            number_of_odd_batches = 0

            # truncation limits based percentile
            # Specify the percentiles
            percentiles = [0, 1, 2, 3]
            reversed_percentiles = [97, 98, 99, 100]

            # Calculate the specified percentiles
            percentile_values = np.percentile(data, percentiles)
            reversed_percentile_values = np.percentile(data, reversed_percentiles)

            error_rates = np.arange(0 * TEa ,1.2 * TEa, 1 * TEa)

            # Set a tolerance close to machine epsilon for floating-point numbers
            tol = np.finfo(float).eps * 100  # Adjust factor as needed

            # Replace values less than tolerance with zero
            error_rates = np.where(np.abs(error_rates) < tol, 0.0, error_rates)

            for error_rate in error_rates:

                # truncation limits loop
                for lower_val, upper_val in zip(percentile_values, reversed(reversed_percentile_values)):
                    # refresh data
                    data = renewable_data
                    day_data = renewable_day_data
                    
                    truncation_limits = (lower_val, upper_val)
                    def truncate(data, truncation_limits):
                        data = data[(data >= truncation_limits[0]) & (data <= truncation_limits[1])]
                        return data

                    data = truncate(data, truncation_limits) # truncate data

                    # box-cox transformation
                    for transformation in ("Raw Data", "Box-Cox Transformed Data"):
                        if transformation == "Raw Data":
                            data = data
                        else:
                            fitted_data, fitted_lambda = stats.boxcox(data)
                            data = pd.Series(fitted_data)

                        # reset index
                        data = data.dropna()
                        day_data = day_data[data.index]
                        data = data.reset_index(drop=True)
                        day_data = day_data.reset_index(drop=True)

                        mean = np.mean(data)
                        std_dev = np.std(data)
                        
                        # Create a dataframe         
                        df = pd.DataFrame({'Day': day_data,'Data': data})
                        
                        # Control limit loop for ewma
                        for limit in np.arange(0.5*std_dev, 4.2*std_dev, 0.5*std_dev):
                            # ewma
                            UCL_input = mean + limit
                            LCL_input = mean - limit

                            # Block size loop for ewma
                            # I limitted the block limit as 160 for this loop
                            if max_block_size > 160:
                                max_block_size_limit = 160
                            else:
                                max_block_size_limit = max_block_size
                                
                            for block_size in np.arange(10 ,max_block_size_limit, 10):
                                
                                alerts = 0
                                eD_index_list = [] # list of rejection indexes

                                # day loop
                                for i in range(1, day_data.nunique()+1): 

                                    # selecting i th day
                                    day_based_df = df[df['Day']==i]

                                    # reset index to add error appropriately
                                    day_based_df = day_based_df.reset_index(drop=True)                        

                                    
                                    # Check if the block has enough data points for applying the error
                                    if len(day_based_df) > error_added_point:
                                        # Apply error after the block_size th index
                                        day_based_df.loc[day_based_df.index[error_added_point:], 'Data'] *= (1 + error_rate / 100)
                                    else:
                                        number_of_odd_batches += 1
                                        pass
                                    
                                    ewma = day_based_df['Data'].ewm(span=block_size, adjust=False).mean()
                                    day_based_df['ewma'] = ewma
                                    day_based_df['EWMA higher than UCL'] = (day_based_df['ewma'] >= UCL_input)      
                                    day_based_df['EWMA lower than LCL'] = (day_based_df['ewma'] <= LCL_input)
                                    
                                    # alerts for rejection rates
                                    if (day_based_df['EWMA higher than UCL'] == True).any() or (day_based_df['EWMA lower than LCL'] == True).any():
                                        alerts += 1
                                
                                    # Find the first index where EWMA values exceed control limits after error_added_point
                                    first_higher_than_UCL_index = day_based_df[(day_based_df.index >= error_added_point) & (day_based_df['EWMA higher than UCL'])].index.min()
                                    first_lower_than_LCL_index = day_based_df[(day_based_df.index >= error_added_point) & (day_based_df['EWMA lower than LCL'])].index.min()

                                    # Skip if index+1 < error_added_point
                                    if first_higher_than_UCL_index is not None and first_higher_than_UCL_index + 1 >= error_added_point:
                                        eD_index_list.append(first_higher_than_UCL_index + 1 - error_added_point)

                                    if first_lower_than_LCL_index is not None and first_lower_than_LCL_index + 1 >= error_added_point:
                                        eD_index_list.append(first_lower_than_LCL_index + 1 - error_added_point)
                                            
                                    pbar.update(1)


                                # Calculate updated alert
                                positive_rate = alerts / day_data.nunique()
                                ANPed = statistics.mean(eD_index_list) if eD_index_list else float('nan')
                                MNPed = statistics.median(eD_index_list) if eD_index_list else float('nan')
                                performance_metrics[f'{lower_val}-{upper_val}, {transformation}, {LCL_input}, {UCL_input}, {block_size}, {ANPed}, {MNPed}, {error_rate}'] = positive_rate
                                    
            # Close tqdm instance
            pbar.close()

            performance_metrics_df = pd.DataFrame.from_dict(performance_metrics, orient='index', columns=['value'])
            performance_metrics_df.reset_index(inplace=True)
            performance_metrics_df.rename(columns={'index': 'parameter'}, inplace=True)
            split_columns = performance_metrics_df['parameter'].str.split(', ', expand=True)

            # Rename the new columns for clarity
            split_columns.columns = ['truncation limit', 'transformation status', 'lower control limit', 'upper control limit', 'block size', 'ANPed', 'MNPed','TEa']

            # Concatenate the split columns with the original DataFrame
            performance_metrics_df = pd.concat([performance_metrics_df, split_columns], axis=1)

            # Drop the original column if you no longer need it
            performance_metrics_df.drop('parameter', axis=1, inplace=True)

            performance_metrics_df['value'] = performance_metrics_df['value'].astype(float)
            performance_metrics_df['TEa'] = performance_metrics_df['TEa'].astype(float)

            performance_metrics_df['lower control limit'] = performance_metrics_df['lower control limit'].astype(float)
            performance_metrics_df['upper control limit'] = performance_metrics_df['upper control limit'].astype(float)
            performance_metrics_df['block size'] = performance_metrics_df['block size'].astype(int)
        
            performance_metrics_df['ANPed'] = performance_metrics_df['ANPed'].astype(float)        
            performance_metrics_df['MNPed'] = performance_metrics_df['MNPed'].astype(float)

            df_TEa_0 = performance_metrics_df[performance_metrics_df['TEa'] == 0]
            df_TEa_3 = performance_metrics_df[performance_metrics_df['TEa'] == TEa]

            # Merge the two DataFrames on corresponding parameters
            merged_df = pd.merge(df_TEa_0, df_TEa_3, on=['truncation limit', 'transformation status', 'lower control limit', 'upper control limit', 'block size'], suffixes=('_0', '_error'))
            
            # drop if ANPed_error is nan
            merged_df.dropna(subset=['ANPed_error'], inplace=True) # drop if ANPed_error is nan
            merged_df['ANPed_error'] = merged_df['ANPed_error'].round().astype(int)  
            
            # Filter parameters where value_0 < 0.1 and TEa_0 = 0
            filtered_df = merged_df[(merged_df['value_0'] < FPR_filter) & (merged_df['TEa_0'] == 0)]

            # Find the maximum value_error when TEa_error = 3 among the filtered parameters
            max_value_error = filtered_df.loc[filtered_df['TEa_error'] == TEa, 'value_error'].max()

            # Find parameters corresponding to the maximum value_error
            max_value_error_params = filtered_df[filtered_df['value_error'] == max_value_error][['truncation limit', 'transformation status', 'lower control limit', 'upper control limit', 'block size', 'ANPed_error', 'MNPed_error']]

            # Based on FPR criteria as 10% if exist; if not returs parameters based on only highest youden index
            if len(filtered_df) > 0:
                # parameters correspond to FPR < 10% exists
                final_df = filtered_df.copy()  # Ensure we're working with a copy
                final_df['youden'] = final_df['value_error'] - final_df['value_0']
            else:
                # parameters correspond to FPR < 10% do not exists
                print("False positive rate is unattainable !")
                final_df = merged_df.copy()  # Ensure we're working with a copy
                final_df['youden'] = final_df['value_error'] - final_df['value_0']    
            # best parameters's performance metrics
            best_performance_output = final_df[final_df['youden'] == final_df['youden'].max()]
            
            df_performance_of_best_parameters = {'Metric': ['Sensitivity (True Positive Rate)', 'Specificity', 'False Positive Rate', 'Youden Index'], 
                                                'Value': [best_performance_output['value_error'].iloc[0], 1-best_performance_output['value_0'].iloc[0], 
                                                        best_performance_output['value_0'].iloc[0], best_performance_output['youden'].iloc[0]]}

            df_performance_of_best_parameters = pd.DataFrame(df_performance_of_best_parameters)

            # Among best youden - lowest ANPed
            Among_best_youden_best_ANPed_df = best_performance_output[best_performance_output['ANPed_error'] == best_performance_output['ANPed_error'].min()]
            # parameters
            st.markdown('##### **:blue[EWMA scheme parameters optimized based on the highest youden index]**')
            parameters_of_best_seleced_metrics = Among_best_youden_best_ANPed_df[['truncation limit', 'transformation status','lower control limit', 'upper control limit', 'block size']]
            st.dataframe(parameters_of_best_seleced_metrics, hide_index=True)  
            # Metrics best youden - lowest ANPed
            st.markdown('###### **:blue[Performance]**')
            best_seleced_metrics = metrics_maker(Among_best_youden_best_ANPed_df)
            st.dataframe(best_seleced_metrics, hide_index=True)
            line_plot_of_ANPed(parameters_of_best_seleced_metrics, renewable_data, renewable_day_data)

            st.write("---")

            # Among acceptable FPR - best ANPed
            merged_df = merged_df.copy()  # Ensure we're working with a copy
            merged_df['youden'] = merged_df['value_error'] - merged_df['value_0']
            try:
                fpr_lower_than_limit_Y05_df = merged_df[(merged_df['value_0'] < FPR_filter) & (merged_df['youden'] > 0.5)]
            except:
                fpr_lower_than_limit_Y05_df = merged_df[merged_df['youden'] > 0.5]

            Among_acceptable_FPR_best_ANPed_df = fpr_lower_than_limit_Y05_df[fpr_lower_than_limit_Y05_df['ANPed_error'] == fpr_lower_than_limit_Y05_df['ANPed_error'].min()]
            if len(Among_acceptable_FPR_best_ANPed_df) > 0:
                # parameters
                st.markdown('##### **:blue[EWMA scheme parameters optimized based with possibly lowest ANPed]**')
                best_NPed_with_FPR_df = Among_acceptable_FPR_best_ANPed_df[['truncation limit', 'transformation status', 'lower control limit', 'upper control limit', 'block size']]
                st.dataframe(best_NPed_with_FPR_df, hide_index=True)
                # performance
                st.markdown('###### **:blue[Performance]**')
                best_seleced_metrics_ANPed = metrics_maker(Among_acceptable_FPR_best_ANPed_df)
                st.dataframe(best_seleced_metrics_ANPed, hide_index=True)
                line_plot_of_ANPed(best_NPed_with_FPR_df, renewable_data, renewable_day_data)
            else: 
                st.info("**No scheme could achieve Youden Index higher than 0.5**", icon = "ℹ️")
            

            mean = np.mean(renewable_data)
            std_dev = np.std(renewable_data)
"""            
#            st.markdown(f"""
 #                           | *:green[Data statistics]* | *:green[Value]* |
  #                          | ----------- | ----------- |
   #                         | **:black[Mean]** | **{round(mean,2)}** |
    #                        | **:black[Standard Deviation]** | **{round(std_dev,2)}** |
     #                       """)

commment_out_2 = """
with tab2:
    # Check if the data is uploaded or not
    if uploaded_file is None:
        st.info('**Desktop application download link for Windows:** [Optimizer-4-PBQC.exe](https://drive.google.com/file/d/1jOCSbMrqBvkCLgFpGjR71Se0j57XrlQE/view?usp=sharing)')
        st.info("**Please firstly upload your data**", icon = "ℹ️")

    else:
    
        try:
            data = refreshable_analyte_data
        except NameError as error:
            print("NameError occurred:", error)
            st.error("Data wasn't uploaded")
            st.info("Please upload your data")
        
        st.write(" ")

        # error addition index set as 10 if bacth size lower than 10 stored (NOT INPUT)
        error_added_point = 10
        st.info(f'**Please be sure the average results for each day/batch be higher than {error_added_point}**', icon = "ℹ️")

        # DATA LOAD
        # Day data - analyte data assignments -  Reset Index of Data
        day_data = refreshable_day_data[data.index] # take day data corresponding respective results
        data = data.reset_index(drop=True)
        day_data = day_data.reset_index(drop=True)

        # data renewal for loop refresh
        renewable_data = data
        renewable_day_data = day_data

        # input widgets
        # input variables
        col1, col2 = st.columns([1,1])
        TEa = col1.number_input('**:red[Allowable error (%)]**', value = 5.0, step = 0.01, format="%.2f", key = 4)
        allowable_FPR = col2.number_input('**:red[Allowable false positive rate (%)]**', value = 10.0, step = 0.1, format="%.1f", key = 5)
        # FPR adjustment (conversion into fraction) for further analysis
        FPR_filter = allowable_FPR/100

        # metric maker function
        def metrics_maker(final_df):
            final_df = final_df.copy()
            final_df['youden'] = final_df['value_error'] - final_df['value_0']
            
            # best parameters's performance metrics
            df_performance_of_best_parameters = {'Metric': ['Sensitivity (True Positive Rate)', 'Specificity', 'False Positive Rate', 
                                                            'Youden Index','ANPed', 'MNPed'], 
                                                'Value': [final_df['value_error'].iloc[0], 1-final_df['value_0'].iloc[0], 
                                                        final_df['value_0'].iloc[0], final_df['youden'].iloc[0], 
                                                        final_df['ANPed_error'].iloc[0], final_df['MNPed_error'].iloc[0]]}

            df_performance_of_best_parameters = pd.DataFrame(df_performance_of_best_parameters)
            return df_performance_of_best_parameters

        # LINE PLOT OF ANPED AND ERROR RATE
        def line_plot_of_ANPed(best_NPed_with_FPR_df, renewable_data, renewable_day_data):
            data = renewable_data
            day_data = renewable_day_data
            best_NPed_with_FPR_df.reset_index(inplace=True)

            split_columns_TL = best_NPed_with_FPR_df['truncation limit'].str.split('-', expand=True)

            split_columns_TL.columns = ['lower truncation limit', 'upper truncation limit']
            # Concatenate the split columns with the original DataFrame
            best_NPed_with_FPR_df = pd.concat([best_NPed_with_FPR_df, split_columns_TL], axis=1)
            # Drop the original column if you no longer need it
            best_NPed_with_FPR_df.drop(['truncation limit','index'], axis=1, inplace=True)
            best_NPed_with_FPR_df['lower truncation limit'] = best_NPed_with_FPR_df['lower truncation limit'].astype(float)
            best_NPed_with_FPR_df['upper truncation limit'] = best_NPed_with_FPR_df['upper truncation limit'].astype(float)

            lower_val = best_NPed_with_FPR_df['lower truncation limit'].loc[0]
            upper_val = best_NPed_with_FPR_df['upper truncation limit'].loc[0]
            transformation =  best_NPed_with_FPR_df['transformation status'].loc[0]
            limit =  best_NPed_with_FPR_df['control limit (h)'].loc[0]

            ANPed_list = []
            error_rate_list = []
            MNPed_list = []

            # Calculate the total number of iterations based on the parameters
            total_iterations = (
                len(np.arange(-1.0 * TEa ,1.1 * TEa, 0.1 * TEa))*
                day_data.nunique()
            )
            # Create tqdm instance
            pbar3 = stqdm(total=total_iterations, unit='iteration')

            error_rates = np.arange(-1.0 * TEa ,1.1 * TEa, 0.1 * TEa)

            # Set a tolerance close to machine epsilon for floating-point numbers
            tol = np.finfo(float).eps * 100  # Adjust factor as needed

            # Replace values less than tolerance with zero
            error_rates = np.where(np.abs(error_rates) < tol, 0.0, error_rates)

            for error_rate in error_rates:

                # refresh data
                data = renewable_data
                day_data = renewable_day_data

                truncation_limits = (lower_val, upper_val)
                def truncate(data, truncation_limits):
                    data = data[(data >= truncation_limits[0]) & (data <= truncation_limits[1])]
                    return data

                data = truncate(data, truncation_limits) # truncate data


                # box-cox transformation
                if transformation == "Raw Data":
                    data = data
                else:
                    fitted_data, fitted_lambda = stats.boxcox(data)
                    data = pd.Series(fitted_data)

                # reset index
                data = data.dropna()
                day_data = day_data[data.index]
                data = data.reset_index(drop=True)
                day_data = day_data.reset_index(drop=True)

                mean = np.mean(data)
                std_dev = np.std(data)
                
                # Create a dataframe         
                df = pd.DataFrame({'Day': day_data,'Data': data})

                # Control limit loop for ewma
                h = limit
                alerts = 0
                eD_index_list = [] # list of rejection indexes
                
                # day loop
                for i in range(1, day_data.nunique()+1): 
                    # selecting i th day
                    day_based_df = df[df['Day']==i]

                    # reset index to add error appropriately
                    day_based_df = day_based_df.reset_index(drop=True)                        
                    
                    # Check if the block has enough data points for applying the error
                    if len(day_based_df) > error_added_point:
                        # Apply error after the block_size th index
                        day_based_df.loc[day_based_df.index[error_added_point:], 'Data'] *= (1 + error_rate / 100)
                    else:
                        number_of_odd_batches += 1
                        pass
                    
                    # This part add cusum results to the dataframe
                    cusum_np_arr = day_based_df['Data']
                    k=0.5    
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

                    day_based_df[f'CUSUM higher than UCL'] = (Cp >= h)      
                    day_based_df[f'CUSUM lower than LCL'] = (Cm >= h)
                    

                    # alerts for rejection rates
                    if (day_based_df['CUSUM higher than UCL'] == True).any() or (day_based_df['CUSUM lower than LCL'] == True).any():
                        alerts += 1
                
                    # Find the first index where CUSUM values exceed control limits after error_added_point
                    first_higher_than_UCL_index = day_based_df[(day_based_df.index >= error_added_point) & (day_based_df['CUSUM higher than UCL'])].index.min()
                    first_lower_than_LCL_index = day_based_df[(day_based_df.index >= error_added_point) & (day_based_df['CUSUM lower than LCL'])].index.min()

                    # Skip if index+1 < error_added_point
                    if first_higher_than_UCL_index is not None and first_higher_than_UCL_index + 1 >= error_added_point:
                        eD_index_list.append(first_higher_than_UCL_index + 1 - error_added_point)

                    if first_lower_than_LCL_index is not None and first_lower_than_LCL_index + 1 >= error_added_point:
                        eD_index_list.append(first_lower_than_LCL_index + 1 - error_added_point)
                            
                    pbar3.update(1)

                # calculate updated alert
                positive_rate = alerts / day_data.nunique()
                ANPed = statistics.mean(eD_index_list) if eD_index_list else float('nan')
                MNPed = statistics.median(eD_index_list) if eD_index_list else float('nan')
        
                error_rate_list.append(error_rate)
                ANPed_list.append(ANPed)
                MNPed_list.append(MNPed)
                
            # Close tqdm instance
            pbar3.close()

            # Create the line plot
            fig = go.Figure()

            # Add trace for ANPed_list on y-axis and error_rate_list on x-axis
            # exlcude error rate zero "0" and corresponding ANPed
            negative_error_rate_list = error_rate_list[:10]
            positive_error_rate_list = error_rate_list[11:]
            negative_ANPed_list = ANPed_list[:10]
            positive_ANPed_list = ANPed_list[11:]
            
            
            # Add trace for MNPed_list on y-axis and error_rate_list on x-axis
            # exlcude error rate zero "0" and corresponding MNPed
            negative_MNPed_list = MNPed_list[:10]
            positive_MNPed_list = MNPed_list[11:]

            fig.add_trace(go.Scatter(
                x=positive_error_rate_list,
                y=positive_ANPed_list,
                mode='lines+markers',
                name='ANPed (Positive Error Rates)'
            ))        
            
            fig.add_trace(go.Scatter(
                x=negative_error_rate_list,
                y=negative_ANPed_list,
                mode='lines+markers',
                name='ANPed (Negative Error Rates)'
            ))

            fig.add_trace(go.Scatter(
                x=positive_error_rate_list,
                y=positive_MNPed_list,
                mode='lines+markers',
                name='MNPed (Positive Error Rates)'
            ))

            fig.add_trace(go.Scatter(
                x=negative_error_rate_list,
                y=negative_MNPed_list,
                mode='lines+markers',
                name='MNPed (Negative Error Rates)'
            ))

            # Add a vertical line at the error rate of 0
            fig.add_vline(x=0, line=dict(color="red", width=1, dash="dash"))
            # Customize the layout
            fig.update_layout(
                title='Line Plot of ANPed and MNPed versus Error Rate',
                xaxis_title='Error Rate',
                yaxis_title='ANPed / MNPed', title_font=dict(color='#cc0000')
            )

            # Show the plot
            return st.plotly_chart(fig, theme="streamlit", use_container_width=True)

        optimize_button = st.button('**:green[Optimize]**', key = 87)
        if optimize_button:
            # Calculate the total number of iterations based on the parameters
            total_iterations_cusum = (
                len(np.arange(0 * TEa, 1.2 * TEa, 1.0 * TEa)) *
                len(np.percentile(data, [0, 1, 2, 3])) *
                2 *
                len(np.arange(5, 65, 5)) *
                day_data.nunique()
            )
            # Create tqdm instance
            pbar2 = stqdm(total=total_iterations_cusum, unit='iteration')

            # performance metrics dictionary
            performance_metrics ={}

            # number batches that its size is lower than block size
            number_of_odd_batches = 0

            # truncation limits based percentile
            # Specify the percentiles
            percentiles = [0, 1, 2, 3]
            reversed_percentiles = [97, 98, 99, 100]

            # Calculate the specified percentiles
            percentile_values = np.percentile(data, percentiles)
            reversed_percentile_values = np.percentile(data, reversed_percentiles)

            error_rates = np.arange(0 * TEa ,1.2 * TEa, 1 * TEa)

            # Set a tolerance close to machine epsilon for floating-point numbers
            tol = np.finfo(float).eps * 100  # Adjust factor as needed

            # Replace values less than tolerance with zero
            error_rates = np.where(np.abs(error_rates) < tol, 0.0, error_rates)

            for error_rate in error_rates:

                # truncation limits loop
                for lower_val, upper_val in zip(percentile_values, reversed(reversed_percentile_values)):
                    # refresh data
                    data = renewable_data
                    day_data = renewable_day_data
                    
                    truncation_limits = (lower_val, upper_val)
                    def truncate(data, truncation_limits):
                        data = data[(data >= truncation_limits[0]) & (data <= truncation_limits[1])]
                        return data

                    data = truncate(data, truncation_limits) # truncate data

                    # box-cox transformation
                    for transformation in ("Raw Data", "Box-Cox Transformed Data"):
                        if transformation == "Raw Data":
                            data = data
                        else:
                            fitted_data, fitted_lambda = stats.boxcox(data)
                            data = pd.Series(fitted_data)

                        # reset index
                        data = data.dropna()
                        day_data = day_data[data.index]
                        data = data.reset_index(drop=True)
                        day_data = day_data.reset_index(drop=True)

                        mean = np.mean(data)
                        std_dev = np.std(data)
                        
                        # Create a dataframe         
                        df = pd.DataFrame({'Day': day_data,'Data': data})
                        
                        # Control limit loop for ewma
                        for limit in np.arange(5, 65, 5):
                            
                            h = limit # control limit
                            alerts = 0
                            eD_index_list = [] # list of rejection indexes

                            # day loop
                            for i in range(1, day_data.nunique()+1): 

                                # selecting i th day
                                day_based_df = df[df['Day']==i]

                                # reset index to add error appropriately
                                day_based_df = day_based_df.reset_index(drop=True)                        

                                
                                # Check if the block has enough data points for applying the error
                                if len(day_based_df) > error_added_point:
                                    # Apply error after the block_size th index
                                    day_based_df.loc[day_based_df.index[error_added_point:], 'Data'] *= (1 + error_rate / 100)
                                else:
                                    number_of_odd_batches += 1
                                    pass
                                
                                # This part add cusum results to the dataframe
                                cusum_np_arr = day_based_df['Data']
                                k=0.5    
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
                                
                                day_based_df[f'CUSUM higher than UCL'] = (Cp >= h)      
                                day_based_df[f'CUSUM lower than LCL'] = (Cm >= h)
                                
                                # alerts for rejection rates
                                if (day_based_df['CUSUM higher than UCL'] == True).any() or (day_based_df['CUSUM lower than LCL'] == True).any():
                                    alerts += 1
                            
                                # Find the first index where CUSUM values exceed control limits after error_added_point
                                first_higher_than_UCL_index = day_based_df[(day_based_df.index >= error_added_point) & (day_based_df['CUSUM higher than UCL'])].index.min()
                                first_lower_than_LCL_index = day_based_df[(day_based_df.index >= error_added_point) & (day_based_df['CUSUM lower than LCL'])].index.min()

                                # Skip if index+1 < error_added_point
                                if first_higher_than_UCL_index is not None and first_higher_than_UCL_index + 1 >= error_added_point:
                                    eD_index_list.append(first_higher_than_UCL_index + 1 - error_added_point)

                                if first_lower_than_LCL_index is not None and first_lower_than_LCL_index + 1 >= error_added_point:
                                    eD_index_list.append(first_lower_than_LCL_index + 1 - error_added_point)
                                        
                                pbar2.update(1)


                            # Calculate updated alert
                            positive_rate = alerts / day_data.nunique()
                            ANPed = statistics.mean(eD_index_list) if eD_index_list else float('nan')
                            MNPed = statistics.median(eD_index_list) if eD_index_list else float('nan')
                            performance_metrics[f'{lower_val}-{upper_val}, {transformation}, {limit}, {ANPed}, {MNPed}, {error_rate}'] = positive_rate
                                    
            # Close tqdm instance
            pbar2.close()

            performance_metrics_df = pd.DataFrame.from_dict(performance_metrics, orient='index', columns=['value'])
            performance_metrics_df.reset_index(inplace=True)
            performance_metrics_df.rename(columns={'index': 'parameter'}, inplace=True)
            split_columns = performance_metrics_df['parameter'].str.split(', ', expand=True)

            # Rename the new columns for clarity
            split_columns.columns = ['truncation limit', 'transformation status', 'control limit (h)', 'ANPed', 'MNPed','TEa']

            # Concatenate the split columns with the original DataFrame
            performance_metrics_df = pd.concat([performance_metrics_df, split_columns], axis=1)

            # Drop the original column if you no longer need it
            performance_metrics_df.drop('parameter', axis=1, inplace=True)

            performance_metrics_df['value'] = performance_metrics_df['value'].astype(float)
            performance_metrics_df['TEa'] = performance_metrics_df['TEa'].astype(float)

            performance_metrics_df['control limit (h)'] = performance_metrics_df['control limit (h)'].astype(float)
        
            performance_metrics_df['ANPed'] = performance_metrics_df['ANPed'].astype(float)        
            performance_metrics_df['MNPed'] = performance_metrics_df['MNPed'].astype(float)

            df_TEa_0 = performance_metrics_df[performance_metrics_df['TEa'] == 0]
            df_TEa_3 = performance_metrics_df[performance_metrics_df['TEa'] == TEa]

            # Merge the two DataFrames on corresponding parameters
            merged_df = pd.merge(df_TEa_0, df_TEa_3, on=['truncation limit', 'transformation status', 'control limit (h)'], suffixes=('_0', '_error'))
            
            # drop if ANPed_error is nan
            merged_df.dropna(subset=['ANPed_error'], inplace=True) # drop if ANPed_error is nan
            merged_df['ANPed_error'] = merged_df['ANPed_error'].round().astype(int)  
            
            # Filter parameters where value_0 < 0.1 and TEa_0 = 0
            filtered_df = merged_df[(merged_df['value_0'] < FPR_filter) & (merged_df['TEa_0'] == 0)]

            # Find the maximum value_error when TEa_error = 3 among the filtered parameters
            max_value_error = filtered_df.loc[filtered_df['TEa_error'] == TEa, 'value_error'].max()

            # Find parameters corresponding to the maximum value_error
            max_value_error_params = filtered_df[filtered_df['value_error'] == max_value_error][['truncation limit', 'transformation status', 'control limit (h)', 'ANPed_error', 'MNPed_error']]

            # Based on FPR criteria as 10% if exist; if not returs parameters based on only highest youden index
            if len(filtered_df) > 0:
                # parameters correspond to FPR < 10% exists
                final_df = filtered_df.copy()  # Ensure we're working with a copy
                final_df['youden'] = final_df['value_error'] - final_df['value_0']
            else:
                # parameters correspond to FPR < 10% do not exists
                print("False positive rate is unattainable !")
                final_df = merged_df.copy()  # Ensure we're working with a copy
                final_df['youden'] = final_df['value_error'] - final_df['value_0']    
            # best parameters's performance metrics
            best_performance_output = final_df[final_df['youden'] == final_df['youden'].max()]
            
            df_performance_of_best_parameters = {'Metric': ['Sensitivity (True Positive Rate)', 'Specificity', 'False Positive Rate', 'Youden Index'], 
                                                'Value': [best_performance_output['value_error'].iloc[0], 1-best_performance_output['value_0'].iloc[0], 
                                                        best_performance_output['value_0'].iloc[0], best_performance_output['youden'].iloc[0]]}

            df_performance_of_best_parameters = pd.DataFrame(df_performance_of_best_parameters)

            # Among best youden - lowest ANPed
            Among_best_youden_best_ANPed_df = best_performance_output[best_performance_output['ANPed_error'] == best_performance_output['ANPed_error'].min()]
            # parameters
            st.markdown('##### **:blue[CUSUM scheme parameters optimized based on the highest youden index]**')
            parameters_of_best_seleced_metrics = Among_best_youden_best_ANPed_df[['truncation limit', 'transformation status', 'control limit (h)']]
            st.dataframe(parameters_of_best_seleced_metrics, hide_index=True)  
            # Metrics best youden - lowest ANPed
            st.markdown('###### **:blue[Performance]**')
            best_seleced_metrics = metrics_maker(Among_best_youden_best_ANPed_df)
            st.dataframe(best_seleced_metrics, hide_index=True)
            line_plot_of_ANPed(parameters_of_best_seleced_metrics, renewable_data, renewable_day_data)

            st.write("---")

            # Among acceptable FPR - best ANPed
            merged_df = merged_df.copy()  # Ensure we're working with a copy
            merged_df['youden'] = merged_df['value_error'] - merged_df['value_0']
            try:
                fpr_lower_than_limit_Y05_df = merged_df[(merged_df['value_0'] < FPR_filter) & (merged_df['youden'] > 0.5)]
            except:
                fpr_lower_than_limit_Y05_df = merged_df[merged_df['youden'] > 0.5]

            Among_acceptable_FPR_best_ANPed_df = fpr_lower_than_limit_Y05_df[fpr_lower_than_limit_Y05_df['ANPed_error'] == fpr_lower_than_limit_Y05_df['ANPed_error'].min()]
            if len(Among_acceptable_FPR_best_ANPed_df) > 0:
                # parameters
                st.markdown('##### **:blue[CUSUM scheme parameters optimized based with possibly lowest ANPed]**')
                best_NPed_with_FPR_df = Among_acceptable_FPR_best_ANPed_df[['truncation limit', 'transformation status', 'control limit (h)']]
                st.dataframe(best_NPed_with_FPR_df, hide_index=True)
                # performance
                st.markdown('###### **:blue[Performance]**')
                best_seleced_metrics_ANPed = metrics_maker(Among_acceptable_FPR_best_ANPed_df)
                st.dataframe(best_seleced_metrics_ANPed, hide_index=True)
                line_plot_of_ANPed(best_NPed_with_FPR_df, renewable_data, renewable_day_data)
            else: 
                st.info("**No scheme could achieve Youden Index higher than 0.5**", icon = "ℹ️")
            
            st.markdown(' ')
            st.write("---")
            mean = np.mean(renewable_data)
            std_dev = np.std(renewable_data)
"""            
#            st.markdown(f"""
 #                           | *:green[Data statistics]* | *:green[Value]* |
  #                          | ----------- | ----------- |
   #                         | **:black[Mean]** | **{round(mean,2)}** |
    #                        | **:black[Standard Deviation]** | **{round(std_dev,2)}** |
     #                       """)



