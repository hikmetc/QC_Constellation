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
    st.image('./images/QC Constellation icon.png')
    st.info('*Developed by Hikmet Can Çubukçu, MD, EuSpLM* <hikmetcancubukcu@gmail.com>')
    

st.markdown("### **:blue[Sigma-metric & Method Decision Charts]**")
st.write(" ")
def round_half_up(n, decimals=0):
            multiplier = 10**decimals
            return math.floor(n * multiplier + 0.5) / multiplier
col1, col2 = st.columns([1,1])
col1.markdown('**:blue[Conventional Sigmametric]**')
col1.latex(r'''\frac{{\text{{TEa(\%) - Bias(\%)}}}}{{\text{{CV}}_{A}(\%)}}''')
TEa_input = col1.number_input('**Total Allowable Error (TEa%)**', min_value=0.0,step = 0.00001)
bias_input = col1.number_input('**Bias (%)**', min_value=0.0, step = 0.00001)
CV_input = col1.number_input('**Imprecision (CVA%)**', min_value=0.0, step = 0.00001)
# alternative sigmametric
col2.markdown('**:blue[Alternative Sigmametric]**')
col2.latex(r'''\frac{{\text{{MAU(\%) or }}\text{{CV}}_I(\%)}}{{\text{{CV}}_A(\%)}}''')
CVI_input = col2.number_input('**Within-subject BV(%CVI) or MAU(%)**', min_value=0.0, step = 0.00001)
#CV2_input = col2.number_input('**Analytical Coefficient of Variation**', min_value=0.0, step = 0.00001)
# Calculate button "Simulate & Calculate"
calc_button = st.button('**:green[Calculate & Plot OPSpecs Chart]**')
if calc_button:
    if not CV_input==0:
        if bias_input < TEa_input:
            sigmametric_result_v1 = (TEa_input-bias_input)/CV_input
            col1.info(f""" **:green[Sigmametric value (conventional)]** : 
                        **{round_half_up((sigmametric_result_v1),2)}** 
                            """)
        else:
            col1.error("""**:red[Attention: Bias ≥ TEa]**""")
    else:
        col1.error("""**:red[Imprecision (%CV) value can not be zero]**""")
    
    
    if not CV_input==0:
        col2.info(f"""
                **:green[Sigmametric value (alternative)]** : **{round_half_up((CVI_input/CV_input),2)}**
                    """)
    else:
        col2.error("""**:red[Imprecision (%CV) value can not be zero]**""")
        
    if not TEa_input == 0:
        y_limit_min = 0
        y_limit_max = TEa_input
        x_limit_min = 0
        x_limit_max = TEa_input/2
        
        sigma_2 = TEa_input/2
        sigma_3 = TEa_input/3
        sigma_4 = TEa_input/4
        sigma_5 = TEa_input/5
        sigma_6 = TEa_input/6
        
        # Create a DataFrame with the data point
        data = {'Imprecision (%CV)': [CV_input], 'Bias (%)': [bias_input]}  # Assuming y=0 for simplicity
        df = pd.DataFrame(data)

        # Create a scatter plot using plotly express
        fig = px.scatter(df, x='Imprecision (%CV)', y='Bias (%)', title='OPSpecs Chart')

        # Add lines for sigma_2, sigma_3, sigma_4, sigma_5, and sigma_6
        fig.add_trace(px.line(x=[sigma_2, 0], y=[0, y_limit_max], line_shape='linear').data[0].update(line=dict(color='red',width=1)))
        fig.add_trace(px.line(x=[sigma_3, 0], y=[0, y_limit_max], line_shape='linear').data[0].update(line=dict(color='orange', width=1)))
        fig.add_trace(px.line(x=[sigma_4, 0], y=[0, y_limit_max], line_shape='linear').data[0].update(line=dict(color='purple', width=1)))
        fig.add_trace(px.line(x=[sigma_5, 0], y=[0, y_limit_max], line_shape='linear').data[0].update(line=dict(color='blue', width=1)))
        fig.add_trace(px.line(x=[sigma_6, 0], y=[0, y_limit_max], line_shape='linear').data[0].update(line=dict(color='green', width=1)))
        
        # Add annotations directly on the lines with adjusted angle
        fig.add_annotation(x=sigma_2/2+sigma_2/20, y=y_limit_max/2, text='2', showarrow=False, font=dict(color='red'), textangle=0)
        fig.add_annotation(x=sigma_3/2+sigma_3/20, y=y_limit_max/2, text='3', showarrow=False, font=dict(color='orange'), textangle=0)
        fig.add_annotation(x=sigma_4/2+sigma_4/20, y=y_limit_max/2, text='4', showarrow=False, font=dict(color='purple'), textangle=0)
        fig.add_annotation(x=sigma_5/2+sigma_5/20, y=y_limit_max/2, text='5', showarrow=False, font=dict(color='blue'), textangle=0)
        fig.add_annotation(x=sigma_6/2+sigma_6/20, y=y_limit_max/2, text='6', showarrow=False, font=dict(color='green'), textangle=0)
                
        # Set x and y axis limits
        fig.update_xaxes(range=[x_limit_min, x_limit_max + x_limit_max*0.2], title_text='Allowable Imprecision (%CV)')
        fig.update_yaxes(range=[y_limit_min, y_limit_max + y_limit_max*0.1], title_text='Allowable Bias (%Bias)')
        # Set title color
        fig.update_layout(title=dict(text='Method decision chart based on conventional sigma-metric', font=dict(color='#cc0000')))
                
        # Show the plot
        st.plotly_chart(fig, use_container_width=True)
        
st.write(" ")
st.markdown("---")
st.write(" ")

st.markdown("**:blue[Normalized method decision chart for comparison of multliple test performances]**")
# Initialize an empty dataframe with the specified number of rows
number_of_rows = 3
df_v2 = pd.DataFrame(
    [{"Test": None, "Bias (%)": None, "Imprecision (%CV)": None, "Total Allowable Error (TEa%)": None} for _ in range(number_of_rows)]
)

# Use st.data_editor to create an editable dataframe
edited_df_v2 = st.data_editor(
    df_v2,
    column_config={
        "Test": st.column_config.TextColumn(
            "Test",
            max_chars=50,
        ),
        "Bias (%)": st.column_config.NumberColumn(
            "Bias (%)",
            help="Bias (%)",
            min_value=0,
            max_value=999999999999999999999999999999999999999,
            format="%g",
        ),
        "Imprecision (%CV)": st.column_config.NumberColumn(
            "Imprecision (%CV)",
            help="Imprecision (%CV)",
            min_value=0.0000000000001,
            max_value=999999999999999999999999999999999999999,
            format="%g",
        ),
        "Total Allowable Error (TEa%)": st.column_config.NumberColumn(
            "Total Allowable Error (TEa%)",
            help="Total Allowable Error (TEa%)",
            min_value=0,
            max_value=999999999999999999999999999999999999999,
            format="%g",
        ),
    },
    hide_index=True, num_rows="dynamic"
)  # An editable dataframe
edited_df_v2 = pd.DataFrame(edited_df_v2)

# Calculate normalized values
edited_df_v2['Normalized Bias'] = 100 * edited_df_v2['Bias (%)'] / edited_df_v2['Total Allowable Error (TEa%)']
edited_df_v2['Normalized CV'] = 100 * edited_df_v2['Imprecision (%CV)'] / edited_df_v2['Total Allowable Error (TEa%)']
edited_df_v2['Sigmametric'] = (edited_df_v2['Total Allowable Error (TEa%)']-edited_df_v2['Bias (%)'])/edited_df_v2['Imprecision (%CV)']
# Set plot limits
x_limit_min_2, x_limit_max_2 = 0, 100  # Assuming CV is a percentage
y_limit_min_2, y_limit_max_2 = 0, 100  # Assuming Bias is a percentage
sigma_22 = x_limit_max_2/2
sigma_33 = x_limit_max_2/3
sigma_44 = x_limit_max_2/4
sigma_55 = x_limit_max_2/5
sigma_66 = x_limit_max_2/6

# Create a scatter plot using plotly express
fig = px.scatter(edited_df_v2, x='Normalized CV', y='Normalized Bias', text = 'Test', title='Normalized OPSpecs Chart')
# Adjust text position
fig.update_traces(textposition='top center')

# Add lines for sigma_2, sigma_3, sigma_4, sigma_5, and sigma_6
fig.add_trace(px.line(x=[x_limit_max_2/2, 0], y=[0, y_limit_max_2], line_shape='linear').data[0].update(line=dict(color='red',width=1)))
fig.add_trace(px.line(x=[x_limit_max_2/3, 0], y=[0, y_limit_max_2], line_shape='linear').data[0].update(line=dict(color='orange', width=1)))
fig.add_trace(px.line(x=[x_limit_max_2/4, 0], y=[0, y_limit_max_2], line_shape='linear').data[0].update(line=dict(color='purple', width=1)))
fig.add_trace(px.line(x=[x_limit_max_2/5, 0], y=[0, y_limit_max_2], line_shape='linear').data[0].update(line=dict(color='blue', width=1)))
fig.add_trace(px.line(x=[x_limit_max_2/6, 0], y=[0, y_limit_max_2], line_shape='linear').data[0].update(line=dict(color='green', width=1)))
    
# Add annotations directly on the lines with adjusted angle
fig.add_annotation(x=sigma_22/2+sigma_22/20, y=y_limit_max_2/2, text='2', showarrow=False, font=dict(color='red'), textangle=0)
fig.add_annotation(x=sigma_33/2+sigma_33/20, y=y_limit_max_2/2, text='3', showarrow=False, font=dict(color='orange'), textangle=0)
fig.add_annotation(x=sigma_44/2+sigma_44/20, y=y_limit_max_2/2, text='4', showarrow=False, font=dict(color='purple'), textangle=0)
fig.add_annotation(x=sigma_55/2+sigma_55/20, y=y_limit_max_2/2, text='5', showarrow=False, font=dict(color='blue'), textangle=0)
fig.add_annotation(x=sigma_66/2+sigma_66/20, y=y_limit_max_2/2, text='6', showarrow=False, font=dict(color='green'), textangle=0)
    
# Set x and y axis limits
fig.update_xaxes(range=[x_limit_min_2, x_limit_max_2/2 + x_limit_max_2*0.2/2], title_text='Normalized Imprecision (Normalized %CV)')
fig.update_yaxes(range=[y_limit_min_2, y_limit_max_2  + y_limit_max_2*0.1], title_text='Normalized Bias (Normalized %Bias)')

# Set title color
fig.update_layout(title=dict(text='Normalized method decision chart based on conventional sigma-metric', font=dict(color='#cc0000')))

# Show the plot
st.plotly_chart(fig, use_container_width=True)
st.dataframe(edited_df_v2[['Test','Sigmametric']],hide_index=True)
