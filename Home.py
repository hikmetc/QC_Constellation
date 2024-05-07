# Developed by Hikmet Can Çubukçu

import streamlit as st


st.set_page_config(
    page_title="QC Constellation",
    page_icon=":milky_way:",
    layout="wide"
)

with st.sidebar:
    st.image('./images/QC_Constellation_sidebar.png')
    #st.markdown('---')
    st.info('*Developed by Hikmet Can Çubukçu, MD, MSc, EuSpLM* <hikmetcancubukcu@gmail.com>')
    
#st.write("# :blue[QC Constellation]")
st.image('./images/QC Constellation icon.png')
st.write(" ")

instructions = """
Welcome to QC Constellation, a pioneering web-based application designed to revolutionize quality 
control practices in clinical laboratory environments. The tool is intricately crafted to navigate 
the complexities of modern quality control, enabling laboratory professionals to implement contemporary 
risk-based quality control practices with precision and confidence.

Quality control in clinical laboratories is a multifaceted domain, where error detection spans 
routine quality control rules to sophisticated methods like moving averages. Tools such as sigma-metric 
measurements and method decision charts play a crucial role in evaluating analytical procedures. 
QC Constellation leverages these tools, ensuring their optimal use based on the specific performance 
of analytical methods and the associated patient risks.

QC Constellation incorporates traditional Westgard rules, trend detection methods like EWMA and CUSUM, and 
the Six Sigma methodology for comprehensive performance evaluation. Furthermore, moving averages charts 
adapted for patient-based QC and its optimization tool are both available. Our application's core lies 
in its ability to make these sophisticated practices accessible and actionable for laboratory professionals 
in a risk-based manner.
"""
st.markdown(instructions)
