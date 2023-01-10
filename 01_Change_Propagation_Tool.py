"""
# Change Propagation Method (CPM)
This app describes the Change Propagation Method (CPM) and it lets you play 
with it.
"""
import streamlit as st
import numpy as np

from cpm.cpm import calculate_all_matrices

from helpers.app_helpers import (
    combined_likelihood_matrix,
    combined_risk_matrix,
    combined_impact_matrix
)

# Set wide display, if not done before
try:
    st.set_page_config(
        layout="wide",
        page_title="Change Propagation Method (CPM) Tool",
        page_icon="🧊",)
except:
    pass

st.title('Change Propagation Tool')

# Upload files
st.header('1. Input files')
uploaded_files = st.file_uploader(
    "Files, e.g. dsm01.csv, dsm02.csv, ...",
    type="csv",
    accept_multiple_files=True
)

if ('uploaded_files' in locals()) and (uploaded_files != []):
    st.header('2. Configuration')
    with st.form(key='configuration_form'):
        change_path_length = st.slider(
            'Change path length',
            min_value=1,
            max_value=10,
            value=3,
            step=1,
            help='The maximum number of changes that can be propagated through the product structure.'
            )
        submit_button = st.form_submit_button(label='Apply CPM')
    if submit_button:
        st.header('3. Results')
        st.button('Download results')


# Product elements
product_elements = ['power supply',
                    'motor',
                    'heating unit',
                    'fan',
                    'control system',
                    'casing']

# Design structure Matrix (DSM)
design_structure_matrix =  [[0,1,1,0,0,0],
                            [1,0,0,1,1,1],
                            [1,0,0,1,1,1],
                            [1,1,1,0,0,1],
                            [0,0,1,1,0,1],
                            [1,1,1,1,1,0]]

# Direct likelihood matrix (l)
direct_likelihood_matrix = [[0.0,0.3,0.3,0.0,0.0,0.0],
                            [0.9,0.0,0.0,0.6,0.3,0.6],
                            [0.9,0.0,0.0,0.6,0.3,0.6],
                            [0.3,0.6,0.9,0.0,0.0,0.9],
                            [0.0,0.0,0.3,0.6,0.0,0.3],
                            [0.3,0.9,0.6,0.9,0.6,0.0]]

# Direct impact matrix (i)
direct_impact_matrix =     [[0.0,0.9,0.9,0.0,0.0,0.0],
                            [0.9,0.0,0.0,0.6,0.3,0.3],
                            [0.6,0.0,0.0,0.3,0.3,0.3],
                            [0.3,0.3,0.6,0.0,0.0,0.3],
                            [0.0,0.0,0.3,0.3,0.0,0.3],
                            [0.3,0.6,0.6,0.9,0.6,0.0]]

cutoff = 3

# Calculate all matrices
likelihood_matrix, risk_matrix, impact_matrix = calculate_all_matrices(
    design_structure_matrix,
    direct_likelihood_matrix,
    direct_impact_matrix,
    cutoff
)

st.write('Likelihood matrix')
st.write(np.array(likelihood_matrix))

st.write('Risk matrix')
st.write(np.array(risk_matrix))

st.write('Impact matrix')
st.write(np.array(impact_matrix))


