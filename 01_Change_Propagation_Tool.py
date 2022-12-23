"""
# Change Propagation Method (CPM)
This app describes the Change Propagation Method (CPM) and it lets you play 
with it.
"""
import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx

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


# Product components
product_components =   ['power supply',
                        'motor',
                        'heating unit',
                        'fan',
                        'control system',
                        'casing']

# Design structure Matrix (DSM)
DSM =  [[0,1,1,0,0,0],
        [1,0,0,1,1,1],
        [1,0,0,1,1,1],
        [1,1,1,0,0,1],
        [0,0,1,1,0,1],
        [1,1,1,1,1,0]]

A = pd.DataFrame(DSM, index=product_components, columns=product_components)
G = nx.from_pandas_adjacency(A, create_using=nx.DiGraph)

# Direct likelihood matrix (l)
direct_likelihood_matrix = [[0.0,0.3,0.3,0.0,0.0,0.0],
                            [0.9,0.0,0.0,0.6,0.3,0.6],
                            [0.9,0.0,0.0,0.6,0.3,0.6],
                            [0.3,0.6,0.9,0.0,0.0,0.9],
                            [0.0,0.0,0.3,0.6,0.0,0.3],
                            [0.3,0.9,0.6,0.9,0.6,0.0]]

# Direct impact matrix (i)
direct_impact_matrix = [[0.0,0.9,0.9,0.0,0.0,0.0],
                        [0.9,0.0,0.0,0.6,0.3,0.3],
                        [0.6,0.0,0.0,0.3,0.3,0.3],
                        [0.3,0.3,0.6,0.0,0.0,0.3],
                        [0.0,0.0,0.3,0.3,0.0,0.3],
                        [0.3,0.6,0.6,0.9,0.6,0.0]]

# Direct risk matrix (r = l * i)
#TODO make into function
direct_risk_matrix = (np.array(direct_likelihood_matrix)*np.array(direct_impact_matrix)).tolist()

# Combined likelihood matrix (L)
clm = combined_likelihood_matrix(DSM,direct_likelihood_matrix)

# Combined risk matrix (R)
crm = combined_risk_matrix(DSM,direct_likelihood_matrix,direct_impact_matrix)

# Combined impact matrix (I) combined_impact_matrix
cim = combined_impact_matrix(DSM,clm,crm)
