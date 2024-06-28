"""
# Change Propagation Method (CPM)
This app describes the Change Propagation Method (CPM) and it lets you play 
with it.
"""
import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import csv
import io
import zipfile
import time
import plotly.express as px
import plotly.graph_objects as go
from ragraph import plot
from ragraph.io.matrix import from_matrix

from typing import Union
from cpm.parse import parse_csv
from cpm.models import ChangePropagationTree

# Set wide display, if not done before
try:
    st.set_page_config(
        layout="wide",
        page_title="Change Propagation Method (CPM) Tool",
        page_icon="🧊",
        initial_sidebar_state="expanded",)
except:
    pass

# Hide the menu and the footer
# Add header {visibility: hidden;}
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

#st.write("Streamlit version:", st.__version__)

st.title('Change Propagation Tool')

# Upload files
st.header('Inputs')

# Example files
with st.expander("Example csv files"):
    st.caption("You can use the following example files to test the app:")
    c1, c2 = st.columns([1,1])
    with open("./inputs/dsm-impact.csv", "r") as f:
        c1.download_button(
            label="DSM Impact",
            data=f,
            file_name="dsm-impact.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with open("./inputs/dsm-likelihood.csv", "r") as f:
        c2.download_button(
            label="DSM Likelihood",
            data=f,
            file_name="dsm-likelihood.csv",
            mime="text/csv",
            use_container_width=True,
        )
with st.expander("Inputs", expanded=True):
    with st.form(key='upload_form'):
        # three columns for the file upload
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            uploaded_dsm_impact = st.file_uploader(
                "Files, e.g. dsm-impact.csv",
                type="csv",
                accept_multiple_files=False,
                key='file_uploader_impact',
                help='Upload the file containing the Impact DSM.'
            )
        with col2:
            uploaded_dsm_likelihood = st.file_uploader(
                "Files, e.g. dsm-likelihood.csv",
                type="csv",
                accept_multiple_files=False,
                key='file_uploader_likelihood',
                help='Upload the file containing the Likelihood DSM.'
            )
        with col3:
            change_path_length = st.slider(
                'Change path length',
                min_value=1,
                max_value=10,
                value=3,
                step=1,
                help='The maximum number of changes that can be propagated through the product structure.',
                key='change_path_length',
            )
            submit_button = st.form_submit_button(
                "Run CPM",
                use_container_width=True,
                type='primary',
            )
if (uploaded_dsm_impact != None and uploaded_dsm_likelihood != None):
    st.header('Results')
    results_container = st.container()

if ('uploaded_dsm_impact' in locals()) and (uploaded_dsm_impact != None) and ('uploaded_dsm_likelihood' in locals()) and (uploaded_dsm_likelihood != None):
    #print('uploaded_files:', [file.name for file in uploaded_files])
    result_files = {}

    #st.subheader(f"{file_number} {uploaded_file.name}")
    #st.write(uploaded_file.getvalue())

    # Create a file objects from the uploaded files
    file_impact = io.StringIO(uploaded_dsm_impact.getvalue().decode("utf-8"))
    file_likelihood = io.StringIO(uploaded_dsm_likelihood.getvalue().decode("utf-8"))
    
    # Run change propagation on entire matrix

    # Create DSMs for Impacts and Likelihoods
    dsm_i = parse_csv(file_impact)
    dsm_l = parse_csv(file_likelihood)

    # Create a matrix in which the results can be stored
    res_mtx: list[list[Union[float, str]]] = []
    for i, icol in enumerate(dsm_l.columns):
        res_mtx.append([icol])

        for j, jcol in enumerate(dsm_l.columns):
            # Run change propagation on each possible pairing
            cpt = ChangePropagationTree(j, i, dsm_impact=dsm_i, dsm_likelihood=dsm_l)
            cpt.propagate(search_depth=4)
            # Store results in matrix
            res_mtx[i].append(cpt.get_risk())

    # 

    st.code(res_mtx)

    # plot the dsm_i matrix witht the 

    # Create CSV string
    delimiter = "; "
    csv = "\t"+delimiter
    csv += delimiter.join(dsm_l.columns) + "\n"
    for line in res_mtx:
        csv_line = delimiter.join(map(str, line))

        csv_line += "\n"
        csv += csv_line

    # Write to file
    with open("cpm.csv", "w") as file:
        file.write(csv)



    st.write("Change Propagation Matrix")
    st.write(res_mtx)
    st.write("CSV file created")
    st.write(csv)


    g = from_matrix(dsm_i.matrix)

    fig = plot.mdm(
        leafs=g.leafs,
        edges=g.edges,
    )
    st.plotly_chart(fig)

# Display the session state
#st.write('Session state: ',st.session_state)