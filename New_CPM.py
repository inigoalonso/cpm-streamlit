"""
# Change Propagation Method (CPM)
This app describes the Change Propagation Method (CPM) and it lets you play 
with it.
"""
import streamlit as st
import csv
import io
import time

from typing import Union
from cpm.parse import parse_csv
from cpm.models import ChangePropagationTree


def calculate_results(dsm_i, dsm_l):
    # Create a matrix in which the results can be stored
    result_matrix: list[list[Union[float, str]]] = []
    for i, icol in enumerate(dsm_l.columns):
        result_matrix.append([icol])

        for j, jcol in enumerate(dsm_l.columns):
            # Run change propagation on each possible pairing
            cpt = ChangePropagationTree(j, i, dsm_impact=dsm_i, dsm_likelihood=dsm_l)
            cpt.propagate(search_depth=4)
            # Store results in matrix
            result_matrix[i].append(cpt.get_risk())
    return result_matrix

# Set wide display, if not done before
try:
    st.set_page_config(
        layout="wide",
        page_title="Change Propagation Method (CPM) Tool",
        page_icon="🧮",
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

# st.toast(f"Streamlit version: {st.__version__}", icon="ℹ️")

st.title('Change Propagation Tool')

# Inputs
st.header('Inputs')

# Example files
with st.expander("Example csv files"):
    st.caption("You can use the following example files to test the app:")
    c1, c2 = st.columns([1,1])
    with open("./inputs/dsm-impact.csv", "r") as file:
        c1.download_button(
            label="DSM Impact",
            data=file,
            file_name="dsm-impact.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with open("./inputs/dsm-likelihood.csv", "r") as file:
        c2.download_button(
            label="DSM Likelihood",
            data=file,
            file_name="dsm-likelihood.csv",
            mime="text/csv",
            use_container_width=True,
        )

# Upload files
with st.expander("Inputs", expanded=True):
    with st.form(key='upload_form'):
        # three columns for the file upload
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.markdown('#### Impact DSM')
            uploaded_dsm_impact = st.file_uploader(
                "Files, e.g. dsm-impact.csv",
                type="csv",
                accept_multiple_files=False,
                key='file_uploader_impact',
                help='Upload the file containing the Impact DSM.',
            )
        with col2:
            st.markdown('#### Likelihood DSM')
            uploaded_dsm_likelihood = st.file_uploader(
                "Files, e.g. dsm-likelihood.csv",
                type="csv",
                accept_multiple_files=False,
                key='file_uploader_likelihood',
                help='Upload the file containing the Likelihood DSM.'
            )
        with col3:
            st.markdown('#### Options')
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

# Results

if ('uploaded_dsm_impact' in locals()) and (uploaded_dsm_impact != None) and ('uploaded_dsm_likelihood' in locals()) and (uploaded_dsm_likelihood != None):

    # Create file objects from the uploaded files
    file_impact = io.StringIO(uploaded_dsm_impact.getvalue().decode("utf-8"))
    file_likelihood = io.StringIO(uploaded_dsm_likelihood.getvalue().decode("utf-8"))
    
    # Run change propagation on entire matrix

    # Create DSMs for Impacts and Likelihoods
    dsm_i = parse_csv(file_impact)
    dsm_l = parse_csv(file_likelihood)
    
    result_matrix = calculate_results(dsm_i, dsm_l)

    # Create CSV string
    delimiter = "; "
    csv = "  "+delimiter
    csv += delimiter.join(dsm_l.columns) + "\n"
    for line in result_matrix:
        csv_line = delimiter.join(map(str, line))
        csv_line += "\n"
        csv += csv_line

    # Timestamp for the file name
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_filename = f"cpm-{timestamp}.csv"

    with st.expander("Results", expanded=True):
        # Display the download button
        st.download_button(
            label=f"Download {results_filename}",
            data=csv,
            file_name=results_filename,
            mime="text/csv",
            key='download_button',
            help='Download the results as a CSV file.',
            use_container_width=True,
            type='primary',
        )
