"""
# Design Of Experiments (DOE)
This app ...
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
        page_title="Design Of Experiments (DOE) Tool",
        page_icon="🧊",)
except:
    pass

st.title('Design Of Experiments (DOE) Tool')
