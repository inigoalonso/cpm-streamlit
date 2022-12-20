"""
# Change Propagation Method (CPM)
This app describes the Change Propagation Method (CPM) and it lets you play with it.
"""
import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pyvis import network as net

from helpers.app_helpers import ppm, combined_likelihood_matrix, combined_risk_matrix, combined_impact_matrix, plot_product_risk_matrix

# Set wide display, if not done before
try:
    st.set_page_config(layout="wide", page_title="Change Propagation Method (CPM)")
except:
    pass

def make_grid(cols,rows):
    grid = [0]*cols
    for i in range(cols):
        with st.container():
            grid[i] = st.columns(rows)
    return grid

st.title('Change Propagation')

# Upload files
uploaded_files = st.file_uploader(
    "Files, e.g. matrices.csv",
    type="csv",
    accept_multiple_files=True
)

if ('uploaded_files' in locals()) and (uploaded_files != []):
    st.write(f"{len(uploaded_files)} files uploaded")
    for file in uploaded_files:
        df_file = pd.read_csv(file)
        check_boxes = st.checkbox(file.name, key=file.id)
        if check_boxes:
            st.write(file.id)


with st.expander('Definitions'):
    st.markdown('''
    $a$: destination of change

    $b$: source of change

    $u$: pen-ultimate sub-system i the chain from **a** to **b**

    $\sigma_{u,a}$: Likelihood of change reaching sub-system **u** from **a**

    $l_{b,u}$: Direct likelihood of change propagation from  **u** to **b**

    Risk of change propagating from **u** to **b**:

    $\rho_{b,u} = \sigma_{u,a} l_{b,u} i_{b,u}$

    Combined risk of change propagating from **b** to **a**:

    $R_{b,a} = 1- \prod{(1 - \rho_{b,u})}$
    ''')

st.header('1. Initial Analysis')
st.subheader('1.1. Create Product Model')


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

from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder

gb = GridOptionsBuilder.from_dataframe(A)

gb.configure_default_column(min_column_width = 2, resizable = False, filterable = False, sorteable = False, editable = True, groupable = False)

gb.configure_grid_options(columnHoverHighlight=True)

gridOptions = gb.build()

AgGrid(
    A,
    gridOptions=gridOptions,
    enable_enterprise_modules=True,
    allow_unsafe_jscode=True
)

col1, col2 = st.columns(2)

with col1:
    st.markdown('''
        a. power supply \n
        b. motor \n
        c. heating unit \n
        d. fan \n
        e. control system \n
        f. casing
    ''')


with col2:
    # fig, ax = plt.subplots()
    # pos = nx.kamada_kawai_layout(G)
    # nx.draw(G,pos, with_labels=True)
    # st.pyplot(fig)

    g=net.Network(height='400px', width='400px',directed=True)
    g.from_nx(G)
    g.save_graph('example.html')
    HtmlFile = open("example.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    st.components.v1.html(source_code, height = 400,width=400)


with st.expander("DSM = "):
    mygrid_DSM = make_grid(len(DSM),len(DSM[0]))
    for i,row in enumerate(DSM):
        for j,cell in enumerate(row):
            mygrid_DSM[i][j].write(cell)

st.subheader('1.2. Complete Dependency Matrices')
st.markdown('''
Within the DSM the column headings show instigating sub-systems and the row headings the affected sub-systems, whose designs change as a result of change to the instigating sub-systems.
''')

# Direct likelihood matrix (l)
direct_likelihood_matrix = [[0.0,0.3,0.3,0.0,0.0,0.0],
                            [0.9,0.0,0.0,0.6,0.3,0.6],
                            [0.9,0.0,0.0,0.6,0.3,0.6],
                            [0.3,0.6,0.9,0.0,0.0,0.9],
                            [0.0,0.0,0.3,0.6,0.0,0.3],
                            [0.3,0.9,0.6,0.9,0.6,0.0]]
with st.expander("Direct likelihood matrix (l) = "):
    mygrid_direct_likelihood_matrix = make_grid(len(direct_likelihood_matrix),len(direct_likelihood_matrix[0]))
    for i,row in enumerate(direct_likelihood_matrix):
        for j,cell in enumerate(row):
            mygrid_direct_likelihood_matrix[i][j].write(cell)

# Direct impact matrix (i)
direct_impact_matrix = [[0.0,0.9,0.9,0.0,0.0,0.0],
                        [0.9,0.0,0.0,0.6,0.3,0.3],
                        [0.6,0.0,0.0,0.3,0.3,0.3],
                        [0.3,0.3,0.6,0.0,0.0,0.3],
                        [0.0,0.0,0.3,0.3,0.0,0.3],
                        [0.3,0.6,0.6,0.9,0.6,0.0]]
with st.expander("Direct impact matrix (i) = "):
    mygrid_direct_impact_matrix = make_grid(len(direct_impact_matrix),len(direct_impact_matrix[0]))
    for i,row in enumerate(direct_impact_matrix):
        for j,cell in enumerate(row):
            mygrid_direct_impact_matrix[i][j].write(cell)

st.write("Calculate the direct risk matrix: r = l * i")

# Direct risk matrix (r)
#TODO make into function
direct_risk_matrix = (np.array(direct_likelihood_matrix)*np.array(direct_impact_matrix)).tolist()
with st.expander("Direct risk matrix (r) = "):
    mygrid_direct_risk_matrix = make_grid(len(direct_risk_matrix),len(direct_risk_matrix[0]))
    for i,row in enumerate(direct_risk_matrix):
        for j,cell in enumerate(row):
            mygrid_direct_risk_matrix[i][j].write(cell)


st.subheader('1.3. Compute Predictive Matrices')
st.markdown('''
Source of change (initiating or instigating component): $a$

Target/Destination: $b$

The combined likelihood algorithm views propagation trees as logic trees. Vertical lines are mathematically represented by $\cup$, while horizontal lines are represented by $\cap$. For each tree, the And/Or summation begins at the bottom, farthest from the instigating subsystem. Through a combination of And and Or evaluations, a single combined likelihood value can be computed at the top of the tree. Since the events are not mutually exclusive the summations take the form:

*And* function as a product of probabilities:

$l_{b,u} \cup l_{b,v} = l_{b,u} \times l_{b,v}$

*Or* function as a sum of probabilities minus the product term (the inverse of the product of inverse probabilities):

$l_{b,u} \cap l_{b,v} = l_{b,u} + l_{b,v} - (l_{b,u} \times l_{b,v}) = 1 - ((1 - l_{b,u}) \times (1 - l_{b,v}))$
''')

# Combined likelihood matrix (L)
clm = combined_likelihood_matrix(DSM,direct_likelihood_matrix)
with st.expander("Combined likelihood matrix (L) = "):
    mygrid_clm = make_grid(len(clm),len(clm[0]))
    for i,row in enumerate(clm):
        for j,cell in enumerate(row):
            mygrid_clm[i][j].write(cell)

# Combined risk matrix (R)
crm = combined_risk_matrix(DSM,direct_likelihood_matrix,direct_impact_matrix)
with st.expander("Combined risk matrix (R) = "):
    mygrid_crm = make_grid(len(crm),len(crm[0]))
    for i,row in enumerate(crm):
        for j,cell in enumerate(row):
            mygrid_crm[i][j].write(cell)

# Combined impact matrix (I) combined_impact_matrix
cim = combined_impact_matrix(DSM,clm,crm)
with st.expander("Combined impact matrix (I) = "):
    mygrid_cim = make_grid(len(cim),len(cim[0]))
    for i,row in enumerate(cim):
        for j,cell in enumerate(row):
            mygrid_cim[i][j].write(cell)

plot_product_risk_matrix(product_components,DSM,clm,cim,crm)