"""
# Change Propagation Method (CPM)
This app describes the Change Propagation Method (CPM) and it lets you play with it.
"""
import streamlit as st
import pandas as pd
import numpy as np

from helpers.app_helpers import ppm, combined_likelihood_matrix, combined_risk_matrix, combined_impact_matrix, plot_product_risk_matrix

# Set wide display, if not done before
try:
    st.set_page_config(layout="wide", page_title="Change Propagation Method (CPM)")
except:
    pass

st.title('Change Propagation')

st.header('Definitions')
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
st.markdown('''
    a. power supply
    b. motor
    c. heating unit
    d. fan
    e. control system
    f. casing
''')
product_components =   ['power supply',
                        'motor',
                        'heating unit',
                        'fan',
                        'control system',
                        'casing']

st.subheader('1.2. Complete Dependency Matrices')
st.markdown('''
Within the DSM the column headings show instigating sub-systems and the row headings the affected sub-systems, whose designs change as a result of change to the instigating sub-systems.
''')
# Design structure Matrix (DSM)
DSM =  [[0,1,1,0,0,0],
        [1,0,0,1,1,1],
        [1,0,0,1,1,1],
        [1,1,1,0,0,1],
        [0,0,1,1,0,1],
        [1,1,1,1,1,0]]
ppm(DSM)

# Direct likelihood matrix (l)
direct_likelihood_matrix = [[0.0,0.3,0.3,0.0,0.0,0.0],
                            [0.9,0.0,0.0,0.6,0.3,0.6],
                            [0.9,0.0,0.0,0.6,0.3,0.6],
                            [0.3,0.6,0.9,0.0,0.0,0.9],
                            [0.0,0.0,0.3,0.6,0.0,0.3],
                            [0.3,0.9,0.6,0.9,0.6,0.0]]
ppm(direct_likelihood_matrix)

# Direct impact matrix (i)
direct_impact_matrix = [[0.0,0.9,0.9,0.0,0.0,0.0],
                        [0.9,0.0,0.0,0.6,0.3,0.3],
                        [0.6,0.0,0.0,0.3,0.3,0.3],
                        [0.3,0.3,0.6,0.0,0.0,0.3],
                        [0.0,0.0,0.3,0.3,0.0,0.3],
                        [0.3,0.6,0.6,0.9,0.6,0.0]]
ppm(direct_impact_matrix)

# Direct risk matrix (r)
#TODO make into function
direct_risk_matrix = (np.array(direct_likelihood_matrix)*np.array(direct_impact_matrix)).tolist()
ppm(direct_risk_matrix)

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
print(clm)

# Combined risk matrix (R)
crm = combined_risk_matrix(DSM,direct_likelihood_matrix,direct_impact_matrix)
print(crm)

# Combined impact matrix (I) combined_impact_matrix
cim = combined_impact_matrix(DSM,clm,crm)
print(cim)

plot_product_risk_matrix(product_components,DSM,clm,cim,crm)