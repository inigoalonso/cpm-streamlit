"""
# Change Propagation Method (CPM)
This app describes the Change Propagation Method (CPM) and it lets you play 
with it.
"""
import streamlit as st
import numpy as np
import csv
import io

from cpm.cpm import calculate_all_matrices

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
        for uploaded_file in uploaded_files:
            st.subheader(uploaded_file.name)
            st.write(uploaded_file.getvalue())

            # Create a file object from the uploaded file
            file = io.StringIO(uploaded_file.getvalue().decode("utf-8"))

            # Create a CSV reader object
            reader = csv.reader(file)
            
            # Count the number of rows
            row_count = sum(1 for row in reader)
            st.write(f"Total number of rows in the present file is {row_count}")
            
            # Go back to the start of the file
            file.seek(0)
            
            # Get the first row to determine the number of columns
            header = next(reader)
            col_count = len(header)
            st.write(f"Total number of columns in the present file is {col_count}")
            
            # Go back to the start of the file
            file.seek(0)

            # Skip the first 4 rows
            for i in range(4):
                next(reader)
            
            # Store the elements of the first column in a list
            product_elements = [row[0] for row in reader if row[0].strip()]
            #st.write(product_elements)
            
            # Go back to the start of the file
            file.seek(0)
                
            # Skip the first 4 rows
            for i in range(4):
                next(reader)

            if col_count - 2 == row_count - 4:
                st.write('This is a simple matrix')
                dimension = col_count - 2
                matrix = np.zeros((dimension, dimension))
                for i, row in enumerate(reader):
                    for j, element in enumerate(row[2:]):
                        if element.strip():
                            matrix[i, j] = float(element)
                st.write(matrix)

                design_structure_matrix = [[1 if x != 0 else x for x in sublist] for sublist in matrix]

                # Calculate all matrices
                likelihood_matrix, risk_matrix, impact_matrix = calculate_all_matrices(
                    design_structure_matrix,
                    matrix,
                    matrix,
                    change_path_length
                )

            elif col_count - 2 == int((row_count - 3) / 2):
                st.write('This is a double matrix')
                dimension = col_count - 2
                direct_likelihood_matrix = np.zeros((dimension, dimension))
                direct_impact_matrix = np.zeros((dimension, dimension))
                for i, row in enumerate(reader):
                    if i % 2 == 0:
                        for j, element in enumerate(row[2:]):
                            if element.strip():
                                direct_likelihood_matrix[int(i/2), j] = float(element)
                    else:
                        for j, element in enumerate(row[2:]):
                            if element.strip():
                                direct_impact_matrix[int(i/2), j] = float(element)
                st.write(direct_likelihood_matrix)
                st.write(direct_impact_matrix)
                
                design_structure_matrix = [[1 if x != 0 else x for x in sublist] for sublist in direct_impact_matrix]

                # Calculate all matrices
                likelihood_matrix, risk_matrix, impact_matrix = calculate_all_matrices(
                    design_structure_matrix,
                    direct_likelihood_matrix,
                    direct_impact_matrix,
                    change_path_length
                )
                
            st.write('Likelihood matrix')
            st.write(np.array(likelihood_matrix))

            st.write('Risk matrix')
            st.write(np.array(risk_matrix))

            st.write('Impact matrix')
            st.write(np.array(impact_matrix))

            st.markdown('---')

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
# likelihood_matrix, risk_matrix, impact_matrix = calculate_all_matrices(
#     design_structure_matrix,
#     direct_likelihood_matrix,
#     direct_impact_matrix,
#     cutoff
# )

#st.write('Likelihood matrix')
#st.write(np.array(likelihood_matrix))

#st.write('Risk matrix')
#st.write(np.array(risk_matrix))

#st.write('Impact matrix')
#st.write(np.array(impact_matrix))


