"""
# Change Propagation Method (CPM)
This app describes the Change Propagation Method (CPM) and it lets you play 
with it.
"""
import streamlit as st
import numpy as np
import csv
import io
import zipfile

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
            #st.write(uploaded_file.getvalue())

            # Create a file object from the uploaded file
            file = io.StringIO(uploaded_file.getvalue().decode("utf-8"))

            # Create a CSV reader object
            reader = csv.reader(file)
            
            # Count the number of rows
            row_count = sum(1 for row in reader)
            #st.write(f"Total number of rows in the present file is {row_count}")
            
            # Go back to the start of the file
            file.seek(0)
            
            # Get the first row to determine the number of columns
            header = next(reader)
            col_count = len(header)
            #st.write(f"Total number of columns in the present file is {col_count}")

            if header[0] == 'DSM EXPORTED FROM CAMBRIDGE ADVANCED MODELLER':
                rows_to_skip = 4
            else:
                rows_to_skip = 1

            # Skip the first n rows
            for i in range(rows_to_skip-1):
                header = next(reader)

            # Store the elements of the first column in a list
            product_elements = [row[0] for row in reader if row[0].strip()]

            if header[0] == 'ELEMENT NAME':
                columns_to_skip = 2
                name_column = 0
            elif header[1] == 'ELEMENT NAME':
                columns_to_skip = 3
                name_column = 1

            # Store the elements of the ELEMENT NAME column in a list
            product_elements = [row[name_column] for row in reader if row[name_column].strip()]
            #st.write(product_elements)
            
            # Go back to the start of the file
            file.seek(0)
            
            # Skip the first n rows
            for i in range(rows_to_skip):
                next(reader)

            if col_count - 2 == row_count - 4:
                st.write('This is a simple matrix')
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

            #elif col_count - 2 == int((row_count - 3) / 2):
            else:
                st.write('This is a double matrix')
                dimension = col_count - columns_to_skip
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
                #st.write(direct_likelihood_matrix)
                #st.write(direct_impact_matrix)
                
                design_structure_matrix = [[1 if x != 0 else x for x in sublist] for sublist in direct_impact_matrix]

                # Calculate all matrices
                likelihood_matrix, risk_matrix, impact_matrix = calculate_all_matrices(
                    design_structure_matrix,
                    direct_likelihood_matrix,
                    direct_impact_matrix,
                    change_path_length
                )
                
            #st.write('Likelihood matrix')
            #st.write(np.array(likelihood_matrix))

            #st.write('Impact matrix')
            #st.write(np.array(impact_matrix))

            st.write('Risk matrix')
            st.write(np.array(risk_matrix))

            # Save the list of lists to a CSV text
            file = io.StringIO()
            writer = csv.writer(file)
            writer.writerows(risk_matrix)

            def callback():
                st.balloons()
            # Provide a download button
            st.download_button(
                label="Press to Download",
                data=file.getvalue(),
                file_name="my_csv_file.csv",
                mime="text/csv",
                on_click=callback,
                key='callback'
            )

            st.markdown('---')

        #st.button('Download results')

