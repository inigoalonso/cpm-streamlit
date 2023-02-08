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

from cpm.cpm import calculate_all_matrices

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

with st.sidebar:
    # Upload files
    st.header('Inputs')
    with st.form(key='upload_form'):
        uploaded_files = st.file_uploader(
            "Files, e.g. dsm01.csv, dsm02.csv, ...",
            type="csv",
            accept_multiple_files=True,
            key='file_uploader',
            help='Upload the files containing the product structure matrices.'
        )
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
            "Run CPM"
        )
    if (uploaded_files != []):
        st.header('Results')
        results_container = st.container()
        if "tabs" not in st.session_state:
            st.session_state["tabs"] = [file.name for file in uploaded_files]

if ('uploaded_files' in locals()) and (uploaded_files != []):
    #print('uploaded_files:', [file.name for file in uploaded_files])
    result_files = {}
    tabs = st.tabs(
        st.session_state["tabs"]
    )
    display_option = ['' for tab in tabs]
    for file_number, uploaded_file in enumerate(uploaded_files):
        with tabs[file_number]:
            #st.subheader(f"{file_number} {uploaded_file.name}")
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

            if header[0] == 'ELEMENT NAME':
                rows_to_skip = 1
                columns_to_skip = 2
                name_column = 0
            elif header[1] == 'ELEMENT NAME':
                rows_to_skip = 1
                columns_to_skip = 3
                name_column = 1

            if header[1] == '':
                rows_to_skip = 4
                # Skip the first n rows
                for i in range(rows_to_skip-1):
                    header = next(reader)

                if header[0] == 'ELEMENT NAME':
                    columns_to_skip = 2
                    name_column = 0
                elif header[1] == 'ELEMENT NAME':
                    columns_to_skip = 3
                    name_column = 1
            else:
                rows_to_skip = 1
                # Skip the first n rows
                for i in range(rows_to_skip-1):
                    header = next(reader)

                if header[0] == 'ELEMENT NAME':
                    columns_to_skip = 2
                    name_column = 0
                elif header[1] == 'ELEMENT NAME':
                    columns_to_skip = 3
                    name_column = 1
            
            # Calculate the dimension of the matrix
            dimension = col_count - columns_to_skip
            
            # Store the elements of the first column in a list
            product_elements = [row[name_column] for row in reader if row[name_column].strip()]

            # st.write(f"rows_to_skip: {rows_to_skip}")
            # st.write(f"columns_to_skip: {columns_to_skip}")
            # st.write(f"name_column: {name_column}")
            # st.write(f"header: {header}")
            # st.write(f"col_count: {col_count}")
            # st.write(f"row_count: {row_count}")
            # st.write(f"dimension: {dimension}")
            # st.write(f"product_elements: {product_elements}")
            
            # Go back to the start of the file
            file.seek(0)
            
            # Skip the first n rows
            for i in range(rows_to_skip):
                next(reader)

            if col_count - 2 == row_count - 4:
                #st.write('This is a simple matrix')
                matrix = np.zeros((dimension, dimension))
                for i, row in enumerate(reader):
                    for j, element in enumerate(row[columns_to_skip:]):
                        if element.strip():
                            #st.write(i, j, element)
                            matrix[i, j] = float(element)
                #st.write(matrix)

                design_structure_matrix = [[1 if x != 0 else x for x in sublist] for sublist in matrix]
                direct_likelihood_matrix = matrix
                direct_impact_matrix = matrix

                # Calculate all matrices
                likelihood_matrix, risk_matrix, impact_matrix = calculate_all_matrices(
                    design_structure_matrix,
                    matrix,
                    matrix,
                    change_path_length
                )

            #elif col_count - 2 == int((row_count - 3) / 2):
            else:
                #st.write('This is a double matrix')
                direct_likelihood_matrix = np.zeros((dimension, dimension))
                direct_impact_matrix = np.zeros((dimension, dimension))
                for i, row in enumerate(reader):
                    if i % 2 == 0:
                        for j, element in enumerate(row[columns_to_skip:]):
                            if element.strip():
                                direct_likelihood_matrix[int(i/2), j] = float(element)
                    else:
                        for j, element in enumerate(row[columns_to_skip:]):
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
            
            datasets = {
                'Combined Risk Matrix': risk_matrix,
                'Direct Likelihood Matrix': direct_likelihood_matrix,
                'Direct Impact Matrix': direct_impact_matrix,
                'Design Structure Matrix': design_structure_matrix,
                'Combined Likelihood Matrix': likelihood_matrix,
                'Combined Impact Matrix': impact_matrix
            }
            display_option[file_number] = st.selectbox(
                'Select matrix to display',
                list(datasets.keys()),
                key="display_"+str(file_number)
            )
            col1, col2 = st.columns([2, 2])
            with col1:
                with st.expander(f'Descriptive statistics of {display_option[file_number]} per product element', expanded=True):
                    df = pd.DataFrame(product_elements, columns=['Element'])
                    df['Mean'] = [np.mean(datasets[display_option[file_number]][i]) for i in range(len(product_elements))]
                    df['Standard Deviation'] = [np.std(datasets[display_option[file_number]][i]) for i in range(len(product_elements))]
                    df['Variance'] = [np.var(datasets[display_option[file_number]][i]) for i in range(len(product_elements))]
                    df['Minimum'] = [np.min(datasets[display_option[file_number]][i]) for i in range(len(product_elements))]
                    df['Maximum'] = [np.max(datasets[display_option[file_number]][i]) for i in range(len(product_elements))]
                    df['Median'] = [np.median(datasets[display_option[file_number]][i]) for i in range(len(product_elements))]
                    st.write(df)
                with st.expander(f'Show {display_option[file_number]}', expanded=False):
                    st.write(np.array(datasets[display_option[file_number]]))
                # Save the list of lists to a CSV text
                file = io.StringIO()
                writer = csv.writer(file)
                writer.writerows(risk_matrix)
                result_files[uploaded_file.name] = file.getvalue()
                # Provide a download button
                st.download_button(
                    label=f"Download Combined Risk Matrix: risk_{uploaded_file.name}",
                    data=file.getvalue(),
                    file_name="risk_"+uploaded_file.name,
                    mime="text/csv",
                    key='download_button_'+str(file_number)
                )

            with col2:
                with st.expander(f'Plot of {display_option[file_number]}', expanded=True):
                    fig = px.imshow(
                        datasets[display_option[file_number]],
                        labels=dict(x="", y=""),
                        x=product_elements,
                        y=product_elements,
                        color_continuous_scale='dense',
                        #title='Combined Risk Matrix',
                        width=800,
                        height=600,
                        #text_auto='.2f',
                        aspect='equal',
                    )
                    fig.update_layout(
                        xaxis={"side": "top"},
                        yaxis={'side': 'left'}  
                    )
                    st.plotly_chart(
                        fig, 
                        use_container_width=True,
                        config={
                            'displaylogo': False,
                            'modeBarButtonsToRemove': ['sendDataToCloud', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d', 'hoverClosestCartesian', 'hoverCompareCartesian', 'toggleSpikelines']
                        }
                    )
                # with st.expander(f'Plot of {display_option[file_number]} as network', expanded=False):
                #     G = nx.from_numpy_matrix(np.array(datasets[display_option[file_number]]))
                #     pos = nx.spring_layout(G)
                #     fig = go.Figure()
                #     fig.add_trace(go.Scatter(
                #         x=[pos[k][0] for k in pos.keys()],
                #         y=[pos[k][1] for k in pos.keys()],
                #         mode='markers',
                #         marker=dict(
                #             size=[100*np.mean(datasets[display_option[file_number]][i]) for i in range(len(product_elements))],
                #             color=[np.mean(datasets[display_option[file_number]][i]) for i in range(len(product_elements))],
                #             colorscale='dense',
                #             showscale=True,
                #         ),
                #         text=[product_elements[i] for i in range(len(product_elements))],
                #         hoverinfo='text',
                #     ))
                #     fig.update_layout(
                #         xaxis=dict(
                #             showgrid=False,
                #             zeroline=False,
                #             showticklabels=False,
                #         ),
                #         yaxis=dict(
                #             showgrid=False,
                #             zeroline=False,
                #             showticklabels=False,
                #         ),
                #         width=800,
                #         height=600,
                #         margin=dict(
                #             l=0,
                #             r=0,
                #             b=0,
                #             t=0,
                #             pad=0
                #         ),
                #         hovermode='closest',
                #         showlegend=False,
                #     )
                #     st.plotly_chart(
                #         fig,
                #         use_container_width=True,
                #         config={
                #             'displaylogo': False,
                #             'modeBarButtonsToRemove': ['sendDataToCloud', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'hoverClosestCartesian', 'hoverCompareCartesian', 'toggleSpikelines']
                #         }
                #     )


    # Create a timestamp
    timestr = time.strftime("%Y%m%d_%H%M%S_")
    # Create a zip file
    with zipfile.ZipFile(f"{timestr}results.zip", mode="w") as archive:
        for file in result_files:
            archive.writestr("risk_"+file, data=result_files[file])
    # Provide a download button
    with open(f"{timestr}results.zip", "rb") as file:
        results_container.download_button(
            label="Download all combined risk matrices",
            data=file,
            file_name=f"{timestr}results.zip",
            mime="application/zip",
        )

# Display the session state
#st.write('Session state: ',st.session_state)