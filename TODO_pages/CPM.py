"""
# Change Propagation Method
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Set sidebar elements
with st.sidebar:
    st.header("Inputs")
    # Example files
    with st.expander("Example csv files"):
        st.caption("You can use the following example files to test the app:")
        c1, c2, c3 = st.columns([1,1,1])
        c1.download_button(
            label="designs",
            data='''"design","scenario","sv"
    "Test1","s1",130
    "Test2","s1",220
    "Test3","s1",33
    "Test4","s1",60
    "Test5","s1",70
    "Test1","s2",100
    "Test2","s2",20
    "Test3","s2",74
    "Test4","s2",6
    "Test5","s2",110
    "Test1","s3",150
    "Test2","s3",50
    "Test3","s3",133
    "Test4","s3",260
    "Test5","s3",75''',
            file_name="designs.csv",
            mime="text/csv"
        )
        c2.download_button(
            label="s1_s2",
            data='''"design","Test1","Test2","Test3","Test4","Test5"
    "Test1",1,1,1,0,0
    "Test2",1,1,1,1,0
    "Test3",1,1,1,0,1
    "Test4",0,1,0,1,1
    "Test5",1,0,0,1,1''',
            file_name="s1_s2.csv",
            mime="text/csv"
        )
        c3.download_button(
            label="s2_s3",
            data='''"design","Test1","Test2","Test3","Test4","Test5"
    "Test1",1,1,1,0,1
    "Test2",0,1,1,1,0
    "Test3",0,1,1,1,1
    "Test4",0,1,0,1,1
    "Test5",1,0,0,1,1''',
            file_name="s2_s3.csv",
            mime="text/csv"
        )
    # Upload designs file
    uploaded_files_1 = st.file_uploader(
        "Designs file, e.g. designs.csv",
        type="csv",
        accept_multiple_files=False
    )
    # When the designs file is uploaded, show the additional uploader
    if uploaded_files_1 is not None:
        df_designs = pd.read_csv(uploaded_files_1)
        # Ordered list of unique scenarios
        scenarios = df_designs['scenario'].unique()
        # Ordered list of unique designs
        designs = df_designs['design'].unique()
        expected_files = []
        for i, scenario in enumerate(scenarios):
            if i < len(scenarios)-1:
                expected_files.append(f"{scenario}_{scenarios[i+1]}.csv")
        uploaded_files_2 = st.file_uploader(
            f"Transitions files, e.g. {expected_files}",
            type="csv",
            accept_multiple_files=True
        )

# Main content


st.title('Change Propagation Method (CPM)')
st.caption("Calculate the change propagation")

st.header('References')

st.markdown('''
- [CPM: A Method for Identifying and Prioritizing Change Propagation in Software Systems](https://ieeexplore.ieee.org/document/1163715)
- [Change Propagation Method (CPM) - Software Engineering Stack Exchange](https://softwareengineering.stackexchange.com/questions/247652/change-propagation-method-cpm)
''')



st.title("VWFO app")
with st.expander("Help"):
    # Help section
    mkd = """
    This app allows you calculate the VWFO of a set of designs.
    For example, if you provide:
    * Set of Designs: `designs.csv`
    * Available transtions between designs: `s1_s2.csv`, `s2_s3.csv`
    You will get the following file on the output:
    * `designs_vwfo.csv`
    * **Inputs 1**: 
        * CSV file: upload the CSV file with your set of designs.
    * **Inputs 2**: 
        * CSV file(s): upload the CSV files with the available transitions between scenarios.
    * **Output Configuration**: 
        * Encoding: select the encoding for the csv files.
        * Separator: select the separator for the csv files. 
    * **Outputs**: 
        * A zip file containing the csv file.
    """
    st.markdown(mkd)
# Placeholder for the reminder to upload files
message_missing_designs = st.empty()

# Display the inputs if the designs are uploaded
if uploaded_files_1 is not None:
    st.markdown(f"#### {len(designs)} Designs and {len(scenarios)} Scenarios")
    st.caption(
        f"Detected {len(designs)} designs {designs} and {len(scenarios)} scenarios {scenarios}"
    )

    # Plot the designs' surplus values for each scenario
    with st.expander("Designs' surplus values per scenario", expanded=True):
        fig = px.line(df_designs, x="scenario", y="sv", color='design')
        st.plotly_chart(fig, use_container_width=True)

    # Placeholder for the reminder to upload files
    message_missing_transitions = st.empty()

# Display the inputs if the transitions are uploaded
if (uploaded_files_1 is not None) and ('uploaded_files_2' in locals()) and (uploaded_files_2 != []):
    # Display the inputs for transitions
    st.markdown("#### Design transitions")
    st.caption("Available transitions between designs")
    # Plot the designs' transitions between scenarios
    with st.expander("Transition paths between scenarios", expanded=True):
        st.write("TODO: plot the transitions between scenarios")
    # Display outputs
    st.markdown("#### Output")
    # Format columns
    c1, c2 = st.columns([1,1])
    # Select current scenario
    current_scenario = c1.selectbox(
        "Select current scenario",
        options=scenarios
    )
    # Select following scenario
    following_scenario = c2.selectbox(
        "Select following scenario",
        options=np.delete(scenarios, np.where(scenarios==current_scenario))
    )
    # Read the transition matrix for the selected scenarios
    for file in uploaded_files_2:
        if file.name == f"{current_scenario}_{following_scenario}.csv":
            df_transition = pd.read_csv(file)
            df_transition.set_index('design', inplace=True)
    # Select an individual design
    # selected_design = c1.selectbox(
    #     "Select design",
    #     options=designs
    # )
    # selected_design_sv = df_designs[
    #     df_designs['design'] == selected_design
    # ]['sv'].values
    # selected_design_current_sv = df_designs[
    #     (df_designs['design'] == selected_design) &
    #     (df_designs['scenario'] == current_scenario)
    # ]['sv'].values[0]
    # selected_design_following_sv = df_designs[
    #     (df_designs['design'] == selected_design) &
    #     (df_designs['scenario'] == following_scenario)
    # ]['sv'].values[0]
    # Read the following designs from the transition matrix
    # following_designs = [
    #     k for k,v in df_transition.loc[selected_design].to_dict().items() if v == 1
    # ]
    # following_designs_sv = df_designs[
    #     (df_designs['design'].isin(following_designs)) &
    #     (df_designs['scenario'] == following_scenario)
    #     ]['sv'].values
    # Calculate the VWFO for the selected design
    # selected_design_current_vwfo = calculate_vwfo(
    #     len(designs),
    #     selected_design_current_sv,
    #     following_designs_sv
    # )
    # c2.write(
    #     f"VWFO of design {selected_design}: {selected_design_current_vwfo}"
    # )
    # Calculate the VWFO for all designs
    vwfos = []
    for design in designs:
        design_current_sv = df_designs[
            (df_designs['design'] == design) &
            (df_designs['scenario'] == current_scenario)
        ]['sv'].values[0]
        design_following_sv = df_designs[
            (df_designs['design'] == design) &
            (df_designs['scenario'] == following_scenario)
        ]['sv'].values[0]
        following_designs = [
            k for k,v in df_transition.loc[design].to_dict().items() if v == 1
        ]
        following_designs_sv = df_designs[
            (df_designs['design'].isin(following_designs)) &
            (df_designs['scenario'] == following_scenario)
        ]['sv'].values
        design_current_vwfo = calculate_vwfo(
            len(designs),
            design_current_sv,
            following_designs_sv
        )
        #st.write(f"VWFO of design {design}: {design_current_vwfo}")
        vwfos.append(design_current_vwfo)
    # Plot the VWFO of all designs
    # df_tips = px.data.tips()
    # fig_tips = px.violin(df_tips, y="total_bill")
    df_vwfos = pd.DataFrame({
        'design': designs,
        'vwfo': vwfos
    })
    fig_tips = px.violin(df_vwfos, y="vwfo")
    st.plotly_chart(fig_tips, use_container_width=True)

# Reminders to upload files
if 'uploaded_files_2' in locals():
    if uploaded_files_2 == []:
        # If something is not working, stop here...
        message_missing_transitions.error("Upload transitions files to continue")
if uploaded_files_1 is None:
    # If something is not working, stop here...
    message_missing_designs.error("Upload designs file to continue")