import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objs as go
import subprocess
import sys
import io

# Initialize session state for the theme and file uploads
# if 'dark_mode' not in st.session_state:
#     st.session_state.dark_mode = True

# if 'members_file_content' not in st.session_state:
#     st.session_state.members_file_content = None

# if 'votes_file_content' not in st.session_state:
#     st.session_state.votes_file_content = None

# if 'merged_df' not in st.session_state:
#     st.session_state.merged_df = None

# if 'graphed' not in st.session_state:
#     st.session_state.graphed = False

# if 'dem_color' not in st.session_state:
#     st.session_state.dem_color = '#003862'

# if 'rep_color' not in st.session_state:
#     st.session_state.rep_color = '#C00000'

# if 'graph_background_color' not in st.session_state:
#     st.session_state.graph_background_color = '#ffffff'

# if 'node_border_color' not in st.session_state:
#     st.session_state.node_border_color = '#000000'

# if 'cpc_threshold' not in st.session_state:
#     st.session_state.cpc_threshold = '0.5'

# if 'within_party_threshold' not in st.session_state:
#     st.session_state.within_party_threshold = '0.75'

# if 'drop_bills' not in st.session_state:
#     st.session_state.drop_bills = False

# # Toggle button for light/dark mode
# def toggle_mode():
#     st.session_state.dark_mode = not st.session_state.dark_mode

def is_color_dark(color):
    hex_color = color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    # Calculate the brightness (luminance) of the color
    brightness = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return brightness < 0.5

def get_border_style(color):
    if is_color_dark(color):
        return "2px solid white"
    else:
        return "2px solid black"

# # Set the background color and text color based on the mode
# def set_theme():
#     if st.session_state.dark_mode:
#         background_color = '#333333'
#         text_color = '#FFFFFF'
#         button_background = '#555555'
#         button_text = '#FFFFFF'
#     else:
#         background_color = '#FFFFFF'
#         text_color = '#000000'
#         button_background = '#DDDDDD'
#         button_text = '#000000'
    
#     return background_color, text_color, button_background, button_text

# background_color, text_color, button_background, button_text = set_theme()

# # Apply the background color and text color to the page
# page_bg = f"""
# <style>
# [data-testid="stAppViewContainer"] {{
#     background-color: {background_color};
#     color: {text_color};
# }}
# [data-testid="stHeader"] {{
#     background-color: {background_color};
# }}
# [data-testid="stSidebar"] {{
#     background-color: {background_color};
#     color: {text_color};
#     border: 2px solid {text_color};  /* Add border to sidebar */
# }}
# [data-testid="stVerticalBlock"] {{
#     color: {text_color};
# }}
# [data-testid="stHorizontalBlock"] {{
#     color: {text_color};
# }}
# [data-testid="stMarkdownContainer"] h1, 
# [data-testid="stMarkdownContainer"] h2, 
# [data-testid="stMarkdownContainer"] h3, 
# [data-testid="stMarkdownContainer"] p {{
#     color: {text_color};
# }}
# /* Ensure the close button is visible */
# .css-1rs6os edgvbvh10 {{
#     color: {text_color};
# }}
# /* Add a margin to the top of the sidebar */
# [data-testid="stSidebar"] > div:first-child {{
#     margin-top: 20px;
# }}
# button {{
#     background-color: {button_background} !important;
#     color: {button_text} !important;
#     border: 1px solid {text_color} !important;
#     padding: 0.5em 1em !important;
#     border-radius: 5px !important;
#     font-size: 1em !important;
# }}
# </style>
# """
# st.markdown(page_bg, unsafe_allow_html=True)

# # Sidebar content
# with st.sidebar:
#     st.header("Menu")
#     if st.button("Toggle Light/Dark Mode", key="toggle_button"):
#         toggle_mode()
#         st.experimental_rerun()

st.title('With Honor - Graphical Analysis')

tab1, tab2, tab3, tab4 = st.tabs(["Settings & Documentation", "Data Consolidation", "Graphing", "Test"])

with tab1:
    # Pick colors
    dem_color = st.color_picker("Pick a color to represent Democrats on the Graph (Default Value is '#003862')", st.session_state.dem_color)
    rep_color = st.color_picker("Pick a color to represent Republicans on the Graph (Default Value is '#C00000')", st.session_state.rep_color)
    graph_background_color = st.color_picker("Pick a color for the background of the graph (Default Value is '#ffffff')", st.session_state.graph_background_color)
    node_border_color = st.color_picker("Pick a color for the node borders on the graphs (Default Value is '#000000')", st.session_state.node_border_color)

    # # Apply border styles
    # dem_border = get_border_style(dem_color)
    # rep_border = get_border_style(rep_color)
    # background_border = get_border_style(st.session_state.graph_background_color)
    # node_border = get_border_style(st.session_state.node_border_color)

    # # Inject CSS for color picker borders
    # css_code = f"""
    # <style>
    #     div[data-baseweb="input"] > div {{
    #         border: {dem_border};
    #     }}
    #     div[data-baseweb="input"]:nth-child(2) > div {{
    #         border: {rep_border};
    #     }}
    #     div[data-baseweb="input"]:nth-child(3) > div {{
    #         border: {background_border};
    #     }}
    #     div[data-baseweb="input"]:nth-child(4) > div {{
    #         border: {node_border};
    #     }}
    # </style>
    # """

    # st.markdown(css_code, unsafe_allow_html=True)

    # Validation inputs
    st.session_state.drop_bills = st.checkbox("Drop bills above 90% yes/no voting?", value=st.session_state.drop_bills)
    st.session_state.cpc_threshold = st.text_input("Enter a value between 0 and 1 for CPC edge plotting threshold", st.session_state.cpc_threshold)
    st.session_state.within_party_threshold = st.text_input("Enter a value between 0 and 1 for within party edge plotting threshold", st.session_state.within_party_threshold)

    # Validation logic
    try:
        numeric_value = float(st.session_state.cpc_threshold)
        if 0 <= numeric_value <= 1:
            st.success(f"Valid CPC threshold input: {st.session_state.cpc_threshold}")
        else:
            st.error("Error: The CPC threshold input must be between 0 and 1.")
    except ValueError:
        st.error("Error: The CPC threshold input must be a numeric value between 0 and 1.")

    try:
        numeric_value = float(st.session_state.within_party_threshold)
        if 0 <= numeric_value <= 1:
            st.success(f"Valid within party threshold input: {st.session_state.within_party_threshold}")
        else:
            st.error("Error: The within party threshold value must be between 0 and 1.")
    except ValueError:
        st.error("Error: The within party threshold must be a numeric value between 0 and 1.")

@st.cache_data
def install_packages():
    # Function to install packages
    def install(package):
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError as e:
            st.error(f"Failed to install {package}. Error: {str(e)}")

    # List of required packages
    required_packages = ['pandas', 'streamlit', 'plotly', 'networkx']
    for package in required_packages:
        install(package)

@st.cache_data
def load_data(file_content):
    data = pd.read_csv(io.BytesIO(file_content))
    return data

@st.cache_data
def process_data(data):
    return data

@st.cache_data
def consolidate_data(members_df, votes_df, filterOverabundance):
    votes_columns = ["congress", "chamber", "rollnumber", "icpsr", "cast_code"]
    members_columns = ["icpsr", "bioname", "party_code", "nominate_dim1", "nominate_dim2"]

    votes_selected = votes_df[votes_columns]
    members_selected = members_df[members_columns]

    # Filter out rows with invalid party_code values
    valid_party_codes = [100, 200]
    members_selected = members_selected[members_selected['party_code'].isin(valid_party_codes)]

    # Merge the DataFrames on the 'icpsr' column
    merged_df = pd.merge(votes_selected, members_selected, on='icpsr')

    # Replace party_code values
    merged_df['party_code'] = merged_df['party_code'].replace({100: 'Democrat', 200: 'Republican'})

    # Replace cast_code values and filter out invalid rows
    cast_code_replacements = {1: 'Yes', 2: 'Yes', 3: 'Yes', 4: 'No', 5: 'No', 6: 'No'}
    merged_df['cast_code'] = merged_df['cast_code'].replace(cast_code_replacements)

    # Remove rows with invalid cast_code values
    valid_cast_codes = ['Yes', 'No']
    filtered_cleaned_df = merged_df[merged_df['cast_code'].isin(valid_cast_codes)]

    if filterOverabundance:
        # Group by rollnumber to check the vote distribution for each bill
        vote_summary = filtered_cleaned_df.groupby('rollnumber')['cast_code'].value_counts(normalize=True).unstack()

        # Identify bills with over 90% Yes or 90% No votes
        bills_to_remove = vote_summary[(vote_summary['Yes'] > 0.9) | (vote_summary['No'] > 0.9)].index

        # Remove rows associated with those bills
        final_df = filtered_cleaned_df[~filtered_cleaned_df['rollnumber'].isin(bills_to_remove)]
        
        return final_df
    else:
        return filtered_cleaned_df

install_packages()

with tab2:
    with st.spinner('Initializing...'):
        votes_columns = ["congress", "chamber", "rollnumber", "icpsr", "cast_code"]
        members_columns = ["icpsr", "bioname", "party_code", "nominate_dim1", "nominate_dim2"]

        members_file = st.file_uploader("Upload your year's members.csv", type='csv')
        votes_file = st.file_uploader("Upload your year's votes.csv", type='csv')

        if members_file is not None:
            st.session_state.members_file_content = members_file.getvalue()
        if votes_file is not None:
            st.session_state.votes_file_content = votes_file.getvalue()

        if st.session_state.members_file_content is not None:
            members_data = load_data(st.session_state.members_file_content)
            for members_item in members_columns:
                if members_item not in members_data.columns:
                    st.error(f'Column "{members_item}" not found in members.csv')
                    members_csv_valid = False
                    break
            else:
                st.write('Members Data Preview:')
                members_editable_data = st.data_editor(members_data)
                members_csv_valid = True
        else:
            members_csv_valid = False
            st.error('Upload members.csv to proceed')

        if st.session_state.votes_file_content is not None:
            votes_data = load_data(st.session_state.votes_file_content)
            for votes_item in votes_columns:
                if votes_item not in votes_data.columns:
                    st.error(f'Column "{votes_item}" not found in votes.csv')
                    votes_csv_valid = False
                    break
            else:
                st.write('Votes Data Preview:')
                votes_editable_data = st.data_editor(votes_data)
                votes_csv_valid = True
        else:
            votes_csv_valid = False
            st.error('Upload votes.csv to proceed')

        if members_csv_valid and votes_csv_valid:
            members_df = process_data(members_editable_data)
            votes_df = process_data(votes_editable_data)
            members_congress_value = members_df['congress'].iloc[0]
            members_chamber_value = members_df['chamber'].iloc[0]
            votes_congress_value = votes_df['congress'].iloc[0]
            votes_chamber_value = votes_df['chamber'].iloc[0]
            if members_congress_value == votes_congress_value and members_chamber_value == votes_chamber_value:
                if st.button('Merge Data'):
                    merged_df = consolidate_data(members_df, votes_df, st.session_state.drop_bills)
                    st.dataframe(merged_df)
                    st.write('Proceed to "Graphing" Tab')
                    st.session_state.merged_df = merged_df
                    st.session_state.graphed = True
            else:
                st.error('Make sure the members.csv and votes.csv have the same congress and chamber values')

with tab3:
    if not st.session_state.graphed:
        st.error('Please merge data before graphing')
    else:
        with st.spinner('Graphing...'):
            def plotCollaborationNetwork(filtered_cleaned_df, cpc_percent, within_party_percent, dem_color, rep_color, bg_color, border_color):
                congress_number = filtered_cleaned_df['congress'].iloc[0]
                congress_name = filtered_cleaned_df['chamber'].iloc[0]
                cross_party_threshold = len(filtered_cleaned_df['rollnumber'].unique()) * cpc_percent
                within_party_threshold = len(filtered_cleaned_df['rollnumber'].unique()) * within_party_percent

                graph_text_color = '#FFFFFF' if is_color_dark(bg_color) else '#000000'
                # Create a graph
                G = nx.Graph()

                # Add nodes with attributes
                for _, row in filtered_cleaned_df.iterrows():
                    G.add_node(row['icpsr'], 
                            party=row['party_code'], 
                            pos=(row['nominate_dim1'], row['nominate_dim2']),
                            bioname=row['bioname'])

                # Group by rollnumber and cast_code
                grouped_df = filtered_cleaned_df.groupby(['rollnumber', 'cast_code'])

                # Add edges with weights
                for (rollnumber, cast_code), group in grouped_df:
                    icpsr_list = group['icpsr'].tolist()
                    for i in range(len(icpsr_list)):
                        for j in range(i + 1, len(icpsr_list)):
                            if G.has_edge(icpsr_list[i], icpsr_list[j]):
                                G[icpsr_list[i]][icpsr_list[j]]['weight'] += 1
                            else:
                                G.add_edge(icpsr_list[i], icpsr_list[j], weight=1)

                # Apply thresholds to edges
                edges_to_remove = []
                for u, v, d in G.edges(data=True):
                    if G.nodes[u]['party'] != G.nodes[v]['party'] and d['weight'] < cross_party_threshold:
                        edges_to_remove.append((u, v))
                    elif G.nodes[u]['party'] == G.nodes[v]['party'] and d['weight'] < within_party_threshold:
                        edges_to_remove.append((u, v))
                G.remove_edges_from(edges_to_remove)

                # Define colors for parties
                party_colors = {'Democrat': dem_color, 'Republican': rep_color}
                node_colors = [party_colors[G.nodes[n]['party']] for n in G.nodes]

                # Get positions for the nodes
                positions = {n: (G.nodes[n]['pos'][0], G.nodes[n]['pos'][1]) for n in G.nodes}
                node_x = [positions[n][0] for n in G.nodes]
                node_y = [positions[n][1] for n in G.nodes]
                node_text = [G.nodes[n]['bioname'] for n in G.nodes]

                cross_party_edges_count = {n: 0 for n in G.nodes}
                for u, v, d in G.edges(data=True):
                    if G.nodes[u]['party'] != G.nodes[v]['party']:
                        cross_party_edges_count[u] += 1
                        cross_party_edges_count[v] += 1

                # Create edges
                edge_x = []
                edge_y = []
                edge_weights = []
                for edge in G.edges(data=True):
                    x0, y0 = positions[edge[0]]
                    x1, y1 = positions[edge[1]]
                    edge_x.append(x0)
                    edge_x.append(x1)
                    edge_x.append(None)
                    edge_y.append(y0)
                    edge_y.append(y1)
                    edge_y.append(None)
                    edge_weights.append(edge[2]['weight'])

                # Create edge traces
                edge_trace = go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=0.05, color='grey'),
                    hoverinfo='none',
                    mode='lines')

                # Create node traces
                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers',
                    hoverinfo='text',
                    text=node_text,
                    marker=dict(
                        showscale=False,
                        color=node_colors,
                        size=10,
                        line = dict(color=border_color, width=2)))

                # Create the figure
                fig = go.Figure(data=[edge_trace, node_trace],
                                layout=go.Layout(
                                    title={
                                        'text': f'{congress_name} Voting Similarity - {congress_number}',
                                        'font': {'color': graph_text_color}
                                    },
                                    titlefont_size=16,
                                    showlegend=False,
                                    hovermode='closest',
                                    margin=dict(b=20,l=5,r=5,t=40),
                                    annotations=[ dict(
                                        text="Hover over nodes to see names",
                                        showarrow=False,
                                        xref="paper", yref="paper",
                                        x=0.005, y=-0.002,
                                        font={'color': graph_text_color})],
                                    xaxis=dict(showgrid=False, zeroline=False, color = graph_text_color, tickfont=dict(color=graph_text_color), linecolor = graph_text_color),
                                    yaxis=dict(showgrid=False, zeroline=False, color = graph_text_color, tickfont=dict(color=graph_text_color), linecolor = graph_text_color),
                                    paper_bgcolor=bg_color,
                                    plot_bgcolor=bg_color))
                                
                return fig
            
            merged_df = st.session_state.merged_df
            fig = plotCollaborationNetwork(merged_df, float(st.session_state.cpc_threshold), float(st.session_state.within_party_threshold), st.session_state.dem_color, st.session_state.rep_color, st.session_state.graph_background_color, st.session_state.node_border_color)
            st.plotly_chart(fig)
