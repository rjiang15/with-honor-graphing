import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objs as go
import sys
import subprocess

# Function to install packages (Run only once)
@st.cache_resource
def install_packages():
    required_packages = ['pandas', 'streamlit', 'plotly', 'networkx', 'kaleido']
    for package in required_packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError as e:
            st.error(f"Failed to install {package}. Error: {str(e)}")

install_packages()

votes_columns = ["congress", "chamber", "rollnumber", "icpsr", "cast_code"]
members_columns = ["icpsr", "bioname", "party_code", "nominate_dim1", "nominate_dim2"]

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

@st.cache_data
def process_data(data):
    return data

@st.cache_data
def consolidate_data(members_df, votes_df, filterOverabundance):
    votes_columns = ["congress", "chamber", "rollnumber", "icpsr", "cast_code"]
    members_columns = ["icpsr", "bioname", "party_code", "nominate_dim1", "nominate_dim2"]

    votes_selected = votes_df[votes_columns]
    members_selected = members_df[members_columns]

    valid_party_codes = [100, 200]
    members_selected = members_selected[members_selected['party_code'].isin(valid_party_codes)]

    merged_df = pd.merge(votes_selected, members_selected, on='icpsr')

    merged_df['party_code'] = merged_df['party_code'].replace({100: 'Democrat', 200: 'Republican'})
    cast_code_replacements = {1: 'Yes', 2: 'Yes', 3: 'Yes', 4: 'No', 5: 'No', 6: 'No'}
    merged_df['cast_code'] = merged_df['cast_code'].replace(cast_code_replacements)

    valid_cast_codes = ['Yes', 'No']
    filtered_cleaned_df = merged_df[merged_df['cast_code'].isin(valid_cast_codes)]

    if filterOverabundance:
        vote_summary = filtered_cleaned_df.groupby('rollnumber')['cast_code'].value_counts(normalize=True).unstack()
        bills_to_remove = vote_summary[(vote_summary['Yes'] > 0.9) | (vote_summary['No'] > 0.9)].index
        final_df = filtered_cleaned_df[~filtered_cleaned_df['rollnumber'].isin(bills_to_remove)]
        return final_df
    else:
        return filtered_cleaned_df

def calculate_party_leaning(df):
    # Determine the leaning for each roll number and party
    leaning = df.groupby(['rollnumber', 'party_code', 'cast_code']).size().unstack(fill_value=0)
    leaning['party_leaning'] = leaning.idxmax(axis=1).apply(lambda x: x)
    
    # Create a mapping of roll number and party to the leaning
    roll_party_leaning = leaning[['party_leaning']].reset_index()
    roll_party_leaning.columns = ['rollnumber', 'party_code', 'party_leaning']
    
    # Merge the leaning information back to the main dataframe
    df = df.merge(roll_party_leaning, on=['rollnumber', 'party_code'], how='left')
    return df

def calculate_collaboration_index_from_df(df):
    # Identify collaborators
    df['collaboration_index'] = (df['cast_code'] != df['party_leaning']).astype(int)
    
    # Aggregate collaboration index by representative
    collaboration_summary = df.groupby(['icpsr', 'bioname', 'party_code'])['collaboration_index'].sum().reset_index()

    # Sort by collaboration index from most to least
    collaboration_summary = collaboration_summary.sort_values(by='collaboration_index', ascending=False)

    # Count the number of unique roll numbers
    unique_rollnumbers = df['rollnumber'].nunique()

    # Add a column for the percentage of collaboration
    collaboration_summary['collaboration_percentage'] = (collaboration_summary['collaboration_index'] / unique_rollnumbers) * 100

    # Round the percentages to the nearest hundredth
    collaboration_summary['collaboration_percentage'] = collaboration_summary['collaboration_percentage'].round(2)

    return collaboration_summary

def calculate_cross_party_roll_numbers(df, percentage):
    cross_party_count = df.groupby('rollnumber').apply(
        lambda x: (x['cast_code'] != x['party_leaning']).sum() / len(x) >= percentage
    ).sum()
    return cross_party_count

def plotCollaborationNetwork(filtered_cleaned_df, cpc_percent, within_party_percent, dem_color, rep_color, bg_color, border_color, edge_width, edge_color, node_border_width, highlight_node=None):
    congress_number = filtered_cleaned_df['congress'].iloc[0]
    congress_name = filtered_cleaned_df['chamber'].iloc[0]
    cross_party_threshold = len(filtered_cleaned_df['rollnumber'].unique()) * cpc_percent
    within_party_threshold = len(filtered_cleaned_df['rollnumber'].unique()) * within_party_percent

    def is_color_dark(color):
        hex_color = color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        brightness = (0.299 * r + 0.587 * g + 0.114 * b) / 255
        return brightness < 0.5

    text_color = '#FFFFFF' if is_color_dark(bg_color) else '#000000'
    G = nx.Graph()

    for _, row in filtered_cleaned_df.iterrows():
        G.add_node(row['icpsr'], 
                   party=row['party_code'], 
                   pos=(row['nominate_dim1'], row['nominate_dim2']),
                   bioname=row['bioname'])

    grouped_df = filtered_cleaned_df.groupby(['rollnumber', 'cast_code'])

    for (rollnumber, cast_code), group in grouped_df:
        icpsr_list = group['icpsr'].tolist()
        for i in range(len(icpsr_list)):
            for j in range(i + 1, len(icpsr_list)):
                if G.has_edge(icpsr_list[i], icpsr_list[j]):
                    G[icpsr_list[i]][icpsr_list[j]]['weight'] += 1
                else:
                    G.add_edge(icpsr_list[i], icpsr_list[j], weight=1)

    edges_to_plot = []
    for u, v, d in G.edges(data=True):
        if (G.nodes[u]['party'] != G.nodes[v]['party'] and d['weight'] >= cross_party_threshold) or \
           (G.nodes[u]['party'] == G.nodes[v]['party'] and d['weight'] >= within_party_threshold):
            edges_to_plot.append((u, v))

    party_colors = {'Democrat': dem_color, 'Republican': rep_color}
    node_colors = [party_colors[G.nodes[n]['party']] for n in G.nodes]

    positions = {n: (G.nodes[n]['pos'][0], G.nodes[n]['pos'][1]) for n in G.nodes}
    node_x = [positions[n][0] for n in G.nodes]
    node_y = [positions[n][1] for n in G.nodes]
    node_text = [G.nodes[n]['bioname'] for n in G.nodes]

    edge_x = []
    edge_y = []
    for u, v in edges_to_plot:
        x0, y0 = positions[u]
        x1, y1 = positions[v]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=edge_width, color=edge_color),
        hoverinfo='none',
        mode='lines')

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=False,
            color=node_colors,
            size=10,
            line=dict(color=border_color, width=node_border_width)))

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title={
                            'text': f'{congress_name} Voting Similarity - {congress_number}',
                            'font': {'color': text_color}
                        },
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            text="Hover over nodes to see names",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002,
                            font={'color': text_color})],
                        xaxis=dict(showgrid=False, zeroline=False, color=text_color, tickfont=dict(color=text_color), linecolor=text_color),
                        yaxis=dict(showgrid=False, zeroline=False, color=text_color, tickfont=dict(color=text_color), linecolor=text_color),
                        paper_bgcolor=bg_color,
                        plot_bgcolor=bg_color))

    if highlight_node is not None:
        highlighted_node_trace = go.Scatter(
            x=[positions[highlight_node][0]],
            y=[positions[highlight_node][1]],
            mode='markers',
            hoverinfo='text',
            text=[G.nodes[highlight_node]['bioname']],
            marker=dict(
                showscale=False,
                color='#FFD700',
                size=12,
                line=dict(color=border_color, width=2)))
        fig.add_trace(highlighted_node_trace)

    return fig, G

def save_fig_as_png(fig, filename):
    fig.write_image(filename)

def display_node_statistics(G, selected_name):
    # Find the node with the selected name
    node = None
    for n, data in G.nodes(data=True):
        if data['bioname'] == selected_name:
            node = n
            break
    
    if node is None:
        st.error(f"Node with name '{selected_name}' not found in the graph.")
        return

    # Find the neighbors with the highest edge weight from the opposite party
    neighbors = [
        (nbr, G[node][nbr]['weight']) 
        for nbr in G.neighbors(node)
        if G.nodes[nbr]['party'] != G.nodes[node]['party']
    ]
    neighbors.sort(key=lambda x: x[1], reverse=True)

    if not neighbors:
        st.info(f"No cross-party neighbors found for '{selected_name}'.")
        return

    st.write(f"Top cross-party neighbors for '{selected_name}':")
    for nbr, weight in neighbors:
        st.write(f"- {G.nodes[nbr]['bioname']} (Number of times voted with selected Congressperson: {weight})")

def clear_members_data():
    if 'members_data' in st.session_state:
        del st.session_state['members_data']
        del st.session_state['members_editable_data']
        del st.session_state['members_csv_valid']

def clear_votes_data():
    if 'votes_data' in st.session_state:
        del st.session_state['votes_data']
        del st.session_state['votes_editable_data']
        del st.session_state['votes_csv_valid']

def clear_merged_data():
    if 'merged_df' in st.session_state:
        del st.session_state['merged_df']
        del st.session_state['graphed']

def clear_graph_data():
    if 'graph' in st.session_state:
        del st.session_state['graph']
    if 'graph_highlighted' in st.session_state:
        del st.session_state['graph_highlighted']
    if 'G' in st.session_state:
        del st.session_state['G']
    if 'selected_name' in st.session_state:
        del st.session_state['selected_name']

st.title('With Honor - Graphical Analysis')

tab1, tab2, tab3, tab4 = st.tabs(["Settings", "Data Consolidation", "Graphing", "Documentation"])

with tab1:
    if 'settings' not in st.session_state:
        st.session_state.settings = {
            'dem_color': "#003862",
            'rep_color': "#C00000",
            'background_color': "#ffffff",
            'node_border_color': "#504D4D",
            'drop_bills': True,
            'cpc_threshold': "0.4",
            'within_party_threshold': "0.75",
            'edge_width': "0.025",
            'edge_color': '#D28DD2',
            'node_border_width': "0.5"
        }

    st.session_state.settings['dem_color'] = st.color_picker("Pick a color to represent Democrats on the Graph (Default = #003862)", st.session_state.settings['dem_color'])
    st.session_state.settings['rep_color'] = st.color_picker("Pick a color to represent Republicans on the Graph (Default = #C00000)", st.session_state.settings['rep_color'])
    st.session_state.settings['background_color'] = st.color_picker("Pick a color for the background of the graph (Default = #ffffff)", st.session_state.settings['background_color'])
    st.session_state.settings['node_border_color'] = st.color_picker("Pick a color for the node borders on the graphs (Default = #504D4D)", st.session_state.settings['node_border_color'])
    st.session_state.settings['edge_width'] = st.text_input("Set edge width (Default = 0.025)", st.session_state.settings['edge_width'])
    st.session_state.settings['edge_color'] = st.color_picker("Pick a color for the edges of the graph (Defualt = #414141)", st.session_state.settings['edge_color'])
    st.session_state.settings['node_border_width'] = st.text_input("Set node border width (Default = 0.5)", st.session_state.settings['node_border_width'])
    st.session_state.settings['drop_bills'] = st.checkbox("Drop bills above 90% yes/no voting? (Default = True)", st.session_state.settings['drop_bills'])
    st.session_state.settings['cpc_threshold'] = st.text_input("Enter a value between 0 and 1 for CPC edge plotting threshold (Default = 0.4)", st.session_state.settings['cpc_threshold'])
    st.session_state.settings['within_party_threshold'] = st.text_input("Enter a value between 0 and 1 for within party edge plotting threshold (Default = 0.75)", st.session_state.settings['within_party_threshold'])

    try:
        numeric_value = float(st.session_state.settings['cpc_threshold'])
        if 0 <= numeric_value <= 1:
            st.success(f"Valid CPC threshold input: {st.session_state.settings['cpc_threshold']}")
        else:
            st.error("Error: The CPC threshold input must be between 0 and 1.")
    except ValueError:
        st.error("Error: The CPC threshold input must be a numeric value between 0 and 1.")

    try:
        numeric_value = float(st.session_state.settings['within_party_threshold'])
        if 0 <= numeric_value <= 1:
            st.success(f"Valid within party threshold input: {st.session_state.settings['within_party_threshold']}")
        else:
            st.error("Error: The within party threshold value must be between 0 and 1.")
    except ValueError:
        st.error("Error: The within party threshold must be a numeric value between 0 and 1.")
    
    try:
        numeric_value = float(st.session_state.settings['edge_width'])
        if 0.01 <= numeric_value <= 2.0:
            st.success(f"Valid edge width input: {st.session_state.settings['edge_width']}")
        else:
            st.error("Error: The edge width input must be between 0.01 and 2.0.")
    except ValueError:
        st.error("Error: The edge width input must be a numeric value between 0.01 and 2.0.")
    
    try:
        numeric_value = float(st.session_state.settings['node_border_width'])
        if 0.1 <= numeric_value <= 5.0:
            st.success(f"Valid node border width input: {st.session_state.settings['node_border_width']}")
        else:
            st.error("Error: The node border width input must be between 0.1 and 5.0.")
    except ValueError:
        st.error("Error: The node border width input must be a numeric value between 0.1 and 5.0.")

with tab2:
    with st.spinner('Initializing...'):
        if 'members_data' not in st.session_state:
            members_file = st.file_uploader("Upload your year's members.csv", type='csv')
            if members_file is not None:
                members_data = load_data(members_file)
                if not members_data.empty:
                    members_data = members_data[members_data['chamber'] != 'President']
                    st.info('Filtered out all president rows from the data.')

                for members_item in members_columns:
                    if members_item not in members_data.columns:
                        st.error(f'Column "{members_item}" not found in members.csv')
                        break
                else:
                    st.session_state.members_data = members_data
                    st.session_state.members_csv_valid = True
                    st.write('Members Data Preview:')
                    st.session_state.members_editable_data = st.data_editor(members_data)
            else:
                st.error('Uploaded members.csv is empty')
                st.session_state.members_csv_valid = False
        else:
            st.write('Members Data Preview:')
            st.data_editor(st.session_state.members_editable_data)
            if st.button('Clear Members Data'):
                clear_members_data()
                st.query_params()  # Updated line here
        
        if 'votes_data' not in st.session_state:
            votes_file = st.file_uploader("Upload your year's votes.csv", type='csv')
            if votes_file is not None:
                votes_data = load_data(votes_file)
                for votes_item in votes_columns:
                    if votes_item not in votes_data.columns:
                        st.error(f'Column "{votes_item}" not found in votes.csv')
                        break
                else:
                    st.session_state.votes_data = votes_data
                    st.session_state.votes_csv_valid = True
                    st.write('Votes Data Preview:')
                    st.session_state.votes_editable_data = st.data_editor(votes_data)
            else:
                st.error('Uploaded votes.csv is empty')
                st.session_state.votes_csv_valid = False
        else:
            st.write('Votes Data Preview:')
            st.data_editor(st.session_state.votes_editable_data)
            if st.button('Clear Votes Data'):
                clear_votes_data()
                st.query_params()  # Updated line here

        if st.button('Clear Merged Data'):
            clear_merged_data()
            st.query_params()  # Updated line here

        if 'graphed' not in st.session_state:
            st.session_state.graphed = False

        if st.session_state.members_csv_valid and st.session_state.votes_csv_valid:
            members_df = process_data(st.session_state.members_editable_data)
            votes_df = process_data(st.session_state.votes_editable_data)
            members_congress_value = members_df['congress'].iloc[0]
            members_chamber_value = members_df['chamber'].iloc[0]
            votes_congress_value = votes_df['congress'].iloc[0]
            votes_chamber_value = votes_df['chamber'].iloc[0]
            if members_congress_value == votes_congress_value and members_chamber_value == votes_chamber_value:
                if 'merged_df' not in st.session_state:
                    merged_df = consolidate_data(members_df, votes_df, st.session_state.settings['drop_bills'])
                    merged_df = calculate_party_leaning(merged_df)
                    st.session_state.merged_df = merged_df
                    st.session_state.graphed = True
                    st.write('Merged Data Preview:')
                    st.dataframe(merged_df)
                else:
                    st.write('Merged Data Preview:')
                    st.dataframe(st.session_state.merged_df)
            else:
                st.error('Make sure the members.csv and votes.csv have the same congress and chamber values')

with tab3:
    if 'stop_graph_generation' not in st.session_state:
        st.session_state.stop_graph_generation = False
    
    if not st.session_state.graphed:
        st.error('Please merge data before graphing')
    else:
        col1, col2 = st.columns([3, 1])  # Adjust column widths
        with col1:
            with st.spinner('Graphing...'):
                if st.button('Generate Graph'):
                    clear_graph_data()  # Clear previous graph data
                    st.session_state.stop_graph_generation = False
                    try:
                        fig, G = plotCollaborationNetwork(
                            st.session_state.merged_df, 
                            float(st.session_state.settings['cpc_threshold']), 
                            float(st.session_state.settings['within_party_threshold']), 
                            st.session_state.settings['dem_color'], 
                            st.session_state.settings['rep_color'], 
                            st.session_state.settings['background_color'], 
                            st.session_state.settings['node_border_color'],
                            float(st.session_state.settings['edge_width']),
                            st.session_state.settings['edge_color'],
                            float(st.session_state.settings['node_border_width'])
                        )
                        st.session_state.graph = fig
                        st.session_state.G = G
                    except Exception as e:
                        st.error(f"Error generating graph: {str(e)}")
                        st.session_state.stop_graph_generation = True

                    if 'selected_name' in st.session_state and st.session_state.selected_name != "No Congressperson Selected":
                        selected_node = None
                        for n, data in st.session_state.G.nodes(data=True):
                            if data['bioname'] == st.session_state.selected_name:
                                selected_node = n
                                break

                        fig_highlighted, _ = plotCollaborationNetwork(
                            st.session_state.merged_df, 
                            float(st.session_state.settings['cpc_threshold']), 
                            float(st.session_state.settings['within_party_threshold']), 
                            st.session_state.settings['dem_color'], 
                            st.session_state.settings['rep_color'], 
                            st.session_state.settings['background_color'], 
                            st.session_state.settings['node_border_color'],
                            float(st.session_state.settings['edge_width']),
                            st.session_state.settings['edge_color'],
                            float(st.session_state.settings['node_border_width']),
                            highlight_node=selected_node
                        )
                        st.session_state.graph_highlighted = fig_highlighted

                # Add stop button
                if st.button('Stop Graph Generation'):
                    st.session_state.stop_graph_generation = True

                if st.session_state.stop_graph_generation:
                    st.write("Graph generation has been stopped.")

                # Add search functionality
                if 'G' in st.session_state:
                    node_names = [data['bioname'] for n, data in st.session_state.G.nodes(data=True)]
                    node_names.sort()
                    node_names.insert(0, "No Congressperson Selected")

                    # Initialize selected_name in session state if not already set
                    if 'selected_name' not in st.session_state:
                        st.session_state.selected_name = "No Congressperson Selected"

                    # Display the graphs side by side
                    with st.container():
                        st.plotly_chart(st.session_state.graph)
                        st.caption("No Congressperson Selected")

                        if "graph_highlighted" in st.session_state:
                            st.plotly_chart(st.session_state.graph_highlighted)
                            st.caption(f"Highlighted Congressperson: {st.session_state.selected_name}")
                        else:
                            st.plotly_chart(st.session_state.graph)
                            st.caption("Highlighted Congressperson: None")

                        # Search box below the graphs
                        selected_name = st.selectbox("Search for a person in the graph:", node_names, key='highlight_selectbox')
                        if selected_name != st.session_state.selected_name:
                            st.session_state.selected_name = selected_name
                            # Highlight the selected node
                            selected_node = None
                            for n, data in st.session_state.G.nodes(data=True):
                                if data['bioname'] == selected_name:
                                    selected_node = n
                                    break

                            # Update the graph with the highlighted node
                            fig, _ = plotCollaborationNetwork(
                                st.session_state.merged_df, 
                                float(st.session_state.settings['cpc_threshold']), 
                                float(st.session_state.settings['within_party_threshold']), 
                                st.session_state.settings['dem_color'], 
                                st.session_state.settings['rep_color'], 
                                st.session_state.settings['background_color'], 
                                st.session_state.settings['node_border_color'],
                                float(st.session_state.settings['edge_width']),
                                st.session_state.settings['edge_color'],
                                float(st.session_state.settings['node_border_width']),
                                highlight_node=selected_node
                            )
                            st.session_state.graph_highlighted = fig
                            st.query_params()  # Updated line here

                        # Display the neighbors list
                        if selected_name and selected_name != "No Congressperson Selected":
                            st.write(f"Neighbors of {selected_name}:")
                            display_node_statistics(st.session_state.G, selected_name)
                        else:
                            st.write("No Congressperson Selected")

                        # Download Graph as PNG
                        if st.session_state.graph:
                            png_image = st.session_state.graph.to_image(format="png")
                            st.download_button(
                                label="Download Graph as PNG",
                                data=png_image,
                                file_name="graph.png",
                                mime="image/png"
                            )

                        if "graph_highlighted" in st.session_state:
                            png_image_highlighted = st.session_state.graph_highlighted.to_image(format="png")
                            st.download_button(
                                label="Download Highlighted Graph as PNG",
                                data=png_image_highlighted,
                                file_name="highlighted_graph.png",
                                mime="image/png"
                            )

        with col2:
            if 'G' in st.session_state:
                merged_df = st.session_state.merged_df

                # Ensure party leanings are calculated
                collaboration_summary_df = calculate_collaboration_index_from_df(merged_df)

                # Calculate the unique roll numbers from Tab 2's merged data preview
                unique_roll_numbers = merged_df['rollnumber'].nunique()

                # Calculate the number of roll numbers with at least X% of representatives voting across the aisle
                cross_party_5 = calculate_cross_party_roll_numbers(merged_df, 0.05)
                cross_party_10 = calculate_cross_party_roll_numbers(merged_df, 0.10)
                cross_party_25 = calculate_cross_party_roll_numbers(merged_df, 0.25)

                st.header("Graph Statistics")
                st.write(f"Unique Roll Numbers: {unique_roll_numbers}")
                st.write(f"Number of Roll Numbers with at least 5% of representatives voting across the aisle: {cross_party_5}")
                st.write(f"Number of Roll Numbers with at least 10% of representatives voting across the aisle: {cross_party_10}")
                st.write(f"Number of Roll Numbers with at least 25% of representatives voting across the aisle: {cross_party_25}")

                # Generate collaboration summary
                st.download_button(
                    label="Download Representatives CSV",
                    data=collaboration_summary_df.to_csv(index=False),
                    file_name='representatives_collaboration_summary.csv',
                    mime='text/csv'
                )

with tab4:
    st.markdown("""
        ### App Documentation
        For detailed documentation on the app, please refer to the following document:
        [View App Documentation](https://docs.google.com/document/d/1wFJXC0VKODcpYaiuMd54VpnZGmC164FQqQeaIEULj_s/edit#heading=h.qsujw3fkoz4m)
                
        —Happy Analysis, Ryan Jiang
    """)
