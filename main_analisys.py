# main_analysis.py 

# --- 1. IMPORTS ---
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
from community import community_louvain
# Imports from other files
from data_processing import load_and_prepare_companies
from ai_engine import setup_semantic_search, search_companies
from ai_engine import enrich_graph_with_name_similarity 


# --- 2. CONFIGURATION ---
DATA_PATH = "BIP-Public-Data-Assessment/companies.csv"
ID_COLUMN = 'company_num'
GRAPH_SAMPLE_SIZE = 100
CHATBOT_SAMPLE_SIZE = 5000

# --- 3. MAIN LOGIC ---

def create_graph_from_data(df):
    """
    Builds a NetworkX graph from a DataFrame of companies.
    
    This version handles multiple address columns, creating edges if
    ANY normalized address is shared between two companies.
    """
    if ID_COLUMN not in df.columns:
        raise ValueError(f"The ID column '{ID_COLUMN}' was not found in the DataFrame.")

    print("\nStarting graph construction...")
    G = nx.Graph()

    # We use the original DataFrame (one row per company) to create the nodes
    # and assign their main attributes.
    print(f"Adding {len(df)} nodes to the graph...")
    for _, row in df.iterrows():
        node_id = row[ID_COLUMN]
        attributes = row.to_dict()
        G.add_node(node_id, **attributes)
    
    # --- CREATE EDGES ---
    print("Creating edges based on ANY shared address...")
    
    # 1. IDENTIFY ADDRESS COLUMNS
    address_cols = ['company_address_1', 'company_address_2', 'company_address_3', 'company_address_4']
    
    # 2. "MELT" THE DATAFRAME
    # Transform the df to long format.
    # id_vars are the columns we want to keep fixed.
    # value_vars are the columns we want to "melt" into one.
    df_long = df.melt(
        id_vars=[ID_COLUMN],
        value_vars=address_cols,
        var_name='address_type',  # New column with the original column name
        value_name='address'      # New column with the address value
    )
    
    # 3. CLEAN AND NORMALIZE THE ADDRESS COLUMN
    # Remove rows where the address is empty or null
    df_long.dropna(subset=['address'], inplace=True)
    df_long = df_long[df_long['address'].str.strip() != '']
    
    # Apply the normalization function (imported from data_processing)
    from data_processing import normalize_address
    df_long['normalized_address'] = df_long['address'].apply(normalize_address)
    
    # Remove rows that became empty after normalization
    df_long = df_long[df_long['normalized_address'] != '']
    
    # 4. GROUP AND CREATE EDGES
    address_groups = df_long.groupby('normalized_address')[ID_COLUMN].apply(list)
    
    edge_count = 0
    for company_list in address_groups:
        # Remove duplicates in case a company has the same address in two columns
        unique_company_list = list(set(company_list))
        if len(unique_company_list) > 1:
            for node1, node2 in combinations(unique_company_list, 2):
                # Add the edge if it doesn't already exist
                if not G.has_edge(node1, node2):
                    G.add_edge(node1, node2, reason='Shared Address')
                    edge_count += 1
    
    print(f"{edge_count} new edges have been added.")
    print("\n--- Graph Successfully Created! ---")
    print(f"Total number of nodes: {G.number_of_nodes()}")
    print(f"Total number of edges: {G.number_of_edges()}")
    return G


def visualize_community_graph(G):
    """
    Detects communities in the graph and creates a visualization.

    Uses the Louvain algorithm to find clusters, then draws
    the graph coloring each cluster and adjusting node sizes
    and edge thickness.
    """
    if G.number_of_nodes() == 0:
        print("The graph is empty, cannot visualize.")
        return

    print("\nStarting advanced visualization by clusters...")

    # 1. COMMUNITY DETECTION (CLUSTERS)
    print("Detecting communities with the Louvain algorithm...")
    # 'best_partition' returns a dictionary: {node_id: community_id}
    partition = community_louvain.best_partition(G)
    
    # Get the number of communities found
    num_communities = len(set(partition.values()))
    print(f"Found {num_communities} communities/clusters.")

    # 2. PREPARATION OF VISUAL ATTRIBUTES
    
    # Assign a color to each community for the nodes
    node_colors = [partition[node] for node in G.nodes()]
    
    # Assign node size based on degree (number of connections)
    node_sizes = [G.degree(node) * 20 + 50 for node in G.nodes()]  # Multiplier for visibility
    
    # Assign edge transparency based on semantic similarity
    # Edges with high similarity will be more opaque.
    edge_alphas = [G.edges[u, v].get('name_similarity', 0.1) for u, v in G.edges()]

    # 3. LAYOUT AND DRAWING
    print("Computing force-directed layout (may take a moment)...")
    # Spring layout is key for the "clusters" appearance
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=2409)

    plt.figure(figsize=(20, 20))
    
    # Draw the graph with all visual attributes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, cmap=plt.cm.jet)
    nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color='gray', width=[alpha * 2 for alpha in edge_alphas])
    
    # Add labels to the most important nodes to avoid clutter
    central_nodes = {node for node, degree in G.degree() if degree > 5}  # Degree threshold
    labels = {node: G.nodes[node]['company_name'] for node in central_nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_color='black')
    
    plt.title(f"Company Graph by Communities (Clusters) - {num_communities} clusters found")
    plt.box(False)
    plt.show()

def run_chatbot():
    """
    Main function that executes the chatbot lifecycle.
    """
    print("\n--- STARTING CHATBOT MODULE ---")
    
    # Step 1: Load optimized data for the chatbot
    print(f"Loading data for chatbot (sample of {CHATBOT_SAMPLE_SIZE})...")
    companies_df = load_and_prepare_companies(DATA_PATH, sample_size=CHATBOT_SAMPLE_SIZE)
    
    # Step 2: Set up the AI engine
    model, embeddings, company_ids, id_to_idx = setup_semantic_search(companies_df)
    
    # Step 3: Interactive loop
    print("\n--- Company Search Chatbot Activated ---")
    print("Hello! Tell me where you want to search for companies and what type.")
    print("Type 'exit' to return to the main menu.")
    
    while True:
        location = input("\nEnter the location (e.g., Dublin): ")
        if location.lower() in ['salir', 'exit', 'quit']:
            break
            
        query = input(f"What type of companies are you looking for in '{location}'? (e.g., Cybersecurity): ")
        if query.lower() in ['salir', 'exit', 'quit']:
            break
            
        results = search_companies(query, location, companies_df, model, embeddings, id_to_idx, top_k=5)
        
        if not results.empty:
            print("\nHere are the 5 most relevant results:")
            for _, row in results.iterrows():
                print(f"  - Name: {row['company_name']} (Similarity: {row['similarity_score']:.2f})")
                print(f"    Address 1: {row.get('company_address_1', 'N/A')}")
                print(f"    Address 2: {row.get('company_address_2', 'N/A')}")
                print(f"    Address 3: {row.get('company_address_3', 'N/A')}")
                print(f"    Address 4: {row.get('company_address_4', 'N/A')}")
        else:
            print("I could not find results for your search.")

    print("\n--- Chatbot deactivated. Returning to main menu. ---")

# --- 4. EXECUTION ---
if __name__ == "__main__":
    
    while True:
        print("\n====== PROJECT CONTROL PANEL ======")
        print("Choose an option:")
        print("  1. Perform Connection Graph Analysis")
        print("  2. Start Company Search Chatbot")
        print("  3. Exit")
        
        choice = input("Enter your choice (1, 2, or 3): ")
        
        if choice == '1':
            print("\n--- STARTING GRAPH ANALYSIS MODULE ---")
            
            # Step 1: Load data
            companies_df = load_and_prepare_companies(DATA_PATH, sample_size=GRAPH_SAMPLE_SIZE)
            
            # Step 2: Create the structural graph
            company_graph = create_graph_from_data(companies_df)
            
            # Step 3: Enrich the graph with AI
            company_graph = enrich_graph_with_name_similarity(company_graph)

            # Step 4: Visualize the graph by communities
            visualize_community_graph(company_graph)
            
        elif choice == '2':
            run_chatbot()
            
        elif choice == '3':
            print("Exiting the program. Goodbye!")
            break
            
        else:
            print("Invalid option. Please choose 1, 2, or 3.")