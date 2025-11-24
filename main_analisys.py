# main_analysis.py 

# --- 1. IMPORTS ---
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from community import community_louvain
from transformers import pipeline
import google.generativeai as genai     
import os
from dotenv import load_dotenv
import graphistry
# Imports from other files
from data_processing import load_and_prepare_companies
from ai_engine import setup_semantic_search, search_companies, enrich_graph_with_name_similarity, summarize_with_llm 
from ai_engine import parse_user_intent
from scraper import scrape_and_save


# --- 2. CONFIGURATION ---
DATA_PATH1 = "BIP-Public-Data-Assessment/datasets/companies.csv"
DATA_PATH2 = "BIP-Public-Data-Assessment/datasets/romania_companies.csv"
ID_COLUMN = 'company_num'
GRAPH_SAMPLE_SIZE = 100
CHATBOT_SAMPLE_SIZE = 80000

env_path = "BIP-Public-Data-Assessment/.env"
load_dotenv(env_path) 
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
    Detects communities in the graph and creates a static visualization with Matplotlib.

    Uses the Louvain algorithm to find clusters, then draws the graph coloring each
    cluster, adjusting node size according to their importance (degree) and edge width
    according to semantic similarity.
    """
    if G.number_of_nodes() == 0:
        print("The graph is empty, cannot visualize.")
        return

    print("\nStarting static visualization with Matplotlib...")

    # --- 1. GRAPH ANALYSIS (Calculations) ---
    print("Detecting communities with the Louvain algorithm...")
    # 'best_partition' returns a dictionary: {node_id: community_id}
    partition = community_louvain.best_partition(G)
    num_communities = len(set(partition.values()))
    print(f"Found {num_communities} communities/clusters.")

    # --- 2. PREPARATION OF VISUAL ATTRIBUTES FOR MATPLOTLIB ---
    # We create attribute lists in the same order as G.nodes() and G.edges()
    
    # a) Node colors, based on their community ID
    node_colors = [partition.get(node) for node in G.nodes()]
    
    # b) Node sizes, based on their degree (number of connections)
    node_sizes = [G.degree(node) * 20 + 50 for node in G.nodes()]  # Multiplier to make it visible
    
    # c) Edge widths, based on semantic similarity
    # Edges with high similarity will be thicker.
    edge_widths = [1 + G.edges[u, v].get('name_similarity', 0) * 3 for u, v in G.edges()]

    # --- 3. LAYOUT AND DRAWING ---
    print("Computing spring layout (may take a moment)...")
    # The spring layout is key to the "clusters" appearance.
    # A 'seed' makes the layout reproducible each time you run it.
    pos = nx.spring_layout(G, k=0.6, iterations=50, seed=42)

    # Create the figure where the graph will be drawn
    plt.figure(figsize=(25, 25))
    
    # a) Draw the edges
    nx.draw_networkx_edges(
        G,
        pos,
        width=edge_widths,
        alpha=0.3,          
        edge_color='gray'
    )
    
    # b) Draw the nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        cmap=plt.cm.viridis  # The color map used for the clusters
    )
    
    # c) Draw the labels
    labels_to_draw = {}
    # Threshold: only show names of nodes with more than 5 connections
    degree_threshold = 5 
    for node, data in G.nodes(data=True):
        if G.degree(node) > degree_threshold:
            labels_to_draw[node] = data.get('company_name', '')
            
    nx.draw_networkx_labels(
        G,
        pos,
        labels=labels_to_draw,
        font_size=10,
        font_color='black'
    )
    
    plt.box(False)  # Hide the graph frame
    print("Displaying the graph...")
    plt.show()

def run_chatbot():
    """
    Main function that executes the chatbot lifecycle.
    Now accepts input in natural language.
    """
    print("\n--- STARTING CHATBOT MODULE ---")

    api_key = os.getenv("GOOGLE_API_KEY")

    
    if not api_key:
        print("Critical Error: GOOGLE_API_KEY environment variable not found.")
        print("Make sure you have a .env file with the format: GOOGLE_API_KEY='your_key'")
        return
    
    try:
        genai.configure(api_key=api_key)
        llm_model = genai.GenerativeModel('gemini-2.5-flash')
        print("Generative AI Model (Gemini) loaded successfully.")
    except Exception as e:
        print(f"Error configuring Gemini model. Verify your API key. Error: {e}")
        return

    
    # Step 1: Load data
    companies_df1 = load_and_prepare_companies(DATA_PATH1, sample_size=CHATBOT_SAMPLE_SIZE, clean_csv_name="companies_clean_sampled.csv")
    companies_df2 = load_and_prepare_companies(DATA_PATH2, sample_size=CHATBOT_SAMPLE_SIZE, clean_csv_name="companies_clean_sampled2.csv")
    companies_df = pd.concat([companies_df1, companies_df2], ignore_index=True)
    
    # Step 2: Prepare the SEMANTIC SEARCH engine (embeddings)
    model, embeddings, company_ids, id_to_idx = setup_semantic_search(companies_df)
    
    # Step 3: Prepare the INTENT ANALYSIS engine (NER)
    print("Loading Named Entity Recognition (NER) model...")
    # 'dslim/bert-base-NER' is a very popular and robust model.
    ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)
    print("Cargando modelo de Resumen (Summarization)...")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    print("Todos los modelos de IA están listos.")

    # Step 4: Interactive loop
    print("\n--- Company Search Chatbot Activated ---")
    print("Hello! Ask me a question like: 'Find technology companies near Dublin'")
    print("Type 'exit' to return to the main menu.")
    
    while True:
        user_input = input("\n> ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        # Analyze the user's input to extract location and topic
        location, query = parse_user_intent(user_input, ner_pipeline)
        
        # Check if we understood the question
        if not location or not query:
            print("Sorry, I didn't quite understand your question. Please make sure to include a location and a company type.")
            continue
            
        # Perform the search with the extracted information
        results = search_companies(query, location, companies_df, model, embeddings, id_to_idx, top_k=5)
        
        # Display the results
        if not results.empty:
            print(f"\n¡Claro! He encontrado la siguiente información sobre '{query}' en '{location}':")

            company_names = []
            for _, row in results.iterrows():
                name = row['company_name']
                company_names.append(name)
            print("\nStarting web scraping for the found companies...\n")
            for name in company_names:
                filename = scrape_and_save(name, location)
                print(f" → Scraped data saved to: {filename}")

            for index, row in results.iterrows():
                company_name = row['company_name']
                print(f"\n--- Summarizing information for: {company_name} ---")
                
                summary = summarize_with_llm(company_name, llm_model)
                
                print(f"🔹 **{company_name}**:")
                print(f"   {summary}")
        else:
            print("I couldn't find results for your search. Try with other terms.")

    print("\n--- Chatbot deactivated. ---")



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
            companies_df1 = load_and_prepare_companies(DATA_PATH1, sample_size=CHATBOT_SAMPLE_SIZE)
            companies_df2 = load_and_prepare_companies(DATA_PATH2, sample_size=CHATBOT_SAMPLE_SIZE)
            companies_df = pd.concat([companies_df1, companies_df2], ignore_index=True)
            
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