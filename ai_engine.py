# ai_engine.py

import pandas as pd
import os
from sentence_transformers import SentenceTransformer, util
import pickle

MODEL_NAME = 'all-MiniLM-L6-v2'
EMBEDDINGS_FILE = 'company_embeddings.pkl'

def setup_semantic_search(df, id_column='company_num', name_column='company_name'):
    """
    Prepares the search engine by loading the model and embeddings.
    If embeddings don't exist or don't match, it generates and saves them.
    """
    print("\n--- Starting AI Engine ---")
    model = SentenceTransformer(MODEL_NAME)
    if os.path.exists(EMBEDDINGS_FILE):
        print("Loading pre-calculated embeddings...")
        with open(EMBEDDINGS_FILE, 'rb') as f:
            data = pickle.load(f)
            # We verify that the saved IDs match those in the current DataFrame
            if set(data['ids']) == set(df[id_column].tolist()):
                print("Embeddings match. Load successful.")
                # Create an ID to index map for fast lookups
                id_to_idx = {id_val: i for i, id_val in enumerate(data['ids'])}
                return model, data['embeddings'], data['ids'], id_to_idx
            else:
                print("Dataset has changed. Recalculating embeddings...")

    print(f"Generating embeddings for {len(df)} companies...")
    names = df[name_column].astype(str).tolist()
    ids = df[id_column].tolist()
    embeddings = model.encode(names, convert_to_tensor=True, show_progress_bar=True)
    
    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump({'embeddings': embeddings, 'ids': ids}, f)
    
    print("Embeddings generated and saved.")
    id_to_idx = {id_val: i for i, id_val in enumerate(ids)}
    return model, embeddings, ids, id_to_idx


def search_companies(query, location_filter, df, model, embeddings, id_to_idx, top_k=5):
    """
    Main search function.
    1. Filters by location (hard filter).
    2. Performs semantic search by topic ONLY in the filtered results.
    """
    print(f"\n>>> CHATBOT: Searching for '{query}' in '{location_filter}'...")
    
    # 1. LOCATION FILTER
    # We search in all available address columns
    address_cols = ['company_address_1', 'company_address_2', 'company_address_3', 'company_address_4']
    
    # We create a boolean mask: it's True for any row that contains the location in ANY of its addresses
    mask = df[address_cols].apply(
        lambda col: col.str.contains(location_filter, case=False, na=False)
    ).any(axis=1)
    
    location_matches = df[mask]
    
    if location_matches.empty:
        print("No companies found in that location.")
        return pd.DataFrame()  # Return an empty DataFrame
    
    print(f"Found {len(location_matches)} companies in '{location_filter}'. Performing semantic analysis...")
    
    # 2. SEMANTIC SEARCH
    # Get the IDs and embedding indices of the filtered companies
    filtered_ids = location_matches['company_num'].tolist()
    filtered_indices = [id_to_idx[id_val] for id_val in filtered_ids]
    
    # Select only the relevant embeddings
    filtered_embeddings = embeddings[filtered_indices]
    
    # Encode the user's query
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    # Calculate similarity ONLY against the subset of embeddings
    cos_scores = util.cos_sim(query_embedding, filtered_embeddings)[0]
    
    # Add the scores directly to the filtered DataFrame
    location_matches['similarity_score'] = cos_scores.cpu().tolist()
    
    # Sort by score and return the best results
    results = location_matches.sort_values(by='similarity_score', ascending=False)
    
    return results.head(top_k)

def enrich_graph_with_name_similarity(G):
    print("\nEnriching graph with name similarity...")
    model = SentenceTransformer(MODEL_NAME)
    node_names = {node_id: data['company_name'] for node_id, data in G.nodes(data=True) if 'company_name' in data}
    node_ids = list(node_names.keys())
    names_to_embed = list(node_names.values())
    
    if not names_to_embed:
        print("No names to process.")
        return G
        
    embeddings = model.encode(names_to_embed, show_progress_bar=True, convert_to_tensor=True)
    embedding_map = {node_id: embeddings[i] for i, node_id in enumerate(node_ids)}

    for u, v in G.edges():
        try:
            embedding1, embedding2 = embedding_map[u], embedding_map[v]
            similarity_score = util.cos_sim(embedding1, embedding2).item()
            G.edges[u, v]['name_similarity'] = round(similarity_score, 4)
        except KeyError:
            G.edges[u, v]['name_similarity'] = 0
            
    print("Edge enrichment completed.")
    return G