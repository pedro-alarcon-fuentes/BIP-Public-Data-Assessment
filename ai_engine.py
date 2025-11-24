# ai_engine.py

import pandas as pd
import os
from sentence_transformers import SentenceTransformer, util
import pickle
import re
import json 

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

def parse_user_intent(sentence, ner_pipeline):
    """
    Analyzes a user sentence to extract the location and search topic.
    Implements a precise strategy to detect the topic based on keywords.
    """
    print(f"Analyzing user intent: '{sentence}'")
    
    # --- LOCATION EXTRACTION ---
    location = None
    processed_sentence = sentence.title()
    ner_results = ner_pipeline(processed_sentence)
    location_parts = [entity['word'] for entity in ner_results if entity['entity_group'] in ['LOC', 'GPE']]
    if location_parts:
        location = " ".join(location_parts)
    else:
        # Plan B (Gazetteer)
        GAZETTEER = ["Dublin 2", "Dublin", "Cork", "Galway", "Limerick", "Waterford", "Bulgaria", "Spain"]
        sentence_lower = sentence.lower()
        for place in GAZETTEER:
            if place.lower() in sentence_lower:
                location = place
                break
    
    if not location:
        print("--> Intent detected: Could not find a location!")
        return None, None

    # --- TOPIC EXTRACTION ---
    query = None
    
    # 1. Define anchor words
    anchor_words = ['companies', 'company', 'businesses', 'business', 'firms', 'firm', 'startups', 'startup', 'agencies', 'agency', 'sector']
    
    # Prepare the sentence for topic analysis (lowercase and without location)
    sentence_for_query = sentence.lower().replace(location.lower(), "")
    tokens = sentence_for_query.split()
    
    # 2. Search for the first anchor word
    anchor_index = -1
    for i, token in enumerate(tokens):
        # Remove punctuation for robust matching (e.g. "companies.")
        cleaned_token = token.strip('.,?!')
        if cleaned_token in anchor_words:
            anchor_index = i
            break
            
    # 3. If anchor found, extract the topic
    if anchor_index != -1:
        print("Anchor word found. Extracting precise topic...")
        # Define window size (1 to 3 words before the anchor)
        start_index = max(0, anchor_index - 3)
        topic_words = tokens[start_index:anchor_index]
        query = " ".join(topic_words)
    
    # 4. If no anchor, clean the sentence as before
    else:
        print("Anchor word not found. Using general cleanup as fallback.")
        query = sentence_for_query
        stop_words = ['in ', 'at ', 'near ', 'around ', 'find ', 'look for ', 'looking for ', 'i want to know about ', 'there', 'are there any']
        for word in stop_words:
            query = query.replace(word, '')
    
    # Final cleanup
    query = query.strip('.,?! ')
    
    # If query is empty after all processing, it's an error. Use a generic one.
    if not query:
        query = "business"

    print(f"--> Intent detected: Location='{location}', Topic='{query}'")
    return location, query

    
def summarize_with_llm(company_name, generative_model, scrape_data_path="scraped_data"):
    """
    Uses an LLM (Gemini) to read scraping data and generate a natural summary.

    Args:
        company_name (str): The company name to search for its JSON file.
        generative_model (GenerativeModel): The initialized Gemini model.
        scrape_data_path (str): The path to the folder containing JSON files.

    Returns:
        str: A conversational summary about the company.
    """
    def sanitize_name(name):
        name = name.lower()
        name = re.sub(r'[^a-z0-9]+', '', name)
        return name

    sanitized_target_name = sanitize_name(company_name)
    found_path = None
    for filename in os.listdir(scrape_data_path):
        if filename.endswith(".json"):
            sanitized_filename = sanitize_name(os.path.splitext(filename)[0])
            if sanitized_filename == sanitized_target_name:
                found_path = os.path.join(scrape_data_path, filename)
                break
    
    if not found_path:
        return "No detailed scraping data found for this company."

    # Read the JSON content
    try:
        with open(found_path, 'r', encoding='utf-8') as f:
            scraped_data_list = json.load(f)
            scraped_data_str = json.dumps(scraped_data_list, indent=2)
    except Exception as e:
        return f"Error reading or processing the JSON file: {e}"

    # --- PROMPT DESIGN FOR GEMINI ---
    prompt = f"""
    You are an expert and concise business analyst, that utilises OSINT. Your task is to summarize company information based on web scraping data provided in JSON format.

    Here is the data for "{company_name}":
    ```json
    {scraped_data_str}
    ```

    Please analyze the JSON and write a summary of 2-3 sentences in a natural and professional tone. Ignore irrelevant text such as menus, privacy policies, or error messages. Focus on:
    1. What is the company and what is its main activity?
    2. What key services or products does it offer?
    3. Where is it located or what markets does it serve?

    If the data is insufficient to form a good summary, simply indicate that the information found is limited. If this happens, fill in some information from your knowledge base that could be useful to the user.


    Summary:
    """

    # --- CALL TO GEMINI API ---
    try:
        response = generative_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error contacting the generative AI API: {e}"
