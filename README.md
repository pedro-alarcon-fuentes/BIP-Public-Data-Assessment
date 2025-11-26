# PAI-Assessment

> An advanced project that combines data analysis, network graphs, and multiple layers of AI (BERT, NER, LLMs) to extract, analyze, and summarize company information from public data.

This project is an Open Source Intelligence (OSINT) tool that allows users to explore connections between companies and obtain intelligent summaries about them through a conversational chatbot.

## ✨ Key Features

-   **Network Analysis (Graphs):** Identifies and visualizes hidden connections between companies based on shared addresses, creating clusters of related entities.
-   **Natural Language Chatbot:** Allows users to make complex queries in natural language (e.g., "Find me cybersecurity companies in Dublin").
-   **Multi-Layer AI Engine:**
    -   **Named Entity Recognition (NER):** Automatically extracts locations and topics from the user's question.
    -   **Semantic Search:** Uses vector embeddings (`All-MiniLM-L6-v2`) to find the most relevant companies by understanding meaning, not just keywords.
    -   **Generative Summarization:** Employs a Large Language Model (**Google Gemini**) to read web-scraped information and generate coherent, human-like summaries.
-   **Real-Time Web Scraping:** Performs Google searches and extracts up-to-date information from the websites of found companies to feed the summarization model.

## 🏛️ Project Architecture

The project is modularized for a clear separation of concerns:

-   `main_analysis.py`: The main entry point and control panel. It orchestrates data loading, graph construction, visualization, and chatbot execution.
-   `data_processing.py`: Contains all the logic for loading, cleaning, normalizing, and sampling the company datasets.
-   `ai_engine.py`: The "brain" of the project. It includes functions for generating embeddings, performing semantic search, parsing user intent (NER), and generating summaries with the LLM.
-   `scraper.py`: A standalone module that handles Google searches and scrapes the content from web pages.

## 🚀 Installation and Setup

Follow these steps to get the project up and running on your local machine.

### 1. Prerequisites

-   **Python 3.10 or higher:** The project relies on libraries that require a modern version of Python. You can download it from [python.org](https://www.python.org/downloads/).
-   **Git:** To clone the repository.

### 2. Clone the Repository

```bash
git clone <your_repository_url>
cd <project_folder_name>
```

### 3. Set Up the Environment

Using a virtual environment is essential to isolate the project's dependencies.

**a) Create the Virtual Environment (`.venv`)**

```bash
python -m venv .venv
```

**b) Activate the Virtual Environment**

-   On **Windows** (PowerShell/CMD):
    ```bash
    .\.venv\Scripts\activate
    ```
-   On **macOS / Linux**:
    ```bash
    source .venv/bin/activate
    ```
Your terminal prompt should now start with `(.venv)`.

### 4. Install Dependencies

With the virtual environment activated, install all required libraries with a single command:

```bash
pip install -r requirements.txt
```

### 5. Configure API Keys (`.env`)

This project requires API keys to work with Google services (Gemini), SerpApi (Google Search), and Graphistry.

**a) Create a `.env` file** in the root directory of the project.

**b) Copy and paste the following content** into your `.env` file and replace the placeholder values with your actual keys.

```env
# Google AI Studio API Key for Gemini
# Get it from: https://aistudio.google.com/
GOOGLE_API_KEY="your_google_key_here"

# SerpApi API Key for Google searches
# Sign up at: https://serpapi.com/
SCRAP_KEY="your_serpapi_key_here"
```

## 🏃 How to Run the Project

Once the setup is complete, run the main script from your terminal (with the `.venv` environment activated):

```bash
python main_analysis.py```

You will be presented with a menu in the terminal:

```
====== PROJECT CONTROL PANEL ======
Choose an option:
  1. Perform Connection Graph Analysis
  2. Start Company Search Chatbot
  3. Exit
```

-   **Option 1:** Will load the data, build the graph, analyze it for communities, and generate the visualization.
-   **Option 2:** Will start the interactive chatbot. You can ask it questions in natural language to find and summarize information about companies.
-   **Option 3:** Will exit the program.

