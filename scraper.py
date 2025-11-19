import json
import requests
from bs4 import BeautifulSoup
import re
import os

def google_search(query, num_results=10):
    url = "https://serpapi.com/search"

    params = {
        "q": query,
        "api_key": "28c73a587706719d4e6d4c15bb1c9935fedcf8374b3e597a57ebca9a5b5cdf5d",
        "num": num_results
    }
    response = requests.get(url, params=params)
    data = response.json()

    urls = []
    for item in data.get("organic_results", []):
        if "link" in item:
            urls.append(item["link"])

    return urls

def generic_scrape(url):
    try:
        r = requests.get(url, timeout=10, headers={
            "User-Agent": "Mozilla/5.0"
        })
    except:
        return None

    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "footer", "nav"]):
        tag.extract()

    text = " ".join(soup.stripped_strings)
    text = re.sub(r"\s+", " ", text)
    title = soup.title.text if soup.title else "No title"

    return {
        "url": url,
        "title": title,
        "text": text[:3000]
    }

def run_general_scraper(keyword):
    print(f"Searching Google for: {keyword}")
    urls = google_search(keyword)

    scraped = []
    for url in urls:
        print(f"Scraping: {url}")
        data = generic_scrape(url)
        if data:
            scraped.append(data)

    return scraped

def scrape_and_save(company_name, location):
    keyword = f"{company_name} {location}"
    results = run_general_scraper(keyword)

    output_dir = "scraped_data"
    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.join(output_dir, f"{company_name.replace(' ', '_')}.json")

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    return filename

if __name__ == "__main__":
    query = "Dublin food"
    results = run_general_scraper(query)

    filename = query.replace(" ", "_") + ".json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\nSaved results to: {filename}")
