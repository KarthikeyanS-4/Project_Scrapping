"""
Script to scrape company details from websites and extract structured information using Gemini API.
"""

import random
import time
from typing import List, Dict
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
import pandas as pd
from keys import API_KEY

# Configure Gemini API
genai.configure(api_key=API_KEY)

MODELS = [
    "gemini-2.0-pro-exp",
    "gemini-1.5-pro",
    "gemini-2.0-flash",
]

# List of URLs to scrape
URLS: List[str] = [
    "https://lenovo.com",
    "https://www.gsk.com",
    "https://www.tcs.com",
    "https://www.ford.com",
    "https://www.siemens-energy.com",
    "https://www.americanexpress.com",
]

RELEVANT_LINKS: List[str] = [
    "home", "about", "contact", "services", "products", "Contact Us",
    "About Lenovo", "Investors", "Vehicles",
]


def get_relevant_links(base_url: str) -> List[str]:
    """
    Extract relevant page links from the homepage.

    Args:
        base_url (str): The base URL of the website.

    Returns:
        List[str]: A list of relevant links.
    """
    try:
        response = requests.get(base_url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as error:
        print(f"Error fetching {base_url}: {error}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    links: List[str] = []
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].lower()
        if any(keyword in href for keyword in RELEVANT_LINKS):
            full_url = requests.compat.urljoin(base_url, href)
            links.append(full_url)
            print(f"Found relevant link: {full_url}")
    return list(set(links))  # Remove duplicates


def scrape_text(url: str) -> str:
    """
    Scrape and clean text from a webpage.

    Args:
        url (str): The URL of the webpage.

    Returns:
        str: Cleaned text from the webpage.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as error:
        print(f"Error scraping {url}: {error}")
        return ""

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "header", "footer", "nav"]):
        tag.extract()
    text = " ".join(soup.stripped_strings)
    return text[:1000]  # Limit text length


def extract_details(text: str) -> str:
    """
    Use Gemini API to extract structured company details.

    Args:
        text (str): The text content to analyze.

    Returns:
        str: Extracted structured details in CSV format.
    """
    prompt = f"""
                Content:
                    {text}
                    You are a technical content writer for a person who is in the field of Market Research meaning
                        he wants constantly wants data of many companies That is why he wants to know about the company
                        -"What is the company's mission statement or core values?"
                        -"What products or services does the company offer?"
                        -"When was the company founded, and who were the founders?"
                        -"Where is the company's headquarters located?"
                        -"Who are the key executives or leadership team members?"
                        -"Has the company received any notable awards or recognitions?"
                        
                        
                        Output format should follow only text in csv format without any markdown elements and the first row should contain the questions and quoted for each question and answer and only one "" so that it can be recognized as a cell in csv.
                        Don't Quote the whole response, only the answers to the questions. And the questions which have comma in them should be quoted.
                        Answer these questions from the given text content.
                        No other text needs to be generated just the answers to these questions.
                        If the answer is not present in the text, then say "Not Available".
                        The answers should be human like and shouldn't explicitly say anything about the availability of information in the text.

                """
    try:
        model = random.choice(MODELS)
        llm = genai.GenerativeModel(model)
        response = llm.generate_content(prompt)
        return response.text.strip()
    except requests.RequestException as error:
        print(f"Error extracting details due to a request issue: {error}")
        return ""


def main() -> None:
    """
    Main function to scrape company details and save them to a CSV file.
    """
    results: List[Dict[str, str]] = []
    for url in URLS:
        print(f"Processing: {url}")
        relevant_pages = get_relevant_links(url)
        combined_text = " ".join(scrape_text(link) for link in relevant_pages)
        structured_data = extract_details(combined_text)
        results.append({"URL": url, "Extracted Details": structured_data})
        time.sleep(5)  # Prevent API rate limiting

    df = pd.DataFrame(results)
    df.to_csv("extracted_company_details_1.csv", index=False)
    print("Data saved to extracted_company_details.csv")


if __name__ == "__main__":
    main()
