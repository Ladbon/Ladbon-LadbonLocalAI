import requests
from bs4 import BeautifulSoup

def search(query: str, max_results: int = 5) -> str:
    """
    Search the web using DuckDuckGo and return snippets
    """
    try:
        resp = requests.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query},
            timeout=10
        )
        resp.raise_for_status()
    except Exception as e:
        return f"Web search error: {str(e)}"
    
    soup = BeautifulSoup(resp.text, "html.parser")  # Use html.parser if lxml not available
    snippets = []
    
    # Extract search results
    results = soup.find_all("div", class_="result", limit=max_results)
    for res in results:
        title = res.find("a", class_="result__a")
        text_snip = res.find("div", class_="result__snippet")
        t = title.get_text() if title else ""
        s = text_snip.get_text() if text_snip else ""
        snippets.append(f"{t}: {s}")
    
    if not snippets:
        return "No search results found."
    
    return "\n\n".join(snippets)