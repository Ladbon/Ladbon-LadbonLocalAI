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
            timeout=20,  # Increased timeout for slower connections
            
        )
        resp.raise_for_status()
    except Exception as e:
        return f"Web search error: {str(e)}"
    
    soup = BeautifulSoup(resp.text, "html.parser")
    snippets = []
    
    # Extract search results using CSS selectors instead of find_all
    results = soup.select("div.result", limit=max_results)
    for res in results:
        title = res.select_one("a.result__a")
        text_snip = res.select_one("div.result__snippet")
        t = title.get_text() if title else ""
        s = text_snip.get_text() if text_snip else ""
        snippets.append(f"{t}: {s}")
    
    if not snippets:
        return "No search results found."
    
    return "\n\n".join(snippets)