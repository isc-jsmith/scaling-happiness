from langchain_community.tools import WikipediaQueryRun
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain_community.utilities import (
    ArxivAPIWrapper,
    PubMedAPIWrapper,
    WikipediaAPIWrapper,
)
from langchain_core.tools import Tool
# import custom_tools

def create_web_tools():
    """
    Create a collection of web-related tools that can be
    plugged into an agent.

    These do not require additional configuration beyond
    network access.
    """
    duckduckgo_search = DuckDuckGoSearchRun()

    wikipedia_wrapper = WikipediaAPIWrapper()
    wikipedia_search = WikipediaQueryRun(api_wrapper=wikipedia_wrapper)

    pubmed_wrapper = PubMedAPIWrapper()
    pubmed_search = Tool(
        name="PubMed Search",
        func=pubmed_wrapper.run,
        description="Search PubMed for clinical and biomedical literature.",
    )

    arxiv_wrapper = ArxivAPIWrapper()
    arxiv_search = Tool(
        name="Arxiv Search",
        func=arxiv_wrapper.run,
        description=(
            "Search Arxiv for scholarly articles. "
            "Use for background on clinical conditions, methods, or models."
        ),
    )

    
    return {
        "duckduckgo": duckduckgo_search,
        "wikipedia": wikipedia_search,
        "pubmed": pubmed_search,
        "arxiv": arxiv_search
        
    }

