"""Tool registry — import every LangChain tool for the agent."""
from __future__ import annotations

from tools.calculator import calculator
from tools.file_ops import delete_file, list_directory, read_file, write_file
from tools.rag_search import search_knowledge_base
from tools.shell import run_command
from tools.web_scrape import scrape_url
from tools.web_search import web_search

ALL_TOOLS = [
    calculator,
    read_file,
    write_file,
    list_directory,
    delete_file,
    run_command,
    web_search,
    scrape_url,
    search_knowledge_base,
]

__all__ = ["ALL_TOOLS"]
