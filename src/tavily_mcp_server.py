"""FastMCP server for Tavily API — validated against official OpenAPI spec."""

import os
from typing import Literal, Any

from dotenv import load_dotenv
from fastmcp import FastMCP
from tavily import TavilyClient

# Load environment variables
load_dotenv()

API_KEY = os.getenv("TAVILY_API_KEY")

mcp = FastMCP("Tavily MCP Server")


def get_tavily_client() -> TavilyClient:
    """Return a validated TavilyClient."""
    if not API_KEY:
        raise ValueError(
            "TAVILY_API_KEY environment variable is required. "
            "Please set it before using this MCP server."
        )
    return TavilyClient(api_key=API_KEY)


# ---------------------------------------------------------------------------
# tavily_search
# ---------------------------------------------------------------------------

@mcp.tool()
def tavily_search(
    query: str,
    search_depth: Literal["basic", "advanced", "fast", "ultra-fast"] = "basic",
    topic: Literal["general", "news", "finance"] = "general",
    max_results: int = 5,
    include_answer: Literal["false", "basic", "advanced"] = "false",
    include_raw_content: Literal["false", "markdown", "text"] = "false",
    chunks_per_source: int = 3,
    time_range: Literal["day", "week", "month", "year"] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    include_images: bool = False,
    include_image_descriptions: bool = False,
    include_favicon: bool = False,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
    country: str | None = None,
    auto_parameters: bool = False,
    exact_match: bool = False,
    include_usage: bool = False,
) -> dict[str, Any]:
    """Execute a search query using Tavily Search.

    Args:
        query: The search query string (required).
        search_depth: Controls latency vs relevance tradeoff.
            'basic' – balanced, 1 credit.
            'advanced' – highest relevance, 2 credits.
            'fast' – lower latency, 1 credit.
            'ultra-fast' – minimum latency, 1 credit.
        topic: Category of search.
            'general' – broad web search.
            'news' – real-time news updates.
            'finance' – financial data.
        max_results: Maximum results to return (0–20, default 5).
        include_answer: Include an LLM-generated answer.
            'false' – no answer.
            'basic' – quick answer.
            'advanced' – detailed answer.
        include_raw_content: Include cleaned HTML content per result.
            'false' – omit. 'markdown' – markdown format. 'text' – plain text.
        chunks_per_source: Max content chunks per source (1–3). Only applies
            when search_depth is 'advanced'.
        time_range: Filter results by publish date ('day','week','month','year').
            Conflicts with start_date/end_date — only one can be used.
        start_date: Return results published after this date (YYYY-MM-DD).
        end_date: Return results published before this date (YYYY-MM-DD).
        include_images: Also run an image search and include results.
        include_image_descriptions: Add descriptive text for each image
            (requires include_images=True).
        include_favicon: Include favicon URL for each result.
        include_domains: List of domains to specifically include (max 300).
        exclude_domains: List of domains to specifically exclude (max 150).
        country: Boost results from a specific country. Only valid when
            topic='general'. Example values: 'united states', 'india', 'germany'.
        auto_parameters: Let Tavily auto-configure search parameters based on
            query intent. Costs 2 credits per request. Explicit values override
            auto ones, except include_answer, include_raw_content, and
            max_results which must always be set manually.
        exact_match: Return only results containing the exact quoted phrase(s)
            from the query.
        include_usage: Include credit usage info in the response.

    Returns:
        dict with keys: query, answer, images, results, response_time,
        and optionally auto_parameters, usage, request_id.
    """
    try:
        client = get_tavily_client()

        params: dict[str, Any] = {
            "query": query,
            "search_depth": search_depth,
            "topic": topic,
            "max_results": max_results,
            "include_images": include_images,
            "include_image_descriptions": include_image_descriptions,
            "include_favicon": include_favicon,
            "auto_parameters": auto_parameters,
            "exact_match": exact_match,
            "include_usage": include_usage,
        }

        # include_answer: convert "false" sentinel to Python False
        params["include_answer"] = False if include_answer == "false" else include_answer

        # include_raw_content: convert "false" sentinel to Python False
        params["include_raw_content"] = False if include_raw_content == "false" else include_raw_content

        # chunks_per_source only applies to advanced search_depth
        if search_depth == "advanced":
            params["chunks_per_source"] = max(1, min(3, chunks_per_source))

        # Date filtering — time_range conflicts with start_date/end_date
        if start_date or end_date:
            if start_date:
                params["start_date"] = start_date
            if end_date:
                params["end_date"] = end_date
        elif time_range:
            params["time_range"] = time_range

        if include_domains:
            params["include_domains"] = include_domains
        if exclude_domains:
            params["exclude_domains"] = exclude_domains
        if country and topic == "general":
            params["country"] = country

        return client.search(**params)

    except Exception as e:
        return {"error": str(e), "message": f"tavily_search failed: {e}"}


# ---------------------------------------------------------------------------
# tavily_extract
# ---------------------------------------------------------------------------

@mcp.tool()
def tavily_extract(
    urls: list[str],
    extract_depth: Literal["basic", "advanced"] = "basic",
    output_format: Literal["markdown", "text"] = "markdown",
    include_images: bool = False,
    include_favicon: bool = False,
    query: str | None = None,
    chunks_per_source: int = 3,
    timeout: float | None = None,
    include_usage: bool = False,
) -> dict[str, Any]:
    """Extract web page content from one or more URLs using Tavily Extract.

    Cost: 1 credit per 5 successful URL extractions (basic);
          2 credits per 5 (advanced).

    Args:
        urls: One or more URLs to extract content from (max 20, required).
        extract_depth: Extraction depth.
            'basic' – standard extraction, 1 credit per 5 URLs.
            'advanced' – richer extraction including tables/embedded content,
            2 credits per 5 URLs. Use for LinkedIn or protected pages.
        output_format: Format of extracted content.
            'markdown' – markdown (default). 'text' – plain text (may increase
            latency).
        include_images: Include images extracted from each page.
        include_favicon: Include favicon URL for each result.
        query: When provided, chunks are reranked by relevance to this query.
            Also enables chunks_per_source.
        chunks_per_source: Max relevant chunks per source (1–5). Only applies
            when query is provided.
        timeout: Max seconds to wait per URL extraction (1.0–60.0).
            Defaults to 10 s (basic) or 30 s (advanced) when omitted.
        include_usage: Include credit usage info in the response.

    Returns:
        dict with keys: results (list of {url, raw_content, images, favicon}),
        failed_results, response_time, and optionally usage, request_id.
    """
    try:
        client = get_tavily_client()

        params: dict[str, Any] = {
            "urls": urls,
            "extract_depth": extract_depth,
            "format": output_format,
            "include_images": include_images,
            "include_favicon": include_favicon,
            "include_usage": include_usage,
        }

        if query:
            params["query"] = query
            params["chunks_per_source"] = max(1, min(5, chunks_per_source))

        if timeout is not None:
            params["timeout"] = max(1.0, min(60.0, timeout))

        return client.extract(**params)

    except Exception as e:
        return {"error": str(e), "message": f"tavily_extract failed: {e}"}


# ---------------------------------------------------------------------------
# tavily_crawl
# ---------------------------------------------------------------------------

@mcp.tool()
def tavily_crawl(
    url: str,
    max_depth: int = 1,
    max_breadth: int = 20,
    limit: int = 50,
    instructions: str | None = None,
    chunks_per_source: int = 3,
    select_paths: list[str] | None = None,
    select_domains: list[str] | None = None,
    exclude_paths: list[str] | None = None,
    exclude_domains: list[str] | None = None,
    allow_external: bool = True,
    include_images: bool = False,
    extract_depth: Literal["basic", "advanced"] = "basic",
    output_format: Literal["markdown", "text"] = "markdown",
    include_favicon: bool = False,
    timeout: float = 150.0,
    include_usage: bool = False,
) -> dict[str, Any]:
    """Crawl a website starting from a base URL using Tavily Crawl.

    Tavily Crawl is a graph-based traversal tool that explores hundreds of
    paths in parallel with built-in extraction and intelligent discovery.

    Cost: 1 credit per 10 successful pages;
          2 credits per 10 when instructions are provided.

    Args:
        url: Root URL to begin the crawl (required).
        max_depth: How many link-hops from the base URL to explore (1–5).
        max_breadth: Max links to follow per page (1–500).
        limit: Total links to process before stopping (minimum 1).
        instructions: Natural language instructions to guide crawl focus.
            Doubles cost to 2 credits per 10 pages.
        chunks_per_source: Max content chunks per source (1–5). Only applies
            when instructions are provided.
        select_paths: Regex patterns — only crawl URLs matching these paths
            (e.g. '/docs/.*', '/api/v1.*').
        select_domains: Regex patterns — restrict crawl to these domains
            (e.g. '^docs\\.example\\.com$').
        exclude_paths: Regex patterns — skip URLs matching these paths
            (e.g. '/private/.*', '/admin/.*').
        exclude_domains: Regex patterns — skip these domains
            (e.g. '^private\\.example\\.com$').
        allow_external: Include external domain links in results.
        include_images: Include images in crawl results.
        extract_depth: 'basic' – standard; 'advanced' – richer content
            including tables/embedded media (higher latency).
        output_format: Content format. 'markdown' (default) or 'text'.
        include_favicon: Include favicon URL for each result.
        timeout: Max seconds for the crawl operation (10–150).
        include_usage: Include credit usage info in the response.

    Returns:
        dict with keys: base_url, results (list of {url, raw_content, favicon}),
        response_time, and optionally usage, request_id.
    """
    try:
        client = get_tavily_client()

        params: dict[str, Any] = {
            "url": url,
            "max_depth": max_depth,
            "max_breadth": max_breadth,
            "limit": limit,
            "allow_external": allow_external,
            "include_images": include_images,
            "extract_depth": extract_depth,
            "format": output_format,
            "include_favicon": include_favicon,
            "timeout": max(10.0, min(150.0, timeout)),
            "include_usage": include_usage,
        }

        if instructions:
            params["instructions"] = instructions
            params["chunks_per_source"] = max(1, min(5, chunks_per_source))
        if select_paths:
            params["select_paths"] = select_paths
        if select_domains:
            params["select_domains"] = select_domains
        if exclude_paths:
            params["exclude_paths"] = exclude_paths
        if exclude_domains:
            params["exclude_domains"] = exclude_domains

        return client.crawl(**params)

    except Exception as e:
        return {"error": str(e), "message": f"tavily_crawl failed: {e}"}


# ---------------------------------------------------------------------------
# tavily_map
# ---------------------------------------------------------------------------

@mcp.tool()
def tavily_map(
    url: str,
    max_depth: int = 1,
    max_breadth: int = 20,
    limit: int = 50,
    instructions: str | None = None,
    select_paths: list[str] | None = None,
    select_domains: list[str] | None = None,
    exclude_paths: list[str] | None = None,
    exclude_domains: list[str] | None = None,
    allow_external: bool = True,
    timeout: float = 150.0,
    include_usage: bool = False,
) -> dict[str, Any]:
    """Map a website's URL structure using Tavily Map.

    Tavily Map traverses websites like a graph and explores hundreds of paths
    in parallel to generate comprehensive site maps.

    Cost: 1 credit per 10 successful pages;
          2 credits per 10 when instructions are provided.

    Args:
        url: Root URL to begin the mapping (required).
        max_depth: How many link-hops from the base URL to explore (1–5).
        max_breadth: Max links to follow per page (1–500).
        limit: Total links to process before stopping (minimum 1).
        instructions: Natural language instructions to guide link selection.
            Doubles cost to 2 credits per 10 pages.
        select_paths: Regex patterns — only map URLs matching these paths
            (e.g. '/docs/.*', '/api/v1.*').
        select_domains: Regex patterns — restrict mapping to these domains
            (e.g. '^docs\\.example\\.com$').
        exclude_paths: Regex patterns — skip URLs matching these paths
            (e.g. '/private/.*', '/admin/.*').
        exclude_domains: Regex patterns — skip these domains
            (e.g. '^private\\.example\\.com$').
        allow_external: Include external domain links in results.
        timeout: Max seconds for the mapping operation (10–150).
        include_usage: Include credit usage info in the response.

    Returns:
        dict with keys: base_url, results (list of discovered URL strings),
        response_time, and optionally usage, request_id.
    """
    try:
        client = get_tavily_client()

        params: dict[str, Any] = {
            "url": url,
            "max_depth": max_depth,
            "max_breadth": max_breadth,
            "limit": limit,
            "allow_external": allow_external,
            "timeout": max(10.0, min(150.0, timeout)),
            "include_usage": include_usage,
        }

        if instructions:
            params["instructions"] = instructions
        if select_paths:
            params["select_paths"] = select_paths
        if select_domains:
            params["select_domains"] = select_domains
        if exclude_paths:
            params["exclude_paths"] = exclude_paths
        if exclude_domains:
            params["exclude_domains"] = exclude_domains

        return client.map(**params)

    except Exception as e:
        return {"error": str(e), "message": f"tavily_map failed: {e}"}


# ---------------------------------------------------------------------------
# tavily_research
# ---------------------------------------------------------------------------

@mcp.tool()
def tavily_research(
    input: str,
    model: Literal["mini", "pro", "auto"] = "auto",
    stream: bool = False,
    output_schema: dict | None = None,
    citation_format: Literal["numbered", "mla", "apa", "chicago"] = "numbered",
) -> dict[str, Any]:
    """Perform comprehensive multi-source research using Tavily Research.

    Conducts multiple searches, analyses sources, and returns a detailed
    research report. The Python SDK handles the async polling internally.

    Args:
        input: The research task or question to investigate (required).
        model: Research depth.
            'mini' – targeted, efficient; best for narrow focused questions.
            'pro' – comprehensive, multi-angle; best for broad complex topics.
            'auto' – automatically selects mini or pro based on the query.
        stream: When True, returns Server-Sent Events for real-time progress.
            The SSE stream emits tool_call events (Planning, WebSearch,
            ResearchSubtopic, Generating), content chunks, and a final sources
            event, then a 'done' sentinel.
        output_schema: Optional JSON Schema defining the expected output
            structure. Must include a 'properties' field. When provided,
            'content' in the response will be a structured dict instead of
            a markdown string. Example:
            {
                "properties": {
                    "company": {"type": "string", "description": "Company name"},
                    "revenue":  {"type": "number", "description": "Annual revenue"}
                },
                "required": ["company"]
            }
        citation_format: Citation style for the report.
            'numbered' (default) | 'mla' | 'apa' | 'chicago'.

    Returns:
        Non-streaming: dict with request_id, created_at, status, input, model,
        response_time. Poll GET /research/{request_id} for the completed report.
        When the SDK resolves: dict with status='completed', content (string or
        structured object), sources (list of {url, title, favicon}),
        response_time.
    """
    try:
        client = get_tavily_client()

        params: dict[str, Any] = {
            "input": input,
            "model": model,
            "stream": stream,
            "citation_format": citation_format,
        }

        if output_schema is not None:
            params["output_schema"] = output_schema

        return client.research(**params)

    except Exception as e:
        return {"error": str(e), "message": f"tavily_research failed: {e}"}


# ---------------------------------------------------------------------------
# Resource — API status
# ---------------------------------------------------------------------------

@mcp.resource("tavily://status")
def tavily_status() -> str:
    """Check Tavily API configuration status."""
    if not API_KEY:
        return (
            "Tavily API Status: NOT CONFIGURED\n\n"
            "TAVILY_API_KEY environment variable is not set.\n"
            "Get your free API key at: https://app.tavily.com\n"
            "Then add TAVILY_API_KEY=tvly-... to your .env file."
        )
    masked = f"{API_KEY[:8]}...{API_KEY[-4:]}"
    return (
        f"Tavily API Status: CONFIGURED\n"
        f"API Key: {masked}\n\n"
        f"Available tools:\n"
        f"  tavily_search   – web search with 19 parameters\n"
        f"  tavily_extract  – URL content extraction\n"
        f"  tavily_crawl    – graph-based website crawling\n"
        f"  tavily_map      – website URL structure mapping\n"
        f"  tavily_research – comprehensive multi-source research\n"
    )


if __name__ == "__main__":
    mcp.run()
