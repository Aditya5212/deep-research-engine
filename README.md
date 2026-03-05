# Deep Research Engine

A FastAPI-based research engine with integrated MCP (Model Context Protocol) servers.

## Features

- **Research Service**: Core research functionality with agent-based processing
- **MCP Server**: Standard MCP server with sample tools and resources
- **Tavily MCP Server**: Complete integration with Tavily API for web search, content extraction, crawling, and research

## Installation

1. Install dependencies using uv:
```bash
uv sync
```

2. Copy the environment variables template:
```bash
cp .env.example .env
```

3. Configure your environment variables in `.env`:
   - `GOOGLE_API_KEY`: Your Google API key for research services
   - `TAVILY_API_KEY`: Your Tavily API key (get it at https://tavily.com)
   - Database credentials for PostgreSQL

## Running the Server

Start the FastAPI server:
```bash
uvicorn main:app --reload
```

The server will be available at:
- Main API: http://localhost:8000
- MCP Server: http://localhost:8000/mcp
- Tavily MCP Server: http://localhost:8000/tavily
- API Docs: http://localhost:8000/docs

## MCP Servers

### Standard MCP Server (http://localhost:8000/mcp)

Sample tools for demonstration:
- `research_status`: Get the status of a research task
- `list_research_tasks`: List all research tasks

### Tavily MCP Server (http://localhost:8000/tavily)

Complete Tavily API integration with the following tools:

#### tavily_search
Search the web for current information with advanced options:
- **search_depth**: `basic`, `advanced`, `fast`, `ultra-fast`
- **time_range**: Filter by `day`, `week`, `month`, `year`
- **date filtering**: Use `start_date` and `end_date` (YYYY-MM-DD)
- **domain filtering**: Include or exclude specific domains
- **max_results**: 5-20 results
- **include_images**: Get related images
- **country**: Boost results from specific countries

Example:
```python
{
    "query": "latest AI developments",
    "search_depth": "advanced",
    "max_results": 10,
    "include_images": true,
    "time_range": "week"
}
```

#### tavily_extract
Extract and parse content from URLs:
- **extract_depth**: `basic` or `advanced` (for tables, embedded content)
- **format**: `markdown` or `text`
- **query**: Rerank content by relevance

Example:
```python
{
    "urls": ["https://example.com/article"],
    "extract_depth": "advanced",
    "format": "markdown"
}
```

#### tavily_crawl
Crawl websites with depth and breadth control:
- **max_depth**: How deep to crawl from the base URL
- **max_breadth**: Links to follow per page
- **limit**: Total links to process
- **instructions**: Natural language filtering
- **select_paths**: Regex patterns for path filtering
- **select_domains**: Domain restrictions

Example:
```python
{
    "url": "https://docs.example.com",
    "max_depth": 2,
    "max_breadth": 10,
    "select_paths": ["/docs/.*"]
}
```

#### tavily_map
Map website structure and discover URLs:
- Returns list of discovered URLs
- Same configuration options as crawl
- Lightweight site analysis

#### tavily_research
Comprehensive multi-source research:
- **model**: `mini`, `pro`, or `auto`
- Returns detailed research synthesis
- Polls for completion automatically

Example:
```python
{
    "input": "What are the latest developments in quantum computing?",
    "model": "pro"
}
```

### Resources

- `tavily://status`: Check Tavily API configuration status

## API Endpoints

- `GET /`: Welcome message
- `GET /health`: Health check endpoint
- Research endpoints: See `/docs` for full API documentation

## Development

The project structure:
```
deep-research-engine/
├── main.py                      # FastAPI application entry point
├── src/
│   ├── mcp_server.py           # Standard MCP server
│   ├── tavily_mcp_server.py    # Tavily MCP integration
│   └── research/               # Research service modules
│       ├── agent.py
│       ├── checkpointer.py
│       ├── controller.py
│       ├── models.py
│       └── service.py
├── pyproject.toml              # Project dependencies
├── docker-compose.yml          # Docker configuration
└── .env.example                # Environment variables template
```

## License

MIT
