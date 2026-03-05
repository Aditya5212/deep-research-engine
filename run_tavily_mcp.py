"""
Standalone Tavily MCP Server

This script runs the Tavily MCP server as a standalone MCP server using stdio transport.
Use this for testing or integrating with MCP clients like Claude Desktop.

Usage:
    python run_tavily_mcp.py
"""

import sys
from src.tavily_mcp_server import mcp

if __name__ == "__main__":
    # Check if user wants to list tools
    if len(sys.argv) > 1 and sys.argv[1] == "--list-tools":
        print("Available Tavily MCP Tools:")
        print("\n1. tavily_search")
        print("   - Search the web for current information")
        print("   - Parameters: query (required), search_depth, topic, time_range, etc.")
        print("\n2. tavily_extract")
        print("   - Extract content from URLs")
        print("   - Parameters: urls (required), extract_depth, format, etc.")
        print("\n3. tavily_crawl")
        print("   - Crawl websites with configurable depth")
        print("   - Parameters: url (required), max_depth, max_breadth, etc.")
        print("\n4. tavily_map")
        print("   - Map website structure")
        print("   - Parameters: url (required), max_depth, max_breadth, etc.")
        print("\n5. tavily_research")
        print("   - Perform comprehensive research")
        print("   - Parameters: input (required), model")
        print("\nResources:")
        print("   - tavily://status: Check API configuration")
        sys.exit(0)
    
    # Run the MCP server
    print("Starting Tavily MCP Server...", file=sys.stderr)
    mcp.run()
