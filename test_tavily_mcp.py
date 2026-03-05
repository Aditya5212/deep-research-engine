"""
Quick test script for Tavily MCP Server

This script demonstrates how to use the Tavily MCP tools.
Make sure to set TAVILY_API_KEY in your .env file before running.
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the Tavily MCP tools
from src.tavily_mcp_server import (
    tavily_search,
    tavily_extract,
    tavily_status,
    get_tavily_client
)


def test_status():
    """Test the Tavily API status."""
    print("=" * 60)
    print("Testing Tavily Status")
    print("=" * 60)
    status = tavily_status()
    print(status)
    print()


def test_search():
    """Test the Tavily search tool."""
    print("=" * 60)
    print("Testing Tavily Search")
    print("=" * 60)
    
    if not os.getenv("TAVILY_API_KEY"):
        print("⚠️  TAVILY_API_KEY not set. Skipping search test.")
        return
    
    # Simple search
    result = tavily_search(
        query="Python programming language",
        max_results=3,
        search_depth="basic"
    )
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
    else:
        print(f"✅ Search successful!")
        print(f"   Query: {result.get('query', 'N/A')}")
        print(f"   Results: {len(result.get('results', []))}")
        if result.get('answer'):
            print(f"   Answer: {result['answer'][:100]}...")
    print()


def test_extract():
    """Test the Tavily extract tool."""
    print("=" * 60)
    print("Testing Tavily Extract")
    print("=" * 60)
    
    if not os.getenv("TAVILY_API_KEY"):
        print("⚠️  TAVILY_API_KEY not set. Skipping extract test.")
        return
    
    result = tavily_extract(
        urls=["https://www.python.org/about/"],
        format="markdown",
        extract_depth="basic"
    )
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
    else:
        print(f"✅ Extract successful!")
        print(f"   URLs processed: {len(result.get('results', []))}")
    print()


async def test_research():
    """Test the Tavily research tool."""
    print("=" * 60)
    print("Testing Tavily Research")
    print("=" * 60)
    
    if not os.getenv("TAVILY_API_KEY"):
        print("⚠️  TAVILY_API_KEY not set. Skipping research test.")
        return
    
    from src.tavily_mcp_server import tavily_research
    
    result = await tavily_research(
        input="What is FastMCP and how does it work?",
        model="mini"
    )
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
    else:
        print(f"✅ Research successful!")
        if result.get('content'):
            print(f"   Content length: {len(result['content'])} characters")
            print(f"   Preview: {result['content'][:150]}...")
    print()


async def main():
    """Run all tests."""
    print("\n🧪 Tavily MCP Server Test Suite\n")
    
    # Test status (works without API key)
    test_status()
    
    # Test search (requires API key)
    test_search()
    
    # Test extract (requires API key)
    test_extract()
    
    # Test research (requires API key, async)
    await test_research()
    
    print("=" * 60)
    print("✅ Test suite completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
