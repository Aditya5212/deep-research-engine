"""
Test NVIDIA models before wiring into the agent.

Tests:
  1. z-ai/glm4.7            via ChatNVIDIA (current agent model)
  2. qwen/qwen3.5-122b      via ChatNVIDIA (candidate replacement — tool-capable)
  3. moonshotai/kimi-k2     via OpenAI client — basic streaming
  4. moonshotai/kimi-k2     via OpenAI client — tool call round-trip

Run:
    uv run python test_nvidia_model.py
"""
import os
from dotenv import load_dotenv

load_dotenv()

PROMPT = "What is 2 + 2? Answer in one sentence."
OLD_KEY = os.getenv("NVIDIA_API_KEY_GLM47")
NEW_KEY = os.getenv("NVIDIA_API_KEY_QWEN")
KIMI_KEY = os.getenv("NVIDIA_API_KEY_KIMI")
KIMI_MODEL = "moonshotai/kimi-k2-instruct"
NVIDIA_BASE = "https://integrate.api.nvidia.com/v1"


# ---------------------------------------------------------------------------
# 1. Current model — z-ai/glm4.7
# ---------------------------------------------------------------------------
def test_glm47():
    print("\n" + "=" * 60)
    print("TEST 1: z-ai/glm4.7  (current agent model, via ChatNVIDIA)")
    print("=" * 60)
    try:
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        model = ChatNVIDIA(
            model="z-ai/glm4.7",
            api_key=OLD_KEY,
            temperature=1,
            top_p=1,
            max_tokens=256,
        )
        response = model.invoke(PROMPT)
        print("PASS — content:", response.content)
        print("usage_metadata:", response.usage_metadata)
        print("response_metadata:", response.response_metadata)
    except Exception as exc:
        print("FAIL —", exc)


# ---------------------------------------------------------------------------
# 2. Candidate model — qwen/qwen3.5-122b via ChatNVIDIA
# ---------------------------------------------------------------------------
def test_qwen_langchain():
    print("\n" + "=" * 60)
    print("TEST 2: qwen/qwen3.5-122b  (candidate, via ChatNVIDIA)")
    print("=" * 60)
    try:
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        model = ChatNVIDIA(
            model="qwen/qwen3.5-122b-a10b",
            api_key=NEW_KEY,
            temperature=0.6,
            top_p=0.95,
            max_tokens=256,
        )
        response = model.invoke(PROMPT)
        print("PASS — content:", response.content)
        print("usage_metadata:", response.usage_metadata)
        print("response_metadata:", response.response_metadata)
    except Exception as exc:
        print("FAIL —", exc)


# ---------------------------------------------------------------------------
# 3. Candidate model — qwen/qwen3.5-122b via raw streaming HTTP (reference)
# ---------------------------------------------------------------------------
def test_qwen_raw():
    print("\n" + "=" * 60)
    print("TEST 3: qwen/qwen3.5-122b  (raw streaming HTTP, reference)")
    print("=" * 60)
    import requests

    headers = {
        "Authorization": f"Bearer {NEW_KEY}",
        "Accept": "text/event-stream",
    }
    payload = {
        "model": "qwen/qwen3.5-122b-a10b",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 256,
        "temperature": 0.6,
        "top_p": 0.95,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        resp = requests.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            headers=headers,
            json=payload,
            stream=True,
            timeout=30,
        )
        resp.raise_for_status()
        print("PASS — streaming response:")
        for line in resp.iter_lines():
            if line:
                text = line.decode("utf-8")
                if text.startswith("data: ") and text != "data: [DONE]":
                    import json
                    chunk = json.loads(text[6:])
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        print(delta, end="", flush=True)
        print()
    except Exception as exc:
        print("FAIL —", exc)


# ---------------------------------------------------------------------------
# 4. kimi-k2-instruct — basic streaming via OpenAI client
# ---------------------------------------------------------------------------
def test_kimi_stream():
    print("\n" + "=" * 60)
    print("TEST 4: moonshotai/kimi-k2-instruct  (streaming, OpenAI client)")
    print("=" * 60)
    try:
        from openai import OpenAI
        client = OpenAI(base_url=NVIDIA_BASE, api_key=KIMI_KEY)
        completion = client.chat.completions.create(
            model=KIMI_MODEL,
            messages=[{"role": "user", "content": PROMPT}],
            temperature=0.6,
            top_p=0.9,
            max_tokens=256,
            stream=True,
        )
        collected = []
        for chunk in completion:
            if not getattr(chunk, "choices", None):
                continue
            delta = chunk.choices[0].delta.content
            if delta is not None:
                print(delta, end="", flush=True)
                collected.append(delta)
        print()
        print("PASS — total chars:", sum(len(c) for c in collected))
    except Exception as exc:
        print("FAIL —", exc)


# ---------------------------------------------------------------------------
# 5. kimi-k2-instruct — tool call round-trip via OpenAI client
# ---------------------------------------------------------------------------
def test_kimi_tool_call():
    print("\n" + "=" * 60)
    print("TEST 5: moonshotai/kimi-k2-instruct  (tool call round-trip)")
    print("=" * 60)
    import json
    try:
        from openai import OpenAI
        client = OpenAI(base_url=NVIDIA_BASE, api_key=KIMI_KEY)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for current information.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The search query"},
                        },
                        "required": ["query"],
                    },
                },
            }
        ]

        messages = [
            {"role": "user", "content": "What's the latest news about AI? Use the web_search tool."}
        ]

        # Step 1 — model should request a tool call
        resp = client.chat.completions.create(
            model=KIMI_MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.6,
            top_p=0.9,
            max_tokens=512,
        )
        msg = resp.choices[0].message
        print("Step 1 finish_reason:", resp.choices[0].finish_reason)
        print("Step 1 tool_calls:", msg.tool_calls)

        if not msg.tool_calls:
            print("WARN — model did not request a tool call; treating as direct answer.")
            print("Content:", msg.content)
            return

        # Step 2 — simulate tool result and send back
        tool_call = msg.tool_calls[0]
        tool_args = json.loads(tool_call.function.arguments)
        fake_result = f"[Simulated result for query: {tool_args.get('query', '')}]"
        print(f"Tool called: {tool_call.function.name}({tool_args})")

        messages.append(msg)  # assistant turn with tool calls
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": fake_result,
        })

        resp2 = client.chat.completions.create(
            model=KIMI_MODEL,
            messages=messages,
            tools=tools,
            temperature=0.6,
            top_p=0.9,
            max_tokens=512,
        )
        final = resp2.choices[0].message.content
        print("Step 2 final answer:", final)
        print("usage:", resp2.usage)
        print("PASS")
    except Exception as exc:
        print("FAIL —", exc)


if __name__ == "__main__":
    # Tests 1-3 (glm4.7 / qwen) already validated in previous run — skipping to save time
    test_kimi_stream()
    test_kimi_tool_call()
    print("\nDone.")
