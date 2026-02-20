import argparse
import json
from neverliie_ai_sdk import Google


def tavily_search(query: str, api_key: str) -> str:
    """Search the web using Tavily API."""
    try:
        import requests
        
        url = "https://api.tavily.com/search"
        headers = {
            "Content-Type": "application/json"
        }
        payload = {
            "api_key": api_key,
            "query": query,
            "search_depth": "basic",
            "include_answer": True,
            "max_results": 5
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract answer if available
        results = []
        if data.get("answer"):
            results.append(f"Answer: {data['answer']}")
        
        # Add search results
        if data.get("results"):
            results.append("\nSources:")
            for i, result in enumerate(data["results"][:3], 1):
                results.append(f"{i}. {result.get('title', 'No title')}")
                results.append(f"   {result.get('content', 'No content')[:200]}...")
                results.append(f"   URL: {result.get('url', 'No URL')}")
        
        return "\n".join(results) if results else "No results found."
        
    except ImportError:
        return "Error: requests library not installed. Run: pip install requests"
    except Exception as e:
        return f"Error searching: {str(e)}"


# Define web search tool for Mistral
web_search_tool = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current information, news, facts, or data. Use this when you need up-to-date information that might not be in your training data.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up on the web"
                }
            },
            "required": ["query"]
        }
    }
}


def handle_tool_calls(response, messages, client, model, tavily_api_key):
    """Handle tool calls from the model and return updated messages."""
    message = response["choices"][0]["message"]
    
    if "tool_calls" not in message:
        return None, message.get("content", "")
    
    print("\n[Executing tool calls...]")
    
    # Add assistant message to conversation
    messages.append(message)
    
    # Execute each tool call
    for tool_call in message["tool_calls"]:
        function_name = tool_call["function"]["name"]
        arguments = json.loads(tool_call["function"]["arguments"])
        
        print(f"[Tool: {function_name}] Query: {arguments.get('query', 'N/A')}")
        
        if function_name == "web_search":
            result = tavily_search(arguments.get("query"), api_key=tavily_api_key)
            print(f"[Search result received]")
        else:
            result = f"Unknown tool: {function_name}"
        
        # Add tool response to conversation
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call["id"],
            "name": function_name,
            "content": result
        })
    
    # Get final response from model
    final_response = client.chat(messages=messages, model=model, tools=[web_search_tool])
    return final_response, final_response["choices"][0]["message"].get("content", "")


def main():
    parser = argparse.ArgumentParser(description="Test Mistral API with Tavily Web Search")
    parser.add_argument("--api-key", required=True, help="Mistral API key")
    parser.add_argument("--message", required=True, help="Message to send")
    parser.add_argument("--model", default="mistral-small-latest", help="Model name")
    parser.add_argument("--stream", action="store_true", help="Use streaming")
    parser.add_argument("--tavily-key", default=None, help="Tavily API key (required with --ws)")
    parser.add_argument("--ws", action="store_true", help="Enable web search with Tavily")
    args = parser.parse_args()

    # Validate: if --ws is used, --tavily-key is required
    if args.ws and not args.tavily_key:
        parser.error("--tavily-key is required when using --ws")

    client = Google(api_key=args.api_key)
    # client = Mistral(api_key=args.api_key)  
    # client = OpenAI(api_key=args.api_key)
    # client = OpenAICompatible(api_key=args.api_key base_url="https://integrate.api.nvidia.com/v1")  # Example for Nvidia NIM with OpenAI-compatible interface
    # client = Anthropic(api_key=args.api_key)
    tavily_api_key = args.tavily_key

    if args.stream:
        if args.ws:
            print("=== Streaming with Web Search ===")
            # Stream with tool support - SDK now handles buffering
            messages = [{"role": "user", "content": args.message}]
            content_buffer = ""
            tool_calls = []
            
            for event in client.chat_stream(
                messages=messages,
                model=args.model,
                tools=[web_search_tool],
                tool_choice="auto"
            ):
                if event["type"] == "content":
                    # Print content in real-time
                    content = event["content"]
                    content_buffer += content
                    print(content, end="", flush=True)
                elif event["type"] == "tool_call":
                    # Collect tool calls
                    tool_calls.append(event["tool_call"])
                    print(f"\n[Tool: {event['tool_call']['function']['name']}]")
            
            # Execute tool calls if any
            if tool_calls:
                print("\n[Executing tool calls...]")
                
                # Build assistant message with tool calls
                assistant_msg = {
                    "role": "assistant",
                    "content": content_buffer if content_buffer else None,
                    "tool_calls": tool_calls
                }
                messages.append(assistant_msg)
                
                # Execute tool calls
                for tool_call in tool_calls:
                    function_name = tool_call["function"]["name"]
                    try:
                        arguments = json.loads(tool_call["function"]["arguments"])
                    except json.JSONDecodeError:
                        arguments = {}
                    
                    print(f"[Tool: {function_name}] Query: {arguments.get('query', 'N/A')}")
                    
                    if function_name == "web_search":
                        result = tavily_search(arguments.get("query"), api_key=tavily_api_key)
                        print(f"[Search result received]")
                    else:
                        result = f"Unknown tool: {function_name}"
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": function_name,
                        "content": result
                    })
                
                # Get final response
                print()  # Newline first
                for event in client.chat_stream(messages=messages, model=args.model, tools=[web_search_tool]):
                    if event["type"] == "content":
                        print(event["content"], end="", flush=True)
                    elif event["type"] == "tool_call":
                        print(f"\n[Tool call: {event['tool_call']['function']['name']}]")
            else:
                print() 
        else:
            print("=== Streaming ===")
            # Simple streaming without tools - SDK yields content events
            for event in client.chat_stream(messages=args.message, model=args.model):
                if event["type"] == "content":
                    print(event["content"], end="", flush=True)
            print()
    else:
        print("=== Chat Completion ===")
        
        if args.ws:
            # Chat with web search tool
            messages = [{"role": "user", "content": args.message}]
            response = client.chat(
                messages=messages,
                model=args.model,
                tools=[web_search_tool],
                tool_choice="auto"
            )
            
            # Check if tool calls were made
            if "tool_calls" in response["choices"][0]["message"]:
                _, final_content = handle_tool_calls(
                    response, messages, client, args.model, tavily_api_key
                )
                print(final_content)
            else:
                print(response["choices"][0]["message"]["content"])
        else:
            # Simple chat without tools (default)
            response = client.chat(messages=args.message, model=args.model)
            print(response["choices"][0]["message"]["content"])

    client.close()


if __name__ == "__main__":
    main()
