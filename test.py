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


def execute_tool_call(function_name, arguments, tavily_api_key):
    """Execute a tool call and return the result."""
    if function_name == "web_search":
        result = tavily_search(arguments.get("query"), api_key=tavily_api_key)
    else:
        result = f"Unknown tool: {function_name}"
    return result


def handle_tool_calls(response, messages, client, model, tavily_api_key, max_iterations=10):
    """Handle tool calls from the model, supporting multiple rounds of tool calls."""
    current_response = response
    
    for iteration in range(max_iterations):
        message = current_response["choices"][0]["message"]
        
        if "tool_calls" not in message:
            return current_response, message.get("content") or ""
        
        print(f"\n[Executing tool calls... (round {iteration + 1})]")
        messages.append(message)
        
        for tool_call in message["tool_calls"]:
            function_name = tool_call["function"]["name"]
            arguments = tool_call["function"]["arguments"]
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            
            print(f"[Tool: {function_name}] Query: {arguments.get('query', 'N/A')}")
            result = execute_tool_call(function_name, arguments, tavily_api_key)
            print(f"[Search result received]")
            
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "name": function_name,
                "content": result
            })
        
        current_response = client.chat(messages=messages, model=model, tools=[web_search_tool])
    
    print(f"\n[Warning: Reached max iterations ({max_iterations})]")
    return current_response, current_response["choices"][0]["message"].get("content") or ""


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
                    content = event["content"]
                    content_buffer += content
                    print(content, end="", flush=True)
                elif event["type"] == "tool_call":
                    tool_calls.append(event["tool_call"])
                    print(f"\n[Tool call: {event['tool_call']['function']['name']}]")
            
            while tool_calls:
                assistant_msg = {
                    "role": "assistant",
                    "content": content_buffer if content_buffer else None,
                    "tool_calls": tool_calls
                }
                messages.append(assistant_msg)
                
                for tool_call in tool_calls:
                    function_name = tool_call["function"]["name"]
                    try:
                        arguments = tool_call["function"]["arguments"]
                        if isinstance(arguments, str):
                            arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                    
                    print(f"\n[Tool: {function_name}] Query: {arguments.get('query', 'N/A')}")
                    result = execute_tool_call(function_name, arguments, tavily_api_key)
                    print(f"[Search result received]")
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": function_name,
                        "content": result
                    })
                
                tool_calls = []
                content_buffer = ""
                
                for event in client.chat_stream(messages=messages, model=args.model, tools=[web_search_tool]):
                    if event["type"] == "content":
                        print(event["content"], end="", flush=True)
                        content_buffer += event["content"]
                    elif event["type"] == "tool_call":
                        tool_calls.append(event["tool_call"])
                        print(f"\n[Tool call: {event['tool_call']['function']['name']}]")
                print()
        else:
            for event in client.chat_stream(messages=args.message, model=args.model):
                if event["type"] == "content":
                    print(event["content"], end="", flush=True)
            print()
    else:
        if args.ws:
            messages = [{"role": "user", "content": args.message}]
            response = client.chat(
                messages=messages,
                model=args.model,
                tools=[web_search_tool],
                tool_choice="auto"
            )
            
            if "tool_calls" in response["choices"][0]["message"]:
                _, final_content = handle_tool_calls(
                    response, messages, client, args.model, tavily_api_key
                )
                print(final_content)
            else:
                print(response["choices"][0]["message"]["content"])
        else:
            response = client.chat(messages=args.message, model=args.model)
            print(response["choices"][0]["message"]["content"])

    client.close()


if __name__ == "__main__":
    main()
