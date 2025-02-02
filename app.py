# Import necessary modules and setup for FastAPI, LangGraph, and LangChain
# FastAPI framework for creating the web application
from fastapi import FastAPI  
# BaseModel for structured data data models
from pydantic import BaseModel
# List type hint for type annotations 
from typing import List 
# TavilySearchResults tool for handling search results from Tavily
from langchain_community.tools.tavily_search import TavilySearchResults
# os module for environment variable handling  
import os
# Function to create a ReAct agent  
from langgraph.prebuilt import create_react_agent 
# ChatGroq class for interacting with LLMs 
from langchain_groq import ChatGroq
import uvicorn
# Load environment variables from .env file
from dotenv import load_dotenv 
 # take environment variables from .env.
load_dotenv() 

# load the API key from the environment variable
groq_api_key = os.getenv("GROQ_API_KEY")
travily_api_key = os.getenv("TAVILY_API_KEY")


# Predefined list of supported model names
MODEL_NAMES = [
    "llama-3.3-70b-versatile", 
    "mixtral-8x7b-32768",
    "gemma2-9b-it"
]


# Initialize the TavilySearchResults tool with a specified maximum number of results.
tool_tavily = TavilySearchResults(max_results=3)  # Allows retrieving up to 3 results

# Combine the TavilySearchResults and ExecPython tools into a list.
tools = [tool_tavily, ]

# FastAPI application setup with a title
app = FastAPI(title="AI Agent Chatbot")

# Define the request schema using Pydantic's BaseModel
class RequestState(BaseModel):
    model_name: str  # Name of the model to use for processing the request
    system_prompt: str  # System prompt for initializing the model
    messages: List[str]  # List of messages in the chat


# Define an endpoint for handling chat requests
@app.post("/chat")
def chat_endpoint(request: RequestState):
    """
    Endpoint for handling chat requests.

    Args:
        request (RequestState): The request data containing model name, system prompt, and messages.

    Returns:
        dict: A dictionary containing the response from the AI agent.
    """
    # Extract the model name from the request
    model_name = request.model_name

    # Check if the provided model name is supported
    if model_name not in MODEL_NAMES:
        return {"error": "Unsupported model name"}

    # Initialize the ChatGroq instance with the provided API key and model name
    llm = ChatGroq(
        api_key=groq_api_key, 
        model_name=model_name
    )
    
    # Create a ReAct agent with the specified system prompt, LLM, and tools
    agent = create_react_agent(
        prompt=request.system_prompt,
        model=llm,
        tools=tools,
    )
    
     # Create the initial state for processing
    state = {"messages": request.messages}
    
    # Process the state using the agent
    result = agent.invoke(state)  # Invoke the agent (can be async or sync based on implementation)
    
    return result


# Run the FastAPI application if executed as main module
if __name__ == "__main__":
    # Start the app on localhost with port 8000
    uvicorn.run(app, host='127.0.0.1', port=8000)  