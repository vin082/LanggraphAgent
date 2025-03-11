import streamlit as st
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, TypedDict
import os
from dotenv import load_dotenv
import json
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
import time
import asyncio

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Streamlit UI setup
st.set_page_config(page_title="🤖 LangGraph Chatbot", page_icon="🤖", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .stChatInput {
        position: fixed;
        bottom: 0;
        width: 100%;
        background-color: #f0f2f6;
        padding: 10px;
        z-index: 1000;
    }
    .stChatMessage {
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .user-message {
        background-color: #0078d4;
        color: white;
        margin-left: auto;
        margin-right: 0;
        max-width: 70%;
    }
    .assistant-message {
        background-color: #f0f2f6;
        color: black;
        margin-left: 0;
        margin-right: auto;
        max-width: 70%;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("🤖 LangGraph Chatbot")
st.write("Welcome to the LangGraph-powered chatbot! Type your message below and press Enter to chat.")

# Load or create FAISS vector database
@st.cache_resource
def load_retriever():
    vector_db_path = "faiss_index"
    
    if os.path.exists(vector_db_path):
        vectorstore = FAISS.load_local(vector_db_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    else:
        urls = [
            "https://lilianweng.github.io/posts/2023-06-23-agent/",
            "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
            "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
        ]
        
        docs = [WebBaseLoader(url).load() for url in urls]
        docs_list = [item for sublist in docs for item in sublist]
        
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=100, chunk_overlap=50
        )
        doc_splits = text_splitter.split_documents(docs_list)
        vectorstore = FAISS.from_documents(
            documents=doc_splits, embedding=OpenAIEmbeddings()
        )
        
        vectorstore.save_local(vector_db_path)
    
    retriever = vectorstore.as_retriever()
    return retriever

# Initialize graph only once (cached)
@st.cache_resource
def create_graph():
    retriever = load_retriever()
    
    retriever_tool = create_retriever_tool(
        retriever,
        "lilian_weng_research",
        "Search for information about Lilian Weng's research and blog posts."
    )
    
    class State(TypedDict):
        messages: Annotated[list, add_messages]
    
    graph_builder = StateGraph(State)
    
    search_tool = TavilySearchResults(max_results=1)
    tools = [search_tool, retriever_tool]
    llm = ChatOpenAI(model="gpt-4", temperature=0.0)
    llm_with_tools = llm.bind_tools(tools)
    
    def chatbot(state: State):
        result = llm_with_tools.invoke(state["messages"])
        return {"messages": [result]}
    
    graph_builder.add_node("chatbot", chatbot)
    tool_node = ToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)
    
    # Define tools condition safely
    def safe_tools_condition(state: State):
        tool_decision = tools_condition(state)
        return tool_decision if tool_decision else "chatbot"

    graph_builder.add_conditional_edges("chatbot", safe_tools_condition)
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.set_entry_point("chatbot")
    graph = graph_builder.compile()
    
    return graph, llm

if "messages" not in st.session_state:
    st.session_state.messages = []
    
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    with st.chat_message(role):
        st.write(content)

user_input = st.chat_input("Type your message here...")

graph, llm = create_graph()

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.write(user_input)
    
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        try:
            start_time = time.time()
            for event in graph.stream({"messages": [HumanMessage(content=user_input)]}):
                for value in event.values():
                    if value["messages"]:
                        assistant_message = value["messages"][-1].content
                        full_response += assistant_message + " "
                        response_placeholder.markdown(full_response)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response.strip()})
            st.caption(f"Response processed in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.session_state.messages.append({"role": "assistant", "content": f"I'm sorry, I encountered an error: {str(e)}"})
