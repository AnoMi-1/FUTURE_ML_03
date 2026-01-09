import os
import uuid
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
 


# Load environment variables
load_dotenv()

# API keys with proper validation
google_api_key = os.getenv('GOOGLE_API_KEY')
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file.")

openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file.")

if not os.getenv("TAVILY_API_KEY"):
    raise ValueError("TAVILY_API_KEY not found in .env file.")

# Initialize search tool 
search_tool = TavilySearch(
    max_results=5,
    topic="general",
    include_domains=['kplc.co.ke'],
)

# Initialize model and embeddings
# model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

model = init_chat_model("gpt-4o-mini", model_provider="openai")


embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004", 
    google_api_key=google_api_key
)

# Create persistent vector store
vector_store = Chroma(
    collection_name="kplc_collection",
    embedding_function=embeddings,
    persist_directory="./kplc_chroma_db",
)

# Load and process PDF
file_path = "./kplc_document.pdf"  # Replace with your actual PDF path
try:
    loader = PyPDFLoader(file_path, mode="page")
    docs = loader.load()
    
    # Chunk documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200, 
        add_start_index=True,
    )
    all_splits = text_splitter.split_documents(docs)
    print(f"Split PDF into {len(all_splits)} chunks")
    
    # Store embeddings
    document_ids = vector_store.add_documents(documents=all_splits)
    print(f"Stored {len(document_ids)} document embeddings")
    
except FileNotFoundError:
    print("ERROR: kplc_document.pdf not found. Please add the file.")
    exit(1)

# RAG retrieval tool
@tool  
def retrieve_context(query: str):
    """Retrieve relevant KPLC information from uploaded documents."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    context = "\n\n".join(
        f"Source: {doc.metadata}\nContent: {doc.page_content}"
        for doc in retrieved_docs
    )
    return context  # ✅ Simple string works perfectly


# Agent prompt
prompt = '''
You are a friendly KPLC customer support assistant. 😊 

INSTRUCTIONS:
1. FIRST check retrieved context from KPLC documents
2. If context insufficient, use search tool for kplc.co.ke info
3. NEVER print raw metadata, source info, or document chunks
4. Give concise, helpful answers in your own words  
5. End with "Visit kplc.co.ke for latest info!" when appropriate

CONTEXT: {context}


EXAMPLES:
[User]: I'd like to install electricity at my place
[You]: To apply for a new electricity connection, you'll need your ID and KRA PIN. The quotation is valid for 90 days. Visit kplc.co.ke for full process! 😊

[User]: How long for meter installation?
[You]: Meter installation timeline varies. Check your quotation status at kplc.co.ke! 😊

Say "I don't know" if unclear.
'''

# Initialize agent with memory and summarization
checkpointer = InMemorySaver()
agent = create_agent(
    model=model,
    tools=[retrieve_context, search_tool],
    system_prompt=prompt,
    checkpointer=checkpointer,
    middleware=[
        SummarizationMiddleware(
            model=model,
            trigger=("tokens", 4000),
            keep=("messages", 20)
        )
    ],
)

# Dynamic chat loop
def run_kplc_chat():
    """Interactive chat with persistent memory."""
    session_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": f"kplc_session_{session_id}"}}
    
    print("🔌 KPLC Assistant Online! Ask about tariffs, meters, or services.")
    print("Type 'exit', 'quit', or 'bye' to end.\n")
    
    while True:
        try:
            user_query = input("\n[You] ➤ ").strip()
            
            if user_query.lower() in ['exit', 'quit', 'bye']:
                print("👋 Thank you for using KPLC Assistant!")
                break
            
            if not user_query:
                print("Please enter a question about KPLC services.\n")
                continue
            
            # response = agent.invoke({"messages": [("human", user_query)]}, config=config)
          
            response = agent.invoke({"messages": [{"role": "user", "content": user_query}]}, config=config)
            print(f"[KPLC] ➤ {response['messages'][-1].content}")
            
            


            
        except KeyboardInterrupt:
            print("\n👋 Chat ended.")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")


# Run the application
if __name__ == "__main__":
    run_kplc_chat()
