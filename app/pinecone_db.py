import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_community.vectorstores import Pinecone as LangchainPinecone

load_dotenv()

def get_vectorstore(documents, embeddings):
    # Create Pinecone client
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    # Get existing index (must already exist in dashboard)
    index = pc.Index("rag-chatbot")

    # Let LangChain wrap the index
    vectorstore = LangchainPinecone(
        index=index,
        embedding=embeddings,
        text_key="text"
    )

    # Add documents
    vectorstore.add_documents(documents)

    return vectorstore
