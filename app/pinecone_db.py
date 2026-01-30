import os
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

def get_vectorstore(docs, embeddings):
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX")

    if not api_key or not index_name:
        raise ValueError("PINECONE_API_KEY or PINECONE_INDEX not set")

    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)

    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embeddings
    )

    # Add docs only once (safe)
    if docs:
        vectorstore.add_documents(docs)

    return vectorstore
