import os
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore


def get_vectorstore(docs, embeddings=None):
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX")

    if not api_key or not index_name:
        raise ValueError("PINECONE_API_KEY or PINECONE_INDEX not set")

    # Initialize Pinecone client
    pc = Pinecone(api_key=api_key)

    # Connect to existing index
    index = pc.Index(index_name)

    # 🔥 IMPORTANT: DO NOT PASS embeddings here
    vectorstore = PineconeVectorStore(
        index=index,
        namespace="default"
    )

    # Add documents only ONCE (avoid duplicates in prod)
    if docs:
        vectorstore.add_documents(docs)

    return vectorstore
