import os
import pinecone
from langchain_community.vectorstores import Pinecone as LangchainPinecone


def get_vectorstore(docs, embeddings):
    pinecone.init(
        api_key=os.environ.get("PINECONE_API_KEY"),
        environment=os.environ.get("PINECONE_ENV")
    )

    index_name = os.environ.get("PINECONE_INDEX")

    index = pinecone.Index(index_name)

    vectorstore = LangchainPinecone.from_documents(
        documents=docs,
        embedding=embeddings,
        index_name=index_name
    )

    return vectorstore
