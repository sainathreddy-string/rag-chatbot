from langchain_core.embeddings import Embeddings

class DummyEmbeddings(Embeddings):
    """
    Dummy embeddings for Render deployment.
    Pinecone already contains vectors, so we do NOT re-embed.
    """

    def embed_documents(self, texts):
        return [[0.0] * 768 for _ in texts]

    def embed_query(self, text):
        return [0.0] * 768


def get_embeddings():
    return DummyEmbeddings()
