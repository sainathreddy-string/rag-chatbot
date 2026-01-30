import os
from flask import Flask, request, jsonify, render_template
from langchain_core.documents import Document
from dotenv import load_dotenv

from app.embeddings import get_embeddings
from app.pinecone_db import get_vectorstore
from app.rag_chain import build_rag_chain
from app.groq_langchain_llm import GroqLangChainLLM

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

app = Flask(
    __name__,
    template_folder="../templates",
    static_folder="../static"
)

# -------- RAG INIT --------
docs = [
    Document(page_content="RAG uses retrieval to improve LLM answers."),
    Document(page_content="Pinecone is a vector database used in RAG systems.")
]

embeddings = get_embeddings()
vectorstore = get_vectorstore(docs, embeddings)

llm = GroqLangChainLLM()
rag_chain = build_rag_chain(llm, vectorstore)
# -------------------------

@app.route("/")
def home():
    return render_template("chat.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    question = data.get("message", "").strip()

    if not question:
        return jsonify({"response": "Please enter a message."})

    try:
        result = rag_chain.invoke({
            "question": question,
            "chat_history": []
        })

        answer = result.get("answer") if isinstance(result, dict) else str(result)

        if not answer or len(answer.strip()) < 15:
            return jsonify({"response": llm.invoke(question)})

        return jsonify({"response": answer})

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"response": llm.invoke(question)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
