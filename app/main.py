import os
from flask import Flask, request, jsonify, render_template
from langchain_core.documents import Document

from app.embeddings import get_embeddings
from app.pinecone_db import get_vectorstore
from app.rag_chain import build_rag_chain
from app.groq_langchain_llm import GroqLangChainLLM


app = Flask(
    __name__,
    template_folder="../templates",
    static_folder="../static"
)

# ---------------- RAG INITIALIZATION (RUNS ONCE) ----------------

docs = [
    Document(page_content="RAG uses retrieval to improve LLM answers."),
    Document(page_content="Pinecone is a vector database used in RAG systems."),
]

embeddings = get_embeddings()
vectorstore = get_vectorstore(docs, embeddings)

llm = GroqLangChainLLM()
rag_chain = build_rag_chain(llm, vectorstore)

# ----------------------------------------------------------------


# 🔹 UI Route
@app.route("/")
def home():
    return render_template("chat.html")


# 🔹 Chat API
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"response": "Please enter a message."})

    try:
        result = rag_chain.invoke({
            "question": user_message,
            "chat_history": []   # REQUIRED
        })

        if isinstance(result, dict):
            answer = result.get("answer", "")
        else:
            answer = str(result)

        # 🔹 Hybrid fallback (LLM knowledge)
        if not answer or len(answer.strip()) < 15:
            fallback = llm.invoke(user_message)
            return jsonify({"response": fallback})

        return jsonify({"response": answer})

    except Exception as e:
        print("🔥 ERROR:", e)
        fallback = llm.invoke(user_message)
        return jsonify({"response": fallback})


# 🔹 REQUIRED FOR CLOUD DEPLOYMENT
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"\n🚀 Server running on port {port}\n")
    app.run(host="0.0.0.0", port=port)
