from langchain.chains import ConversationalRetrievalChain


def build_rag_chain(llm, vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    return chain
