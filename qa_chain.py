from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from config import GEMINI_API_KEY, CHAT_MODEL, TEMPERATURE, MAX_TOKENS


PROMPT_TEMPLATE = """You are a helpful assistant.
Use ONLY the context below to answer the question.
If the answer is not in the context, say
"I couldn't find that in the document."

Context:
{context}

Question: {question}

Answer:"""

def build_qa_chain(vector_store):
    llm = ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=TEMPERATURE,
        max_output_tokens=MAX_TOKENS
    )

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs,
         "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def get_answer(chain_and_retriever, question: str) -> dict:
    chain, retriever = chain_and_retriever
    answer = chain.invoke(question)
    sources = retriever.invoke(question)
    return {
        "answer": answer,
        "sources": sources
    }