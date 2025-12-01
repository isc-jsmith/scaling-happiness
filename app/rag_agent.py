from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI


def build_rag_chain(retriever, openai_api_key: str):
    llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=openai_api_key)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are an AI assistant specialized in generating synthetic "
                    "clinical data. Use the provided context to generate "
                    "comprehensive and accurate synthetic clinical data based on "
                    "the user's request. The data can be in natural language or "
                    "FHIR format, as specified by the user. If the user asks for "
                    "FHIR format, ensure the output strictly adheres to the "
                    "FHIR schema relevant to the request. If the context is "
                    "insufficient, state that you cannot fulfill the request.\n"
                    "Retrieved context: {context}"
                ),
            ),
            ("human", "{question}"),
        ]
    )

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

