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
                    "the user's request. The data should both natural language and "
                    "FHIR format. The FHIR output should adhere to the AU Core profile "
                    "and valid dummy identifers should be generated. All output should include " 
                    "realistic pathology results and SNOMED condition codes. The content of the natural "
                    "language output must match the FHIR output. If the context is "
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

