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
                    "the user's request. You MUST respond strictly as a single valid "
                    "JSON object with two top-level keys:\n"
                    '- "natural_language": a natural language clinical summary.\n'
                    '- "fhir_bundle": a FHIR Bundle resource following AU Core.\n\n'
                    "Requirements:\n"
                    "- The FHIR bundle must adhere to the AU Core profile.\n"
                    "- The FHIR bundle must be a 'transaction' type bundle. Collection is not supported.\n"
                    "- The FHIR bundle must use correct resource URIs. For example, the reference in the subject field should use refer to the patient resource, not the urn:uuid:\n"
                    "- The FHIR bundle must contain all mandatory fields. For example: 'The type 'MedicationRequest' requires a property named 'intent'\n"
                    "- Generate only dummy identifiers.\n"
                    "- Include realistic pathology results and SNOMED condition codes.\n"
                    "- Ensure the natural language summary is fully consistent with the FHIR bundle.\n"
                    "- Do not include explanations, markdown, or any text outside the JSON.\n\n"
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
