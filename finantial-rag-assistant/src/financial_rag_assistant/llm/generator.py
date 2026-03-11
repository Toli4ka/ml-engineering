from dataclasses import dataclass
from typing import List

from openai import OpenAI

from financial_rag_assistant.retrieval.vector_store import SearchResult


@dataclass
class GeneratedAnswer:
    answer: str
    sources: List[SearchResult]


def build_context(search_results: List[SearchResult]) -> str:
    if not search_results:
        raise ValueError("No search results provided.")

    context_parts = []

    for result in search_results:
        context_parts.append(
            f"[Source: page {result.page_number}, chunk {result.chunk_id}]\n"
            f"{result.text}"
        )

    return "\n\n".join(context_parts)


def generate_answer(
    question: str,
    search_results: List[SearchResult],
    client: OpenAI,
    model: str = "gpt-4.1-mini"
) -> GeneratedAnswer:
    if not question.strip():
        raise ValueError("Question must not be empty.")

    if not search_results:
        raise ValueError("No search results provided.")

    context = build_context(search_results)

    system_prompt = (
        "You are a helpful assistant for question answering over documents. "
        "Answer the user's question using only the provided context. "
        "If the answer is not contained in the context, say that the document "
        "does not provide enough information. "
        "Be accurate, concise, and do not invent facts."
    )

    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer based only on the context above."
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    answer_text = response.choices[0].message.content
    if answer_text is None:
        raise ValueError("Model returned an empty response.")

    return GeneratedAnswer(
        answer=answer_text.strip(),
        sources=search_results
    )
