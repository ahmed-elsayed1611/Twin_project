from functools import cache

import opik
from fastapi import FastAPI, HTTPException
from loguru import logger
from opik import opik_context
from pydantic import BaseModel

from llm_engineering import settings
from llm_engineering.application.rag.retriever import ContextRetriever
from llm_engineering.application.utils import misc
from llm_engineering.domain.embedded_chunks import EmbeddedChunk
from llm_engineering.infrastructure.opik_utils import configure_opik
from llm_engineering.model.inference import InferenceExecutor, LLMInferenceOpenAICompatible, LLMInferenceSagemakerEndpoint

configure_opik()

app = FastAPI()

# Returned instead of calling the model when retrieval finds nothing: without context the model would answer
# from memory, which is the problem RAG is there to solve.
NO_CONTEXT_ANSWER = "I couldn't find anything about this in the knowledge base, so I won't guess."


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str


@cache
def get_retriever() -> ContextRetriever:
    """One retriever for the whole app instead of a new one for every request."""

    return ContextRetriever(mock=False)


@opik.track
def call_llm_service(query: str, context: str | None) -> str:
    if settings.LLM_INFERENCE_BACKEND == "openai_compatible":
        llm = LLMInferenceOpenAICompatible()
    else:
        llm = LLMInferenceSagemakerEndpoint(
            endpoint_name=settings.SAGEMAKER_ENDPOINT_INFERENCE, inference_component_name=None
        )
    answer = InferenceExecutor(llm, query, context).execute()

    return answer


@opik.track
def rag(query: str) -> str:
    documents = get_retriever().search(query, k=3)
    if not documents:
        logger.warning("No context was retrieved for the query. Answering without calling the model.")
        opik_context.update_current_trace(tags=["rag", "no_context"])

        return NO_CONTEXT_ANSWER

    context = EmbeddedChunk.to_context(documents)

    answer = call_llm_service(query, context)

    opik_context.update_current_trace(
        tags=["rag"],
        metadata={
            "model_id": settings.OPENAI_MODEL_ID
            if settings.LLM_INFERENCE_BACKEND == "openai_compatible"
            else settings.HF_MODEL_ID,
            "embedding_model_id": settings.TEXT_EMBEDDING_MODEL_ID,
            "temperature": settings.TEMPERATURE_INFERENCE,
            "query_tokens": misc.compute_num_tokens(query),
            "context_tokens": misc.compute_num_tokens(context),
            "answer_tokens": misc.compute_num_tokens(answer),
        },
    )

    return answer


# A plain `def`, not `async def`: rag() blocks for seconds (OpenAI, Qdrant, SageMaker). FastAPI runs plain
# endpoints in a thread pool, so one slow request no longer holds up every other request.
@app.post("/rag", response_model=QueryResponse)
def rag_endpoint(request: QueryRequest):
    try:
        answer = rag(query=request.query)

        return {"answer": answer}
    except Exception as e:
        # The details stay in the server log; the caller must not see hosts, URLs or credentials.
        logger.exception("The RAG request failed.")

        raise HTTPException(status_code=500, detail="Internal error while answering the query.") from e
