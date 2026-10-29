import time

from loguru import logger

from llm_engineering.model.inference.inference import LLMInferenceSagemakerEndpoint
from llm_engineering.model.inference.run import TwinReplyExecutor
from llm_engineering.settings import settings

# The SFT notebook's own test messages, so replies can be compared with what the twin wrote on Colab.
TEST_MESSAGES = ["عامل ايه النهاردة؟", "معلش هو انت فاضي دلوقتي؟", "احكيلي عن نفسك"]

if __name__ == "__main__":
    llm = LLMInferenceSagemakerEndpoint(
        endpoint_name=settings.SAGEMAKER_ENDPOINT_INFERENCE, inference_component_name=None
    )
    for message in TEST_MESSAGES:
        start = time.perf_counter()
        reply = TwinReplyExecutor(llm, message).execute()
        logger.info(f"{time.perf_counter() - start:.1f}s | Contact: {message} | twin: {reply}")
