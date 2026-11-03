import json

from loguru import logger

try:
    from sagemaker.compute_resource_requirements.resource_requirements import ResourceRequirements
except ModuleNotFoundError:
    logger.warning("Couldn't load SageMaker imports. Run 'poetry install --with aws' to support AWS.")

from llm_engineering.settings import settings

hugging_face_deploy_config = {
    "HF_MODEL_ID": settings.HF_MODEL_ID,
    "HUGGING_FACE_HUB_TOKEN": settings.HUGGINGFACE_ACCESS_TOKEN,
    "SM_NUM_GPUS": json.dumps(settings.SM_NUM_GPUS),  # Number of GPU used per replica
    "MAX_INPUT_LENGTH": json.dumps(settings.MAX_INPUT_LENGTH),  # Max length of input text
    "MAX_TOTAL_TOKENS": json.dumps(settings.MAX_TOTAL_TOKENS),  # Max length of the generation (including input text)
    "MAX_BATCH_TOTAL_TOKENS": json.dumps(settings.MAX_BATCH_TOTAL_TOKENS),
    "MAX_BATCH_PREFILL_TOKENS": json.dumps(settings.MAX_BATCH_TOTAL_TOKENS),
    "HF_MODEL_QUANTIZE": "bitsandbytes",
}

# Hugging Face's CPU inference container (DEPLOYMENT_TARGET="cpu") is not TGI: it loads a plain transformers
# pipeline for HF_TASK and has none of TGI's token limits or batching settings. The token lets it download a
# private repo; the toolkit has read it as HF_API_TOKEN, newer versions as HF_TOKEN.
hugging_face_cpu_deploy_config = {
    "HF_MODEL_ID": settings.HF_MODEL_ID,
    "HF_TASK": "text-generation",
    "HF_API_TOKEN": settings.HUGGINGFACE_ACCESS_TOKEN,
    "HF_TOKEN": settings.HUGGINGFACE_ACCESS_TOKEN,
    # The model server starts one worker per vCPU by default, each with its own copy of the model. 8 copies of a
    # 3B model in 32-bit (~12 GB each) on a 32 GB ml.m5.2xlarge died with "Cannot allocate memory" (29 Oct).
    "SAGEMAKER_MODEL_SERVER_WORKERS": "1",
}


model_resource_config = ResourceRequirements(
    requests={
        "copies": settings.COPIES,  # Number of replicas.
        "num_accelerators": settings.GPUS,  # Number of GPUs required.
        "num_cpus": settings.CPUS,  # Number of CPU cores required.
        "memory": 5 * 1024,  # Minimum memory required in Mb (required)
    },
)
