from loguru import logger

try:
    from sagemaker.enums import EndpointType
    from sagemaker.huggingface import get_huggingface_llm_image_uri
except ModuleNotFoundError:
    logger.warning("Couldn't load SageMaker imports. Run 'poetry install --with aws' to support AWS.")

from llm_engineering.model.utils import ResourceManager
from llm_engineering.settings import settings

from .config import hugging_face_cpu_deploy_config, hugging_face_deploy_config, model_resource_config
from .sagemaker_huggingface import DeploymentService, SagemakerHuggingfaceStrategy, sagemaker_session

# AWS publishes its Deep Learning Containers (the TGI and CPU inference images) from this account in every region.
DLC_ACCOUNT_ID = "763104351884"


def create_endpoint(endpoint_type=EndpointType.INFERENCE_COMPONENT_BASED) -> None:
    assert settings.AWS_ARN_ROLE is not None, "AWS_ARN_ROLE is not set in the .env file."
    assert settings.DEPLOYMENT_TARGET in ("gpu", "cpu"), "DEPLOYMENT_TARGET must be 'gpu' or 'cpu'."

    logger.info(
        f"Creating a {settings.DEPLOYMENT_TARGET} endpoint with endpoint_type = {endpoint_type} "
        f"and model_id = {settings.HF_MODEL_ID}"
    )

    if settings.DEPLOYMENT_TARGET == "cpu":
        llm_image = (
            f"{DLC_ACCOUNT_ID}.dkr.ecr.{settings.AWS_REGION}.amazonaws.com/"
            f"huggingface-pytorch-inference:{settings.CPU_INFERENCE_IMAGE_TAG}"
        )
        config, instance_type, resources = hugging_face_cpu_deploy_config, settings.CPU_INSTANCE_TYPE, None
    else:
        llm_image = get_huggingface_llm_image_uri("huggingface", version="2.2.0", session=sagemaker_session())
        config, instance_type, resources = hugging_face_deploy_config, settings.GPU_INSTANCE_TYPE, model_resource_config

    resource_manager = ResourceManager()
    deployment_service = DeploymentService(resource_manager=resource_manager)

    SagemakerHuggingfaceStrategy(deployment_service).deploy(
        role_arn=settings.AWS_ARN_ROLE,
        llm_image=llm_image,
        config=config,
        endpoint_name=settings.SAGEMAKER_ENDPOINT_INFERENCE,
        endpoint_config_name=settings.SAGEMAKER_ENDPOINT_CONFIG_INFERENCE,
        gpu_instance_type=instance_type,
        resources=resources,
        endpoint_type=endpoint_type,
    )


if __name__ == "__main__":
    create_endpoint(endpoint_type=EndpointType.MODEL_BASED)
