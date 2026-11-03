from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict
from zenml.client import Client
from zenml.exceptions import EntityExistsError

# Printed at start-up. Never add keys, tokens or connection strings with passwords here.
LOGGED_SETTINGS = (
    "DATABASE_NAME",
    "USE_QDRANT_CLOUD",
    "QDRANT_DATABASE_HOST",
    "QDRANT_DATABASE_PORT",
    "QDRANT_TIMEOUT",
    "TEXT_EMBEDDING_MODEL_ID",
    "RERANKING_CROSS_ENCODER_MODEL_ID",
    "RAG_MODEL_DEVICE",
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # --- Required settings even when working locally. ---

    # OpenAI API
    OPENAI_MODEL_ID: str = "gpt-4o-mini"
    OPENAI_API_KEY: str | None = None
    # Any OpenAI-compatible provider, e.g. Cerebras: "https://api.cerebras.ai/v1" (then use its model id and key).
    OPENAI_BASE_URL: str | None = None

    # Huggingface API
    HUGGINGFACE_ACCESS_TOKEN: str | None = None

    # Comet ML (during training)
    COMET_API_KEY: str | None = None
    COMET_PROJECT: str = "twin"

    # --- Required settings when deploying the code. ---
    # --- Otherwise, default values values work fine. ---

    # MongoDB database
    DATABASE_HOST: str = "mongodb://llm_engineering:llm_engineering@127.0.0.1:27017"
    DATABASE_NAME: str = "twin"

    # Qdrant vector database
    USE_QDRANT_CLOUD: bool = False
    QDRANT_DATABASE_HOST: str = "localhost"
    QDRANT_DATABASE_PORT: int = 6333
    QDRANT_CLOUD_URL: str = "str"
    QDRANT_APIKEY: str | None = None
    # Seconds per request. Without it the client waits only 5 seconds; creating a collection took 75 seconds
    # on a busy disk (Docker Desktop on Windows).
    QDRANT_TIMEOUT: int = 120

    # AWS Authentication
    AWS_REGION: str = "eu-central-1"
    AWS_ACCESS_KEY: str | None = None
    AWS_SECRET_KEY: str | None = None
    AWS_ARN_ROLE: str | None = None

    # --- Optional settings used to tweak the code. ---

    # AWS SageMaker
    HF_MODEL_ID: str = "mlabonne/TwinLlama-3.1-8B-DPO"
    GPU_INSTANCE_TYPE: str = "ml.g5.2xlarge"
    SM_NUM_GPUS: int = 1
    MAX_INPUT_LENGTH: int = 2048
    MAX_TOTAL_TOKENS: int = 4096
    MAX_BATCH_TOTAL_TOKENS: int = 4096
    COPIES: int = 1  # Number of replicas
    GPUS: int = 1  # Number of GPUs
    CPUS: int = 2  # Number of CPU cores
    # "gpu": the book's TGI container on GPU_INSTANCE_TYPE. "cpu": Hugging Face's CPU inference container on
    # CPU_INSTANCE_TYPE, for an account without GPU quota (mine was refused one in Oct 2026). The image tag is
    # passed explicitly because SDK 2.232 only knows CPU images up to transformers 4.37.
    DEPLOYMENT_TARGET: str = "gpu"
    CPU_INSTANCE_TYPE: str = "ml.m5.2xlarge"
    CPU_INFERENCE_IMAGE_TAG: str = "2.6.0-transformers4.51.3-cpu-py312-ubuntu22.04"

    SAGEMAKER_ENDPOINT_CONFIG_INFERENCE: str = "twin"
    SAGEMAKER_ENDPOINT_INFERENCE: str = "twin"
    # "sagemaker": the deployed fine-tuned twin. "openai_compatible": OPENAI_MODEL_ID via OPENAI_BASE_URL (OpenAI,
    # Cerebras, ...), to run the RAG pipeline before the twin is deployed.
    LLM_INFERENCE_BACKEND: str = "sagemaker"
    TEMPERATURE_INFERENCE: float = 0.01
    TOP_P_INFERENCE: float = 0.9
    # 150 tokens (about 110 words) cut LinkedIn posts off mid-sentence. 600 still fits the endpoint:
    # up to MAX_INPUT_LENGTH (2048) prompt tokens + 600 stays under MAX_TOTAL_TOKENS (4096).
    MAX_NEW_TOKENS_INFERENCE: int = 600

    # My WhatsApp twin (POST /reply): the system message it was fine-tuned with (SFT notebook). Generation matches the
    # evaluation notebook (greedy, repetition_penalty 1.1, no_repeat_ngram_size 4) except the length: on the CPU
    # endpoint (ml.m5.2xlarge, 32-bit, ~1 token/s) 100 new tokens ran past the real-time endpoint's 60-second
    # limit (29 Oct: "احكيلي عن نفسك" timed out). 40 is still a long WhatsApp reply.
    TWIN_SYSTEM_MESSAGE: str = (
        "Continue the conversation as Ahmed. Match the language, tone, and context of the "
        "conversation. Do not claim to know facts that are not supported by the dialogue."
    )
    TWIN_MAX_NEW_TOKENS: int = 40

    # RAG
    TEXT_EMBEDDING_MODEL_ID: str = "sentence-transformers/all-MiniLM-L6-v2"
    RERANKING_CROSS_ENCODER_MODEL_ID: str = "cross-encoder/ms-marco-MiniLM-L-4-v2"
    RAG_MODEL_DEVICE: str = "cpu"

    # LinkedIn Credentials
    LINKEDIN_USERNAME: str | None = None
    LINKEDIN_PASSWORD: str | None = None

    @property
    def OPENAI_MAX_TOKEN_WINDOW(self) -> int:
        official_max_token_window = {
            "gpt-3.5-turbo": 16385,
            "gpt-4-turbo": 128000,
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000,
        }.get(self.OPENAI_MODEL_ID, 128000)

        max_token_window = int(official_max_token_window * 0.90)

        return max_token_window

    @classmethod
    def load_settings(cls) -> "Settings":
        """
        Tries to load the settings from the ZenML secret store. If the secret does not exist, it initializes the settings from the .env file and default values.

        Returns:
            Settings: The initialized settings object.
        """

        try:
            logger.info("Loading settings from the ZenML secret store.")

            settings_secrets = Client().get_secret("settings")
            settings = Settings(**settings_secrets.secret_values)
        except (RuntimeError, KeyError):
            logger.warning(
                "Failed to load settings from the ZenML secret store. Defaulting to loading the settings from the '.env' file."
            )
            settings = Settings()

        # A value in .env silently beats the default in this class, so show what is actually in use.
        # Only settings without secrets are printed.
        active = {name: getattr(settings, name) for name in LOGGED_SETTINGS}
        logger.info(f"Active settings: {active}")

        return settings

    def export(self) -> None:
        """
        Exports the settings to the ZenML secret store.
        """

        env_vars = settings.model_dump()
        for key, value in env_vars.items():
            env_vars[key] = str(value)

        client = Client()

        try:
            client.create_secret(name="settings", values=env_vars)
        except EntityExistsError:
            logger.warning(
                "Secret 'scope' already exists. Delete it manually by running 'zenml secret delete settings', before trying to recreate it."
            )


settings = Settings.load_settings()
