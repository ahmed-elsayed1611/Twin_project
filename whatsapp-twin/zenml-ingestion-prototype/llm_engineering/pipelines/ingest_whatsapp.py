from zenml import pipeline

from llm_engineering.steps.whatsapp.ingest_whatsapp_steps import create_instruction_dataset, parse_whatsapp_data


@pipeline
def ingest_whatsapp_pipeline(
    export_path: str,
    self_sender: str = "Ahmed",
    anonymize: bool = True,
    min_length: int = 5,
):
    """Pipeline to transform raw WhatsApp data into a fine-tuning dataset."""
    turns = parse_whatsapp_data(
        export_path=export_path,
        self_sender=self_sender,
        anonymize=anonymize,
        min_length=min_length,
    )

    dataset = create_instruction_dataset(
        turns_data=turns,
        self_sender=self_sender,
    )

    return dataset


if __name__ == "__main__":
    ingest_whatsapp_pipeline(
        export_path="data/whatsapp/whatsapp_chat.txt",
        self_sender="Ahmed",
    )
