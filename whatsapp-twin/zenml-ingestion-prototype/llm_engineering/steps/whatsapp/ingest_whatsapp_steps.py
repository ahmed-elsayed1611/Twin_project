from typing import List

from zenml import step

from llm_engineering.application.dataset.whatsapp_extractor import (
    ConversationTurn,
    InstructDatasetSample,
    WhatsAppExtractor,
)


@step
def parse_whatsapp_data(
    export_path: str,
    self_sender: str = "Ahmed",
    anonymize: bool = True,
    min_length: int = 5,
) -> List[dict]:
    """Parses a raw WhatsApp export and returns cleaned turns."""
    extractor = WhatsAppExtractor(self_sender=self_sender, anonymize=anonymize, min_length=min_length)
    # JSON-mode dicts (ISO timestamps) so ZenML can serialize the step output.
    return [turn.model_dump(mode="json") for turn in extractor.iter_turns(export_path)]


@step
def create_instruction_dataset(
    turns_data: List[dict],
    self_sender: str = "Ahmed",
) -> List[InstructDatasetSample]:
    """Transforms turns into instruction-response pairs."""
    turns = [ConversationTurn(**t) for t in turns_data]

    extractor = WhatsAppExtractor(self_sender=self_sender)
    samples = extractor.extract_samples_from_turns(turns)

    print(f"Extracted {len(samples)} valid instruction pairs from {len(turns)} turns.")
    return samples
