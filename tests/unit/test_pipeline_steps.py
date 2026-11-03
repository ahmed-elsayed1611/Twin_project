from types import SimpleNamespace

import pytest

from llm_engineering.application.networks.base import SingletonMeta
from llm_engineering.application.utils import split_user_full_name
from llm_engineering.domain.exceptions import ImproperlyConfigured
from steps.feature_engineering.load_to_vector_db import load_to_vector_db
from steps.feature_engineering.rag import _add_embeddings_metadata


@pytest.mark.parametrize("name", ["", "   ", None])
def test_empty_user_name_is_refused(name) -> None:  # 3.3
    with pytest.raises(ImproperlyConfigured):
        split_user_full_name(name)


def test_extra_spaces_in_user_name_are_ignored() -> None:  # 3.3
    assert split_user_full_name("Maxime  Labonne") == ("Maxime", "Labonne")


def test_step_metadata_does_not_change_the_chunks() -> None:  # 4.10
    chunk = SimpleNamespace(
        metadata={"embedding_size": 384},
        author_full_name="Test Author",
        get_category=lambda: "articles",
    )

    metadata = _add_embeddings_metadata([chunk], {})

    assert metadata["articles"]["authors"] == ["Test Author"]
    assert chunk.metadata == {"embedding_size": 384}


def test_failed_insert_fails_the_step() -> None:  # 4.9
    class _FailingDocument:
        @classmethod
        def bulk_insert(cls, documents) -> bool:
            return False

        @classmethod
        def get_collection_name(cls) -> str:
            return "fake_collection"

    with pytest.raises(RuntimeError, match="fake_collection"):
        load_to_vector_db.entrypoint([_FailingDocument()])


def test_singleton_refuses_different_arguments() -> None:  # 4.15
    class _Model(metaclass=SingletonMeta):
        def __init__(self, model_id: str = "default") -> None:
            self.model_id = model_id

    first = _Model(model_id="a")

    assert _Model() is first
    assert _Model(model_id="a") is first
    with pytest.raises(ValueError, match="already exists"):
        _Model(model_id="b")
