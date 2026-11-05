def recall_at_k(retrieved_ids: list[str], correct_id: str, k: int) -> float:
    top = retrieved_ids[:k]
    if correct_id in top:
        return 1.0
    return 0.0


def reciprocal_rank(retrieved_ids: list[str], correct_id: str) -> float:
    for position, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id == correct_id:
            return 1 / position
    return 0.0