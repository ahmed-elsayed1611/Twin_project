from llm_engineering.application.evaluation.retrieval_metrics import recall_at_k, reciprocal_rank

RESULTS = ["D7", "D3", "D1"]


def test_recall_found_inside_k():
    assert recall_at_k(RESULTS, "D3", k=2) == 1.0


def test_recall_outside_k():
    assert recall_at_k(RESULTS, "D3", k=1) == 0.0


def test_recall_not_found():
    assert recall_at_k(RESULTS, "D9", k=3) == 0.0


def test_rr_first_position():
    assert reciprocal_rank(RESULTS, "D7") == 1.0


def test_rr_second_position():
    assert reciprocal_rank(RESULTS, 'D3') == 0.5


def test_rr_not_found():
    assert reciprocal_rank(RESULTS, "D9") == 0.0


def test_rr_duplicates_uses_first():
    assert reciprocal_rank(["D3", "D3", "D1"], "D3") == 1.0