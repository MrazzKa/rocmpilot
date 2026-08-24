from training.audit_dataset import audit_splits, check_parameter_consistency


def _row(instruction, output, row_id):
    return {"id": row_id, "instruction": instruction, "input": "", "output": output}


def test_audit_detects_within_and_cross_split_duplicates():
    duplicate = _row("Use cuda:0", "Keep cuda:0", "train-1")
    whitespace_variant = _row("Use   cuda:0", "Keep\n cuda:0", "test-1")
    splits = {
        "train": [duplicate, dict(duplicate, id="train-2")],
        "validation": [_row("Different", "Different output", "val-1")],
        "test": [whitespace_variant],
    }
    audit = audit_splits(splits, near_threshold=0.8)
    assert audit["splits"]["train"]["exact_duplicates"]["duplicate_instances"] == 1
    pair = audit["cross_split"]["train__test"]
    assert pair["exact"]["shared_groups"] == 0
    assert pair["normalized"]["shared_groups"] == 1


def test_metadata_consistency_check_reports_missing_output_value():
    record = {
        "instruction": "device=cuda:7",
        "output": "device=cuda:2",
        "parameters": {"device": "cuda:7"},
    }
    check = check_parameter_consistency(record)
    assert check["status"] == "inconsistent"
    assert check["details"][0]["present"] == {"instruction": True, "output": False}

