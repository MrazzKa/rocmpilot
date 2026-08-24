from training.generate_dataset import TEMPLATES, generate_examples, split_examples


def test_sampled_parameter_is_reused_in_instruction_and_output():
    examples = generate_examples(seed=7, examples_per_template=3)
    assert len(examples) == len(TEMPLATES) * 3
    for example in examples:
        [(name, value)] = example["parameters"].items()
        assert value in example["instruction"], (name, example["id"])
        assert value in example["output"], (name, example["id"])


def test_generation_is_deterministic_for_fixed_seed():
    first = generate_examples(seed=123, examples_per_template=12)
    second = generate_examples(seed=123, examples_per_template=12)
    different = generate_examples(seed=124, examples_per_template=12)
    assert first == second
    assert first != different


def test_parameter_holdout_keeps_values_in_one_split():
    examples = generate_examples(seed=11, examples_per_template=50)
    splits = split_examples(examples, seed=11, strategy="parameter_holdout")
    ownership = {}
    for split, rows in splits.items():
        assert rows
        for row in rows:
            value = next(iter(row["parameters"].values()))
            key = (row["template_id"], value)
            assert key not in ownership or ownership[key] == split
            ownership[key] = split

