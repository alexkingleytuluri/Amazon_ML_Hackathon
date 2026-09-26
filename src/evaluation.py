import pandas as pd


def parse_match_ids(value):
    """
    Convert comma-separated matched IDs into a set.
    Empty or missing values become an empty set.
    """

    if pd.isna(value):
        return set()

    value = str(value).strip()

    if value == "":
        return set()

    return {
        entity_id.strip()
        for entity_id in value.split(",")
        if entity_id.strip()
    }


def calculate_f05(precision, recall):
    """
    Calculate F0.5 score.
    """

    denominator = (0.25 * precision) + recall

    if denominator == 0:
        return 0.0

    return (1.25 * precision * recall) / denominator


def evaluate_entity(true_ids, predicted_ids):
    """
    Calculate Precision, Recall and F0.5
    for one Source 1 entity.
    """

    # Convert to sets so duplicates don't affect the score
    true_ids = set(true_ids)
    predicted_ids = set(predicted_ids)

    # Correct no-match
    if len(true_ids) == 0 and len(predicted_ids) == 0:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f05": 1.0
        }

    # False-positive singleton
    if len(true_ids) == 0 and len(predicted_ids) > 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f05": 0.0
        }

    # Number of correct matches
    true_positives = len(true_ids.intersection(predicted_ids))

    # Precision
    if len(predicted_ids) > 0:
        precision = true_positives / len(predicted_ids)
    else:
        precision = 0.0

    # Recall
    recall = true_positives / len(true_ids)

    # F0.5
    f05 = calculate_f05(precision, recall)

    return {
        "precision": precision,
        "recall": recall,
        "f05": f05
    }


def evaluate(ground_truth, predictions):
    """
    Calculate metrics for all Source 1 entities.

    ground_truth and predictions should be dictionaries:

    {
        "S1-001": {"S2-001", "S3-001"},
        "S1-002": {"S2-002"},
        "S1-003": set()
    }
    """

    results = []

    for source1_id, true_ids in ground_truth.items():

        predicted_ids = predictions.get(source1_id, set())

        metrics = evaluate_entity(
            true_ids,
            predicted_ids
        )

        results.append({
            "source1_entity_id": source1_id,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f05": metrics["f05"]
        })

    results_df = pd.DataFrame(results)

    if results_df.empty:
        return {
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f05": 0.0,
            "per_entity": results_df
        }

    return {
        "macro_precision": results_df["precision"].mean(),
        "macro_recall": results_df["recall"].mean(),
        "macro_f05": results_df["f05"].mean(),
        "per_entity": results_df
    }