from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from abac_rules import is_similar_rule_decision
import logging

def evaluate_rules(rules, decisions):
    if not rules or not decisions:
        logging.warning("Empty rules or decisions detected. Please check the input data.")
        return None

    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    predicted_decisions = []
    true_decisions = []

    for decision in decisions:
        matching_rules = [rule for rule in rules if is_similar_rule_decision(rule, decision)]
        if matching_rules:
            rule_decisions = [rule.decision for rule in matching_rules]
            if len(set(rule_decisions)) == 1:
                predicted_decision = rule_decisions[0]
            else:
                predicted_decision = max(set(rule_decisions), key=rule_decisions.count)
            predicted_decisions.append(predicted_decision)
            true_decisions.append(decision.decision)
            if predicted_decision == decision.decision:
                if predicted_decision == "permit":
                    true_positives += 1
                else:
                    true_negatives += 1
            else:
                if predicted_decision == "permit":
                    false_positives += 1
                else:
                    false_negatives += 1
        else:
            predicted_decisions.append("deny")
            true_decisions.append(decision.decision)
            false_negatives += 1

    if not predicted_decisions or not true_decisions:
        logging.warning("No predicted or true decisions found. Evaluation metrics cannot be calculated.")
        return 0.0, 0.0, 0.0, 0.0


    pos_label = "permit"

    precision = precision_score(true_decisions, predicted_decisions, pos_label=pos_label, zero_division=1)
    recall = recall_score(true_decisions, predicted_decisions, pos_label=pos_label, zero_division=1)
    f1 = f1_score(true_decisions, predicted_decisions, pos_label=pos_label, zero_division=1)
    accuracy = accuracy_score(true_decisions, predicted_decisions)

    return precision, recall, f1, accuracy