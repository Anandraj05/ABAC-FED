import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from data_preprocessing import preprocess_data, validate_data
from feature_engineering import engineer_features
from policy_transfer import policy_transfer_local_log, policy_transfer_local_learning, policy_transfer_hybrid_learning, process_rule
from evaluation import evaluate_rules
from visualization import visualize_results
from abac_rules import ABACRule, AccessControlDecision
import logging
import configparser
from multiprocessing import Pool
import random

logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def random_forest_policy_learner(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    return rf

def main():
    try:
        config = configparser.ConfigParser()
        config.read('config.ini')

        data_path = config.get('Data', 'dataset_path')

        data = pd.read_csv(data_path, delimiter=',')
        print("Dataset loaded successfully:")
        print(data.head())

        data = validate_data(data)
        data = preprocess_data(data)

        # Handle datetime columns
        datetime_cols = ['Timestamp']
        for col in datetime_cols:
            if col in data.columns:
                data[col] = pd.to_datetime(data[col], errors='coerce')
                data[col + '_Year'] = data[col].dt.year
                data[col + '_Month'] = data[col].dt.month
                data[col + '_Day'] = data[col].dt.day


        data = data.drop(datetime_cols, axis=1)

        string_cols = ['Resource ID']
        for col in string_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col].str.replace('R', ''), errors='coerce')

        data = engineer_features(data)

        numerical_cols = ['Behavioral Score']
        if all(col in data.columns for col in numerical_cols):
            scaler = StandardScaler()
            data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

        # Perform stratified k-fold cross-validation
        k = 10
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

        # Initialize lists to store evaluation metrics for each fold
        precision_log_scores = []
        recall_log_scores = []
        f1_log_scores = []
        accuracy_log_scores = []

        precision_learning_scores = []
        recall_learning_scores = []
        f1_learning_scores = []
        accuracy_learning_scores = []

        precision_hybrid_scores = []
        recall_hybrid_scores = []
        f1_hybrid_scores = []
        accuracy_hybrid_scores = []

        for train_index, test_index in skf.split(data, data['Access Granted']):
            source_data = data.iloc[train_index]
            target_data = data.iloc[test_index]

            source_rules = []
            for _, row in source_data.iterrows():
                user_attr = {
                    "User ID": row["User ID"],
                    "User Role_Admin": row["User Role_Admin"] if "User Role_Admin" in row else 0,
                    "User Role_Faculty": row["User Role_Faculty"] if "User Role_Faculty" in row else 0,
                    "User Role_Student": row["User Role_Student"] if "User Role_Student" in row else 0,
                    "Behavioral Score": row["Behavioral Score"],
                }
                resource_attr = {
                    "Resource ID": row["Resource ID"],
                    "Resource Sensitivity_High": row["Resource Sensitivity_High"] if "Resource Sensitivity_High" in row else 0,
                    "Resource Sensitivity_Low": row["Resource Sensitivity_Low"] if "Resource Sensitivity_Low" in row else 0,
                    "Resource Sensitivity_Medium": row["Resource Sensitivity_Medium"] if "Resource Sensitivity_Medium" in row else 0,
                }
                action = {row["Action"]} if "Action" in row else set()
                decision = "permit" if row["Access Granted"] else "deny"
                source_rules.append(ABACRule(user_attr, resource_attr, action, decision))

            local_decisions = []
            for _, row in target_data.iterrows():
                user = row["User ID"]
                user_attr = {
                    "User ID": row["User ID"],
                    "User Role_Admin": row["User Role_Admin"] if "User Role_Admin" in row else 0,
                    "User Role_Faculty": row["User Role_Faculty"] if "User Role_Faculty" in row else 0,
                    "User Role_Student": row["User Role_Student"] if "User Role_Student" in row else 0,
                    "Behavioral Score": row["Behavioral Score"],
                }
                resource = row["Resource ID"]
                resource_attr = {
                    "Resource ID": row["Resource ID"],
                    "Resource Sensitivity_High": row["Resource Sensitivity_High"] if "Resource Sensitivity_High" in row else 0,
                    "Resource Sensitivity_Low": row["Resource Sensitivity_Low"] if "Resource Sensitivity_Low" in row else 0,
                    "Resource Sensitivity_Medium": row["Resource Sensitivity_Medium"] if "Resource Sensitivity_Medium" in row else 0,
                }
                action = row["Action"] if "Action" in row else ""
                decision = "permit" if row["Access Granted"] == 1 else "deny"


                if random.random() < 0.2:
                    decision = "permit" if decision == "deny" else "deny"

                local_decisions.append(AccessControlDecision(user, user_attr, resource, resource_attr, action, decision))

            with Pool() as pool:
                # Policy transfer using local log
                args_log = [(rule, local_decisions, policy_transfer_local_log, None) for rule in source_rules]
                target_rules_log = pool.map(process_rule, args_log)
                target_rules_log = [rule for sublist in target_rules_log for rule in sublist]

                # Policy transfer using local learning
                args_learning = [(rule, local_decisions, policy_transfer_local_learning, random_forest_policy_learner) for
                                 rule in source_rules]
                target_rules_learning = pool.map(process_rule, args_learning)
                target_rules_learning = [rule for sublist in target_rules_learning for rule in sublist]

                # Policy transfer using hybrid learning
                args_hybrid = [(rule, local_decisions, policy_transfer_hybrid_learning, random_forest_policy_learner) for
                               rule in source_rules]
                target_rules_hybrid = pool.map(process_rule, args_hybrid)
                target_rules_hybrid = [rule for sublist in target_rules_hybrid for rule in sublist]

            # Evaluate the transferred rules for each fold
            precision_log, recall_log, f1_score_log, accuracy_log = evaluate_rules(target_rules_log, local_decisions)
            precision_learning, recall_learning, f1_score_learning, accuracy_learning = evaluate_rules(target_rules_learning, local_decisions)
            precision_hybrid, recall_hybrid, f1_score_hybrid, accuracy_hybrid = evaluate_rules(target_rules_hybrid, local_decisions)

            # Append evaluation metrics for each fold
            precision_log_scores.append(precision_log)
            recall_log_scores.append(recall_log)
            f1_log_scores.append(f1_score_log)
            accuracy_log_scores.append(accuracy_log)

            precision_learning_scores.append(precision_learning)
            recall_learning_scores.append(recall_learning)
            f1_learning_scores.append(f1_score_learning)
            accuracy_learning_scores.append(accuracy_learning)

            precision_hybrid_scores.append(precision_hybrid)
            recall_hybrid_scores.append(recall_hybrid)
            f1_hybrid_scores.append(f1_score_hybrid)
            accuracy_hybrid_scores.append(accuracy_hybrid)

        # Calculate average evaluation metrics across all folds
        avg_precision_log = sum(precision_log_scores) / k
        avg_recall_log = sum(recall_log_scores) / k
        avg_f1_log = sum(f1_log_scores) / k
        avg_accuracy_log = sum(accuracy_log_scores) / k

        avg_precision_learning = sum(precision_learning_scores) / k
        avg_recall_learning = sum(recall_learning_scores) / k
        avg_f1_learning = sum(f1_learning_scores) / k
        avg_accuracy_learning = sum(accuracy_learning_scores) / k

        avg_precision_hybrid = sum(precision_hybrid_scores) / k
        avg_recall_hybrid = sum(recall_hybrid_scores) / k
        avg_f1_hybrid = sum(f1_hybrid_scores) / k
        avg_accuracy_hybrid = sum(accuracy_hybrid_scores) / k

        print("Cross-validation results:")
        print("Local Log: Accuracy={}".format(
            avg_accuracy_log))
        print("Local Learning: Accuracy={}".format(
             avg_accuracy_learning))
        print("Hybrid Learning: Accuracy={}".format(
            avg_accuracy_hybrid))

        # Visualization
        methods = ['Local Log', 'Local Learning', 'Hybrid Learning']
        precisions = [avg_precision_log, avg_precision_learning, avg_precision_hybrid]
        recalls = [avg_recall_log, avg_recall_learning, avg_recall_hybrid]
        f1_scores = [avg_f1_log, avg_f1_learning, avg_f1_hybrid]
        accuracies = [avg_accuracy_log, avg_accuracy_learning, avg_accuracy_hybrid]
        visualize_results(methods, precisions, recalls, f1_scores, accuracies)

        # Log the evaluation results
        logging.info(f"Cross-validation Results:")
        logging.info(
            f"Local Log: Avg. Precision={avg_precision_log}, Avg. Recall={avg_recall_log}, Avg. F1-Score={avg_f1_log}, Avg. Accuracy={avg_accuracy_log}")
        logging.info(
            f"Local Learning: Avg. Precision={avg_precision_learning}, Avg. Recall={avg_recall_learning}, Avg. F1-Score={avg_f1_learning}, Avg. Accuracy={avg_accuracy_learning}")
        logging.info(
            f"Hybrid Learning: Avg. Precision={avg_precision_hybrid}, Avg. Recall={avg_recall_hybrid}, Avg. F1-Score={avg_f1_hybrid}, Avg. Accuracy={avg_accuracy_hybrid}")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()