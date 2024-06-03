from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from abac_rules import ABACRule


def random_forest_policy_learner(decisions):
    X = []
    y = []
    for decision in decisions:
        user_attrs = [{"user_{}".format(k): v} for k, v in decision.user_attr.items()]
        resource_attrs = [{"resource_{}".format(k): v} for k, v in decision.resource_attr.items()]
        X.append({**dict(pair for d in user_attrs for pair in d.items()), **dict(pair for d in resource_attrs for pair in d.items())})
        y.append(decision.decision)

    if len(set(y)) == 1:
        return [ABACRule(decisions[0].user_attr, decisions[0].resource_attr, decisions[0].action, decisions[0].decision)]

    vec = DictVectorizer()
    X = vec.fit_transform(X).toarray()
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf.fit(X, y)

    learned_rules = []
    for estimator in clf.estimators_:
        for path in estimator.decision_path(X).toarray():
            user_expr = {}
            resource_expr = {}
            for node_id, value in enumerate(path):
                if value:
                    feature = estimator.tree_.feature[node_id]
                    threshold = estimator.tree_.threshold[node_id]
                    feature_name = vec.get_feature_names_out()[feature]
                    if feature_name.startswith("user_"):
                        attr_key, attr_val = feature_name.split("=", 1)
                        attr_key = attr_key.split("_", 1)[1]
                        user_expr[attr_key] = user_expr.get(attr_key, set()) | {attr_val}
                    elif feature_name.startswith("resource_"):
                        attr_key, attr_val = feature_name.split("=", 1)
                        attr_key = attr_key.split("_", 1)[1]
                        resource_expr[attr_key] = resource_expr.get(attr_key, set()) | {attr_val}
            decision = "permit" if estimator.tree_.value[node_id][0][1] > estimator.tree_.value[node_id][0][0] else "deny"
            learned_rules.append(ABACRule(user_expr, resource_expr, decisions[0].action, decision))

    return learned_rules