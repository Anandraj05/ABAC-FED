class ABACRule:
    def __init__(self, user_attr, resource_attr, action, decision):
        self.user_attr = user_attr
        self.resource_attr = resource_attr
        self.action = action
        self.decision = decision

    def __repr__(self):
        return f"<{self.user_attr}, {self.resource_attr}, {self.action}, {self.decision}>"

    def __eq__(self, other):
        if isinstance(other, ABACRule):
            return (
                self.user_attr == other.user_attr and
                self.resource_attr == other.resource_attr and
                self.action == other.action and
                self.decision == other.decision
            )
        return False

    def __hash__(self):
        return hash((frozenset(self.user_attr.items()), frozenset(self.resource_attr.items()), self.action, self.decision))


class AccessControlDecision:
    def __init__(self, user, user_attr, resource, resource_attr, action, decision):
        self.user = user
        self.user_attr = user_attr
        self.resource = resource
        self.resource_attr = resource_attr
        self.action = action
        self.decision = decision

    def __repr__(self):
        return f"<{self.user}, {self.user_attr}, {self.resource}, {self.resource_attr}, {self.action}, {self.decision}>"


def is_similar_rule_decision(rule, decision):
    return (
        all(attr_val == decision.user_attr.get(attr_key) or (str(attr_val) == str(decision.user_attr.get(attr_key))) for attr_key, attr_val in rule.user_attr.items()) and
        all(attr_val == decision.resource_attr.get(attr_key) or (str(attr_val) == str(decision.resource_attr.get(attr_key))) for attr_key, attr_val in rule.resource_attr.items()) and
        rule.action == decision.action
    )


def is_similar_rules(rule1, rule2):
    return (
        all(rule1.user_attr.get(attr_key) == attr_val or (str(rule1.user_attr.get(attr_key)) == str(attr_val)) for attr_key, attr_val in rule2.user_attr.items()) and
        all(rule1.resource_attr.get(attr_key) == attr_val or (str(rule1.resource_attr.get(attr_key)) == str(attr_val)) for attr_key, attr_val in rule2.resource_attr.items()) and
        rule1.action == rule2.action
    )


def is_consistent_rule_decision(rule, decision):
    return is_similar_rule_decision(rule, decision) and rule.decision == decision.decision


def is_consistent_rules(rule1, rule2):
    return is_similar_rules(rule1, rule2) and rule1.decision == rule2.decision


def adapt_rules(rule1, rule2):
    if rule1.action != rule2.action:
        return []

    mutual_user_attr = {attr_key: attr_val for attr_key, attr_val in rule1.user_attr.items() if
                        attr_key in rule2.user_attr and (rule2.user_attr.get(attr_key) == attr_val or (rule2.user_attr.get(attr_key) == str(attr_val) if isinstance(attr_val, str) else rule2.user_attr.get(attr_key) == attr_val))}
    mutual_resource_attr = {attr_key: attr_val for attr_key, attr_val in rule1.resource_attr.items() if
                            attr_key in rule2.resource_attr and (rule2.resource_attr.get(attr_key) == attr_val or (rule2.resource_attr.get(attr_key) == str(attr_val) if isinstance(attr_val, str) else rule2.resource_attr.get(attr_key) == attr_val))}

    adapted_rules = []

    if mutual_user_attr and mutual_resource_attr:
        adapted_rules.append(ABACRule(mutual_user_attr, mutual_resource_attr, rule1.action, "permit"))
        adapted_rules.append(ABACRule(mutual_user_attr, mutual_resource_attr, rule1.action, "deny"))

    non_mutual_user_attr = {attr_key: attr_val for attr_key, attr_val in rule1.user_attr.items() if
                            not rule2.user_attr.get(attr_key) or (rule2.user_attr.get(attr_key) != attr_val and (not isinstance(attr_val, str) or rule2.user_attr.get(attr_key) != str(attr_val)))}
    non_mutual_resource_attr = {attr_key: attr_val for attr_key, attr_val in rule1.resource_attr.items() if
                                not rule2.resource_attr.get(attr_key) or (rule2.resource_attr.get(attr_key) != attr_val and (not isinstance(attr_val, str) or rule2.resource_attr.get(attr_key) != str(attr_val)))}

    if non_mutual_user_attr and non_mutual_resource_attr:
        adapted_rules.append(ABACRule(non_mutual_user_attr, non_mutual_resource_attr, rule1.action, rule1.decision))
    elif non_mutual_user_attr:
        adapted_rules.append(ABACRule(non_mutual_user_attr, rule1.resource_attr, rule1.action, rule1.decision))
        adapted_rules.append(ABACRule(non_mutual_user_attr, rule2.resource_attr, rule2.action, rule2.decision))
    elif non_mutual_resource_attr:
        adapted_rules.append(ABACRule(rule1.user_attr, non_mutual_resource_attr, rule1.action, rule1.decision))
        adapted_rules.append(ABACRule(rule2.user_attr, non_mutual_resource_attr, rule2.action, rule2.decision))

    return adapted_rules