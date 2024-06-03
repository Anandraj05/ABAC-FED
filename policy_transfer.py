from abac_rules import ABACRule, is_similar_rule_decision, is_consistent_rule_decision, adapt_rules

def policy_transfer_local_log(rule, local_decisions):
    matched_decisions = [d for d in local_decisions if is_similar_rule_decision(rule, d)]
    consistent_decisions = [d for d in matched_decisions if is_consistent_rule_decision(rule, d)]
    inconsistent_decisions = [d for d in matched_decisions if not is_consistent_rule_decision(rule, d)]

    if consistent_decisions and not inconsistent_decisions:
        return [rule]
    elif inconsistent_decisions:
        adapted_rules = set()
        for decision in inconsistent_decisions:
            adapted_rules.update(adapt_rules(rule, ABACRule(decision.user_attr, decision.resource_attr, decision.action, decision.decision)))
        return list(adapted_rules)
    else:
        return [rule]

def policy_transfer_local_learning(rule, local_decisions, policy_learner):
    matched_decisions = [d for d in local_decisions if is_similar_rule_decision(rule, d)]
    consistent_decisions = [d for d in matched_decisions if is_consistent_rule_decision(rule, d)]
    inconsistent_decisions = [d for d in matched_decisions if not is_consistent_rule_decision(rule, d)]

    if consistent_decisions and not inconsistent_decisions:
        return [rule]
    elif inconsistent_decisions:
        return policy_learner(inconsistent_decisions)
    else:
        return [rule]

def policy_transfer_hybrid_learning(rule, local_decisions, policy_learner):
    matched_decisions = [d for d in local_decisions if is_similar_rule_decision(rule, d)]
    consistent_decisions = [d for d in matched_decisions if is_consistent_rule_decision(rule, d)]
    inconsistent_decisions = [d for d in matched_decisions if not is_consistent_rule_decision(rule, d)]

    if consistent_decisions and not inconsistent_decisions:
        return [rule]
    elif inconsistent_decisions:
        adapted_rules = set(adapt_rules(rule, ABACRule(inconsistent_decisions[0].user_attr, inconsistent_decisions[0].resource_attr, inconsistent_decisions[0].action, inconsistent_decisions[0].decision)))
        learned_rules = policy_learner(inconsistent_decisions)
        return list(adapted_rules.union(learned_rules))  # Use set union to avoid duplicates
    else:
        return [rule]

def process_rule(args):
    source_rule, local_decisions, policy_transfer_func, policy_learner = args
    policy_transfer_functions = {
        'policy_transfer_local_log': policy_transfer_local_log,
        'policy_transfer_local_learning': policy_transfer_local_learning,
        'policy_transfer_hybrid_learning': policy_transfer_hybrid_learning
    }
    func_name = policy_transfer_func.__name__  # Get the name of the function
    if func_name == 'policy_transfer_local_log':
        return policy_transfer_functions[func_name](source_rule, local_decisions)
    else:
        return policy_transfer_functions[func_name](source_rule, local_decisions, policy_learner)
