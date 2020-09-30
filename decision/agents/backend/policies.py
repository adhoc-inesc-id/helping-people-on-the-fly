import numpy as np


def sample_action(policy):
    """ Alias for action from policy. """
    return action_from_policy(policy)


def action_from_policy(policy):
    """
    Samples an action from a given policy using numpy.
    """
    return np.random.choice(range(len(policy)), p=policy)


def policy_from_action(action, num_actions):
    """
    Build a deterministic policy given an action and the total number of actions.
    """
    policy = np.zeros((num_actions,))
    policy[action] = 1.0
    return policy


def deterministic_policy(action, num_actions):
    """
    Alias for policy_from_action(action, num_actions).
    """
    return policy_from_action(action, num_actions)


def random_action(num_actions):
    """Returns a random action sampled from a uniform distribution."""
    return np.random.choice(range(num_actions))


def random_policy(num_actions):
    """
    Returns a policy where all actions have equal probabilities, i.e., an uniform distribution.
    """
    return np.zeros((num_actions,)) + 1 / num_actions


def greedy_policy(q_values, randomly_solve_ties=True):
    """
    Returns a policy with certain probability of executing the action that maximizes Q(s,a).
    """
    policy = np.zeros((len(q_values),))
    if randomly_solve_ties:
        greedy_actions = np.argwhere(q_values == np.max(q_values)).reshape(-1)
        probability = 1.0 / len(greedy_actions)
        np.put(policy, greedy_actions, probability)
    else:
        greedy_action = q_values.argmax()
        policy[greedy_action] = 1.0
    return policy


def epsilon_greedy_policy(q_values, epsilon):
    """
    Returns a random policy if the exploration condition is met and the greedy policy otherwise.
    """
    return greedy_policy(q_values) if np.random.uniform(0, 1) > epsilon else random_policy(len(q_values))


def lazy_epsilon_greedy_policy(q_function, num_actions, epsilon):
    """
    An epsilon greedy policy which only runs if exploitation condition is met.
    """
    return greedy_policy(q_function()) if np.random.uniform(0, 1) > epsilon else random_policy(num_actions)
