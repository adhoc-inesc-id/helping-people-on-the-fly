from typing import Sequence

import numpy as np
from yaaf.policies import deterministic_policy

from agents.Agent import Agent
from environment.mdp.MultiAgentMarkovDecisionProcess import MultiAgentMarkovDecisionProcess as MMDP


class BOPA(Agent):

    def __init__(self, possible_mmdps: Sequence[MMDP], index: int = 0):
        super(BOPA, self).__init__("BOPA")
        self._mmdps = possible_mmdps
        self._beliefs = np.ones(len(self._mmdps)) / len(self._mmdps)
        self._q_stars = [mmdp.q_star for mmdp in possible_mmdps]
        self._optimal_policies = [mmdp.pi_star for mmdp in possible_mmdps]
        self.joint_action_space = [(ja[0], ja[1]) for ja in possible_mmdps[0].joint_action_space]
        self.num_disjoint_actions = possible_mmdps[0].num_disjoint_actions
        self.index = index

    def policy(self, observation):
        return self._greedy_policy(observation) if len(self._mmdps) == 1 else self._bopa_policy(observation)

    def _greedy_policy(self, observation):
        qstar = self._q_stars[0]
        qvalues = qstar[self._mmdps[0].state_index(observation)]
        greedy_joint_action = self.joint_action_space[qvalues.argmax()]
        greedy_action = greedy_joint_action[self.index]
        return deterministic_policy(greedy_action, self.num_disjoint_actions)

    def _bopa_policy(self, observation):
        num_actions = self._optimal_policies[0].shape[1]
        x = self._mmdps[0].state_index(observation)
        shared_policy = np.zeros(num_actions)
        for a in range(num_actions):
            pi_mmdps_actions = np.array([pi[x][a] for pi in self._optimal_policies])
            shared_policy[a] = pi_mmdps_actions.dot(self._beliefs)
        individual_policy = self._disjoint_policy(shared_policy)
        return individual_policy

    def loss(self, observation):
        num_actions = self._optimal_policies[0].shape[1]
        num_models = len(self._mmdps)
        losses = np.zeros((num_models, num_actions))
        for a in range(num_actions):
            for m, mmdp in enumerate(self._mmdps):
                V = mmdp.v
                Q = mmdp.q_star
                x = mmdp.state_index(observation)
                losses[m, a] = V[x] - Q[x, a]
        return losses

    def _disjoint_policy(self, policy):
        pi = np.zeros(self.num_disjoint_actions)
        for a, action_probability in enumerate(policy):
            joint_action = self.joint_action_space[a]
            action = joint_action[self.index]
            pi[action] += action_probability
        return pi

    def _bopa_reinforce(self, timestep):
        a = timestep.action
        state = timestep.observation
        next_state = timestep.next_observation
        model_priors = np.ones(len(self._mmdps)) / len(self._mmdps)
        for m, mmdp in enumerate(self._mmdps):
            x, y = mmdp.state_index(state), mmdp.state_index(next_state)
            Qstar = self._q_stars[m][x]
            greedy_joint_actions = list(np.argwhere(Qstar == np.amax(Qstar)).flatten())
            greedy_joint_action = [self.joint_action_space[a] for a in greedy_joint_actions]
            greedy_teammate_actions = [a[1] for a in greedy_joint_action]
            accum = 0.0
            for a_other in greedy_teammate_actions:
                ja = self.joint_action_space.index((a, a_other))
                P = mmdp.P[ja, x, y]
                policy = 1.0 / len(greedy_teammate_actions)
                accum += P * policy
            model_priors[m] = accum
        new_beliefs = model_priors * self._beliefs + 0.0001
        self._beliefs = new_beliefs / np.sum(new_beliefs)
        info = {f"MMDP #{i}'s Likelihood": prob for i, prob in enumerate(self._beliefs)}
        loss = self.loss(timestep.observation)
        # TODO - Add to info
        return info

    def _greedy_reinforce(self):
        self._beliefs = np.ones(len(self._mmdps)) / len(self._mmdps)
        info = {f"MMDP #{i}'s Likelihood": prob for i, prob in enumerate(self._beliefs)}
        return info

    def _reinforce(self, timestep):
        return self._greedy_reinforce() if len(self._mmdps) == 1 else self._bopa_reinforce(timestep)

class TaskInferenceAnalyzer:

    def __init__(self):
        self.name = "Task Inference Analyzer"

    def reset(self):
        del self._belief_history

    def result(self):
        return np.array(self._belief_history)

    def __call__(self, timestep):
        agent_info = timestep.info["agent"]
        num_possible_tasks = len(agent_info.keys())
        if not hasattr(self, "_belief_history"):
            self._belief_history = [[1/num_possible_tasks for _ in range(num_possible_tasks)]]
        beliefs = [agent_info[f"MMDP #{i}'s Likelihood"] for i in range(num_possible_tasks)]
        self._belief_history.append(beliefs)
