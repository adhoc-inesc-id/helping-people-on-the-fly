import random

from decision.agents.backend.agents.Agent import Agent
from decision.agents.backend.policies import deterministic_policy


class SubOptimalAgent(Agent):

    def __init__(self, index, mmdp, stay_probability=0.30):
        super().__init__("Sub Optimal Teammate")
        self.mmdp = mmdp
        self.Qstar = mmdp.q_star
        self.pi_star = mmdp.pi_star
        self.id = index
        self.stay_probability = stay_probability

    def policy(self, observation):
        qvalues = self.Qstar[self.mmdp.state_index(observation)]
        greedy_joint_action = self.mmdp.action_space[qvalues.argmax()]
        greedy_action = greedy_joint_action[self.id]

        if random.random() <= self.stay_probability:
            return deterministic_policy(self.mmdp.stay, self.mmdp.num_disjoint_actions)
        else:
            return deterministic_policy(greedy_action, self.mmdp.num_disjoint_actions)

    def _reinforce(self, timestep):
        pass
