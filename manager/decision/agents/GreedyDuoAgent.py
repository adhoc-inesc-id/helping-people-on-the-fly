from agents.Agent import Agent
from yaaf.policies import deterministic_policy


class GreedyDuoAgent(Agent):

    def __init__(self, index, mmdp):
        super().__init__("Greedy Teammate")
        self.mmdp = mmdp
        self.Qstar = mmdp.q_star
        self.pi_star = mmdp.pi_star
        self.joint_action_space = mmdp.joint_action_space
        self.num_disjoint_actions = mmdp.num_disjoint_actions
        self.id = index

    def policy(self, observation):
        qvalues = self.Qstar[self.mmdp.state_index(observation)]
        greedy_joint_action = self.joint_action_space[qvalues.argmax()]
        greedy_action = greedy_joint_action[self.id]
        return deterministic_policy(greedy_action, self.num_disjoint_actions)

    def _reinforce(self, timestep):
        pass
