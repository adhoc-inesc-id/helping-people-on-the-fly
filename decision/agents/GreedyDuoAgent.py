from agents.backend.agents.Agent import Agent
from agents.backend.policies import deterministic_policy


class GreedyDuoAgent(Agent):

    def __init__(self, index, mmdp):
        super().__init__("Greedy Teammate")
        self.mmdp = mmdp
        self.Qstar = mmdp.q_star
        self.pi_star = mmdp.pi_star
        self.id = index

    def policy(self, observation):
        qvalues = self.Qstar[self.mmdp.state_index(observation)]
        greedy_joint_action = self.mmdp.action_space[qvalues.argmax()]
        greedy_action = greedy_joint_action[self.id]
        return deterministic_policy(greedy_action, self.mmdp.num_disjoint_actions)

    def _reinforce(self, timestep):
        pass
