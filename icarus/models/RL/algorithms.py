from icarus.registry import register_rl_algorithms
import pandas as pd
import random


class EpsilonGreedy():
    def __init__(self, epsilon, n_actions):
        self.epsilon = epsilon # probability of explore
        self.counts = [0 for col in range(n_actions)]
        self.values = [0.0 for col in range(n_actions)]
        return

    def ind_max(self, x):
        m = max(x)
        return x.index(m)

    def select_action(self):
        if random.random() > self.epsilon:
            return self.ind_max(self.values)
        else:
            return random.randrange(len(self.values))

    def update(self, action, reward):
        self.counts[action] = self.counts[action] + 1
        n = self.counts[action]

        value = self.values[action]
        new_value = ((n-1) / float(n)) * value + (1 / float(n)) * reward # weighted average of the previously estimated value and the reward we just received
        self.values[action] = new_value
        return

class Qlearning:
    def __init__(self,
                 possible_states,
                 possible_actions,
                 initial_reward = 0,
                 epsilon = .5,
                 learning_rate = .001,
                 discount_factor = 1.0):
        """
        Initialise the q learning class with an initial matrix and the parameters for learning.
        :param possible_states: list of states the agent can be in
        :param possible_actions: list of actions the agent can perform
        :param initial_reward: the initial Q-values to be used in the matrix
        :param learning_rate: the learning rate used for Q-learning
        :param discount_factor: the discount factor used for Q-learning
        """
        # Initialize the matrix with Q-values
        init_data = [[float(initial_reward) for _ in possible_states]
                     for _ in possible_actions]
        self._qmatrix = pd.DataFrame(data=init_data,
                                     index=possible_actions,
                                     columns=possible_states)

        # Initialize epsilon greedy policy
        self._epsilon_greedy = EpsilonGreedy(epsilon, len(possible_actions))

        # Save the parameters
        self._learn_rate = learning_rate
        self._discount_factor = discount_factor

    def get_best_action(self, state):
        """
        Retrieve the action resulting in the highest Q-value for a given state.
        :param state: the state for which to determine the best action
        :return: the best action from the given state
        """
        # Return the action (index) with maximum Q-value
        return self._qmatrix[[state]].idxmax().iloc[0]

    def update_model(self, state, action, reward, next_state):
        """
        Update the Q-values for a given observation.
        :param state: The state the observation started in
        :param action: The action taken from that state
        :param reward: The reward retrieved from taking action from state
        :param next_state: The resulting next state of taking action from state
        """
        # Update q_value for a state-action pair Q(s,a):
        # Q(s,a) = Q(s,a) + α( r + γmaxa' Q(s',a') - Q(s,a) )
        q_sa = self._qmatrix.loc[action, state]
        max_q_sa_next = self._qmatrix.loc[self.get_best_action(next_state), next_state]
        r = reward
        alpha = self._learn_rate
        gamma = self._discount_factor

        # Do the computation
        new_q_sa = q_sa + alpha * (r + gamma * max_q_sa_next - q_sa)
        self._qmatrix.at[action, state] = new_q_sa


@register_rl_algorithms('Q-LEARNING')
class Q_learning:
    def __init__(self, *args, **kwargs):
        self.node = kwargs['node']
        self.network_model = kwargs['network_model']
        kwargs.pop('node')
        kwargs.pop('network_model')
        self.model = Qlearning(*args, **kwargs)
    
    def get_node_state(self):
        """Returns a state/observation of the current environment

        Parameters
        ----------
        network : icarus.execution.network.NetworkModel

        Returns
        ----------
        state : list
            Current observation
        """
        state = self.network_model.n_contents*['0']
        for content in self.network_model.cache[self.node].dump():
            state[content-1] = '1'

        return int(''.join(state), 2)

    def get_best_action(self, state):
        return self.model.get_best_action(state)

    def train(self, *args):
        return self.model.update_model(*args)
