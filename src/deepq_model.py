import os
import numpy as np
import tensorflow as tf

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class DeepQNetwork(tf.keras.Model):
    def __init__(self, state_size, num_actions, gamma):
        """
        The DeepQNetwork class that inherits from tf.keras.Model
        The forward pass calculates the policy for the agent given a batch of states.
        :param state_size: number of parameters that define the state. You don't necessarily have to use this, 
                           but it can be used as the input size for your first dense layer.
        :param num_actions: number of actions in an environment
        """
        super(DeepQNetwork, self).__init__()
        self.num_actions = num_actions
        self.gamma = gamma
        
        # TODO: Define network parameters and optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.dense_size = 256

        self.qvalue_network = tf.keras.Sequential([
            #tf.keras.layers.Dense(self.dense1_size, input_shape=(state_size,), activation = 'relu'),
            tf.keras.layers.Dense(self.dense_size, activation = 'relu'),
            tf.keras.layers.Dense(self.num_actions)
        ])


    def call(self, states):
        """
        Performs the forward pass on a batch of states to generate the action probabilities.
        This returns a policy tensor of shape [episode_length, num_actions], where each row is a
        probability distribution over actions for each state.
        :param states: An [episode_length, state_size] dimensioned array
        representing the history of states of an episode
        :return: A [episode_length,num_actions] matrix representing the Q values of each action
        for each state in the episode
        """
        qvals = self.qvalue_network(states)
        return qvals

    def loss(self, states, actions, rewards, next_states, done):
        """
        Computes the loss for the agent. Make sure to understand the handout clearly when implementing this.
        :param states: A batch of states of shape [episode_length, state_size]
        :param actions: History of actions taken at each timestep of the episode (represented as an [episode_length] array)
        :param discounted_rewards: Discounted rewards throughout a complete episode (represented as an [episode_length] array)
        :return: loss, a Tensorflow scalar
        """
        # Use MSE between Q_new and Q for loss
        allq_vals = self.call(states)
        # output is batch_size x num_actions
        # need to grab the Q-value corresponding to the action we chose
        predq = tf.gather_nd(allq_vals, tf.where(actions))
        # output is batch_size x 1

        # for each index, if the game is done, we don't add the value of the next state
        targetq = rewards.numpy()
        #np.zeros(rewards.shape)
        for i in range(len(done)):
            #targetq[i] = rewards[i]
            if not(done[i]):
                # there might be a bug with self.call(next_states[i]) since data isn't batched
                # may have to expand dims or mess with output dims
                input_states = tf.expand_dims(next_states[i], axis=0)
                future_state_val = tf.reduce_max(self.call(input_states))
                targetq[i] += self.gamma * future_state_val

        diffq = predq - targetq
        loss = tf.reduce_sum(tf.multiply(diffq, diffq))
        return loss