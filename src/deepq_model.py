import os
import gym
import numpy as np
import tensorflow as tf

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class DeepQNetwork(tf.keras.Model):
    def __init__(self, state_size, num_actions):
        """
        The DeepQNetwork class that inherits from tf.keras.Model
        The forward pass calculates the policy for the agent given a batch of states.

        :param state_size: number of parameters that define the state. You don't necessarily have to use this, 
                           but it can be used as the input size for your first dense layer.
        :param num_actions: number of actions in an environment
        """
        super(DeepQNetwork, self).__init__()
        self.num_actions = num_actions
        
        # TODO: Define network parameters and optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.dense1_size = 10
        self.dense2_size = 10

        self.qvalue_network = tf.keras.Sequential([
            tf.keras.layers.Dense(self.dense1_size, input_shape=(state_size,), activation = 'relu'),
            tf.keras.layers.Dense(self.num_actions),
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
        prob = self.qvalue_network(states)
        return prob

    def loss(self, states, actions, rewards, next_states):
        """
        Computes the loss for the agent. Make sure to understand the handout clearly when implementing this.

        :param states: A batch of states of shape [episode_length, state_size]
        :param actions: History of actions taken at each timestep of the episode (represented as an [episode_length] array)
        :param discounted_rewards: Discounted rewards throughout a complete episode (represented as an [episode_length] array)
        :return: loss, a Tensorflow scalar
        """
        # TODO:
        # Use MSE between Q_new and Q for loss
        #tf.reduce_max(next_qvals, axis=[1])
        pass

