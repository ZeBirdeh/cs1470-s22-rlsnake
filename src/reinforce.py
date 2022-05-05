import os
import gym
import numpy as np
import tensorflow as tf

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# DO NOT ALTER MODEL CLASS OUTSIDE OF TODOs. OTHERWISE, YOU RISK INCOMPATIBILITY
# WITH THE AUTOGRADER AND RECEIVING A LOWER GRADE.


class Reinforce(tf.keras.Model):
    def __init__(self, state_size, num_actions):
        """
        The Reinforce class that inherits from tf.keras.Model
        The forward pass calculates the policy for the agent given a batch of states.

        :param state_size: number of parameters that define the state. You don't necessarily have to use this, 
                           but it can be used as the input size for your first dense layer.
        :param num_actions: number of actions in an environment
        """
        super(Reinforce, self).__init__()
        self.num_actions = num_actions
        
        # TODO: Define network parameters and optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.dense1_size = 10
        self.dense2_size = 10

        self.policy_network = tf.keras.Sequential([
            tf.keras.layers.Dense(self.dense1_size, input_shape=(state_size,), activation = 'relu'),
            tf.keras.layers.Dense(self.dense2_size, activation = 'relu'),
            tf.keras.layers.Dense(self.num_actions, activation = 'softmax'),
        ])


    def call(self, states):
        """
        Performs the forward pass on a batch of states to generate the action probabilities.
        This returns a policy tensor of shape [episode_length, num_actions], where each row is a
        probability distribution over actions for each state.

        :param states: An [episode_length, state_size] dimensioned array
        representing the history of states of an episode
        :return: A [episode_length,num_actions] matrix representing the probability distribution over actions
        for each state in the episode
        """
        # TODO: implement this ~
        #print(states.shape)
        #print(states)
        prob = self.policy_network(states)
        return prob

    def loss(self, states, actions, discounted_rewards):
        """
        Computes the loss for the agent. Make sure to understand the handout clearly when implementing this.

        :param states: A batch of states of shape [episode_length, state_size]
        :param actions: History of actions taken at each timestep of the episode (represented as an [episode_length] array)
        :param discounted_rewards: Discounted rewards throughout a complete episode (represented as an [episode_length] array)
        :return: loss, a Tensorflow scalar
        """
        # TODO: implement this
        # Hint: Use gather_nd to get the probability of each action that was actually taken in the episode.
        # - \sum log p(a | s) D(s, a)
        # get the probabilities of each action
        action_probs = self.call(states)
        # transpose actions to be vertical [episode_length, 1]
        action_indices = tf.reshape(tf.convert_to_tensor(actions, dtype=tf.int32), (-1, 1))
        probs = tf.gather_nd(action_probs, action_indices, batch_dims=1)
        #print(discounted_rewards, probs)
        loss = -tf.tensordot(tf.convert_to_tensor(discounted_rewards, dtype=tf.float32), tf.math.log(probs), axes=1)
        return loss

