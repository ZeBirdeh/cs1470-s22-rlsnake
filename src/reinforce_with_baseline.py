import os
import gym
import numpy as np
import tensorflow as tf

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# DO NOT ALTER MODEL CLASS OUTSIDE OF TODOs. OTHERWISE, YOU RISK INCOMPATIBILITY
# WITH THE AUTOGRADER AND RECEIVING A LOWER GRADE.


class ReinforceWithBaseline(tf.keras.Model):
    def __init__(self, state_size, num_actions):
        """
        The ReinforceWithBaseline class that inherits from tf.keras.Model.

        The forward pass calculates the policy for the agent given a batch of states. During training,
        ReinforceWithBaseLine estimates the value of each state to be used as a baseline to compare the policy's
        performance with.

        :param state_size: number of parameters that define the state. You don't necessarily have to use this, 
                           but it can be used as the input size for your first dense layer.
        :param num_actions: number of actions in an environment
        """
        super(ReinforceWithBaseline, self).__init__()
        self.num_actions = num_actions

        # TODO: Define actor network parameters, critic network parameters, and optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.dense1_size = 25
        self.dense2_size = 25
        #self.dense3_size = 16

        self.value_dense1_size = 8
        self.value_dense2_size = 8

        self.policy_network = tf.keras.Sequential([
            tf.keras.layers.Dense(self.dense1_size, input_shape=(state_size,), activation = 'relu'),
            tf.keras.layers.Dense(self.dense2_size, activation = 'relu'),
            #tf.keras.layers.Dense(self.dense3_size),
            tf.keras.layers.Dense(self.num_actions, activation = 'softmax'),
        ])

        self.value_network = tf.keras.Sequential([
            tf.keras.layers.Dense(self.value_dense1_size, input_shape=(state_size,), activation = 'relu'),
            tf.keras.layers.Dense(self.value_dense2_size, activation = 'relu'),
            tf.keras.layers.Dense(1)
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
        # TODO: implement this!
        prob = self.policy_network(states)
        return prob

    def value_function(self, states):
        """
        Performs the forward pass on a batch of states to calculate the value function, to be used as the
        critic in the loss function.

        :param states: An [episode_length, state_size] dimensioned array representing the history of states
        of an episode.
        :return: A [episode_length] matrix representing the value of each state.
        """
        # TODO: implement this :D
        return self.value_network(states)

    def loss(self, states, actions, discounted_rewards):
        """
        Computes the loss for the agent. Refer to the lecture slides referenced in the handout to see how this is done.

        Remember that the loss is similar to the loss as in part 1, with a few specific changes.

        1) In your actor loss, instead of element-wise multiplying with discounted_rewards, you want to element-wise multiply with your advantage. 
        See handout/slides for definition of advantage.
        
        2) In your actor loss, you must use tf.stop_gradient on the advantage to stop the loss calculated on the actor network 
        from propagating back to the critic network.
        
        3) See handout/slides for how to calculate the loss for your critic network.

        :param states: A batch of states of shape (episode_length, state_size)
        :param actions: History of actions taken at each timestep of the episode (represented as an [episode_length] array)
        :param discounted_rewards: Discounted rewards throughout a complete episode (represented as an [episode_length] array)
        :return: loss, a TensorFlow scalar
        """
        # TODO: implement this :)
        # Hint: use tf.gather_nd (https://www.tensorflow.org/api_docs/python/tf/gather_nd) to get the probabilities of the actions taken by the model
        # Calculate the advantage, D(s, a) - V(s)
        discounted_rewards = tf.convert_to_tensor(discounted_rewards, dtype=tf.float32)
        state_values = tf.squeeze(self.value_function(states))
        advantage = discounted_rewards - state_values

        loss_critic = tf.tensordot(advantage, advantage, axes=1)

        # Calculate actor loss
        action_probs = self.call(states)
        # transpose actions to be vertical [episode_length, 1]
        action_indices = tf.reshape(tf.convert_to_tensor(actions, dtype=tf.int32), (-1, 1))
        probs = tf.gather_nd(action_probs, action_indices, batch_dims=1)
        # use stop_gradient to treat advantage as a constant
        loss_actor = -tf.tensordot(tf.stop_gradient(advantage), tf.math.log(probs), axes=1)
        return loss_critic + loss_actor
