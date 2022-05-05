import os
import sys
import gym
from pylab import *
import numpy as np
import tensorflow as tf
from reinforce import Reinforce
from reinforce_with_baseline import ReinforceWithBaseline

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def visualize_episode(env, model):
    """
    HELPER - do not edit.
    Takes in an enviornment and a model and visualizes the model's actions for one episode.
    We recomend calling this function every 20 training episodes. Please remove all calls of 
    this function before handing in.

    :param env: The cart pole enviornment object
    :param model: The model that will decide the actions to take
    """

    done = False
    state = env.reset()
    env.render()

    while not done:
        newState = np.reshape(state, [1, state.shape[0]])
        prob = model.call(newState)
        newProb = np.reshape(prob, prob.shape[1])
        action = np.random.choice(np.arange(newProb.shape[0]), p = newProb)

        state, _, done, _ = env.step(action)
        env.render()


def visualize_data(total_rewards):
    """
    HELPER - do not edit.
    Takes in array of rewards from each episode, visualizes reward over episodes

    :param total_rewards: List of rewards from all episodes
    """

    x_values = arange(0, len(total_rewards), 1)
    y_values = total_rewards
    plot(x_values, y_values)
    xlabel('episodes')
    ylabel('cumulative rewards')
    title('Reward by Episode')
    grid(True)
    show()


def discount(rewards, discount_factor=.99):
    """
    Takes in a list of rewards for each timestep in an episode, and
    returns a list of the discounted rewards for each timestep, which
    are calculated by summing the rewards for each future timestep, discounted
    by how far in the future it is.
    For example, in the simple case where the episode rewards are [1, 3, 5] 
    and discount_factor = .99 we would calculate:
    dr_1 = 1 + 0.99 * 3 + 0.99^2 * 5 = 8.8705
    dr_2 = 3 + 0.99 * 5 = 7.95
    dr_3 = 5
    and thus return [8.8705, 7.95 , 5].
    Refer to the slides for more details about how/why this is done.

    :param rewards: List of rewards from an episode [r_{t1},r_{t2},...]
    :param discount_factor: Gamma discounting factor to use, defaults to .99
    :returns: discounted_rewards: list containing the discounted rewards for each timestep in the original rewards list
    """
    # TODO: Compute discounted rewards
    rewards = np.array(rewards)
    temp_rewards = rewards
    for i in range(len(rewards)):
        # multiply by discount factor and shift left
        temp_rewards = np.append((discount_factor * temp_rewards)[1:], 0)
        # add to rewards
        rewards = rewards + temp_rewards
    return rewards




def generate_trajectory(env, model):
    """
    Generates lists of states, actions, and rewards for one complete episode.

    :param env: The openai gym environment
    :param model: The model used to generate the actions
    :returns: A tuple of lists (states, actions, rewards), where each list has length equal to the number of timesteps in the episode
    """
    states = []
    actions = []
    rewards = []
    state = env.reset()
    done = False

    while not done:
        # TODO:
        # 1) use model to generate probability distribution over next actions
        # 2) sample from this distribution to pick the next action
        action_prob = model(np.reshape(state, (1, state.shape[0])))
        #action = tf.random.categorical(action_logits, 1).numpy()[0][0]
        newProb = np.reshape(action_prob, action_prob.shape[1])
        action = np.random.choice(np.arange(newProb.shape[0]), p = newProb)

        states.append(state)
        actions.append(action)
        state, rwd, done, _ = env.step(action)
        rewards.append(rwd)

    return states, actions, rewards


def train(env, model):
    """
    This function should train your model for one episode.
    Each call to this function should generate a complete trajectory for one
    episode (lists of states, action_probs, and rewards seen/taken in the episode), and
    then train on that data to minimize your model loss.
    Make sure to return the total reward for the episode

    :param env: The openai gym environment
    :param model: The model
    :returns: The total reward for the episode
    """

    # TODO:
    # 1) Use generate trajectory to run an episode and get states, actions, and rewards.
    # 2) Compute discounted rewards.
    # 3) Compute the loss from the model and run backpropagation on the model.
    states, actions, rewards = generate_trajectory(env, model)
    discounted_rewards = discount(rewards)

    with tf.GradientTape() as tape:
        loss = model.loss(tf.convert_to_tensor(states, dtype=tf.float32), actions, discounted_rewards)

    gradients = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return tf.reduce_sum(rewards)



def main():
    if len(sys.argv) != 2 or sys.argv[1] not in {"REINFORCE", "REINFORCE_BASELINE"}:
        print("USAGE: python assignment.py <Model Type>")
        print("<Model Type>: [REINFORCE/REINFORCE_BASELINE]")
        exit()

    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    num_actions = env.action_space.n

    # Initialize model
    if sys.argv[1] == "REINFORCE":
        model = Reinforce(state_size, num_actions)
    elif sys.argv[1] == "REINFORCE_BASELINE":
        model = ReinforceWithBaseline(state_size, num_actions)

    # TODO:
    # 1) Train your model for 650 episodes, passing in the environment and the agent.
    # 1a) OPTIONAL: Visualize your model's performance every 20 episodes.
    # 2) Append the total reward of the episode into a list keeping track of all of the rewards.
    # 3) After training, print the average of the last 50 rewards you've collected.
    all_rewards = []
    for i in range(650):
        total_reward = train(env, model)
        if i%20 == 0:
            visualize_episode(env, model)
        all_rewards.append(total_reward)
    print("Average of last 50 rewards: {0:.2f}".format(np.mean(all_rewards[:-50])))

    # TODO: Visualize your rewards.
    visualize_data(all_rewards)


if __name__ == '__main__':
    main()
