import os
import sys
from pylab import *
import numpy as np
import tensorflow as tf
from deepq_model import DeepQNetwork
from agent import Agent
from game import SnakeGameAI, Direction, Point

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def visualize_episode(env, model):
    """
    HELPER - do not edit.
    Takes in an environment and a model and visualizes the model's actions for one episode.
    We recomend calling this function every 20 training episodes. Please remove all calls of 
    this function before handing in.

    :param env: The cart pole environment object
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

    # x_values = arange(0, len(total_rewards), 1)
    # y_values = total_rewards
    # plot(x_values, y_values)
    # xlabel('episodes')
    # ylabel('cumulative rewards')
    # title('Reward by Episode')
    # grid(True)
    # show()
    pass 


def train_step(model, state, action, reward, next_state, done):
    state = tf.convert_to_tensor(state, dtype= tf.float32)
    next_state = tf.convert_to_tensor(next_state, tf.float32)
    action = tf.convert_to_tensor(action, tf.float32)
    reward = tf.convert_to_tensor(reward, tf.float32)

    # sometimes we input data that is not "batched", 
    # this makes it a size one tensor in the batch dimension
    if len(state.shape) == 1:
        state = tf.expand_dims(state, 0)
        next_state = tf.expand_dims(next_state, 0)
        action = tf.expand_dims(action, 0)
        reward = tf.expand_dims(reward, 0)
        done = (done, )
    
    with tf.GradientTape() as tape:
        loss = model.loss(state, action, reward, next_state, done)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def main():
    # if len(sys.argv) != 2 or sys.argv[1] not in {"REINFORCE", "REINFORCE_BASELINE"}:
    #     print("USAGE: python assignment.py <Model Type>")
    #     print("<Model Type>: [REINFORCE/REINFORCE_BASELINE]")
    #     exit()

    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    num_actions = env.action_space.n
    gamma = 0.99

    # Initialize data
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

    # Initialize model
    # model = DeepQNetwork(state_size, num_actions, gamma)

    # # TODO:
    # # 1) Train your model for 650 episodes, passing in the environment and the agent.
    # # 1a) OPTIONAL: Visualize your model's performance every 20 episodes.
    # # 2) Append the total reward of the episode into a list keeping track of all of the rewards.
    # # 3) After training, print the average of the last 50 rewards you've collected.
    # all_rewards = []
    # for i in range(650):

    #     # TODO: run a simulation of the game, and call train_step on each step


    #     total_reward = train_step(  )
    #     if i%20 == 0:
    #         visualize_episode(env, model)
    #     all_rewards.append(total_reward)
    # print("Average of last 50 rewards: {0:.2f}".format(np.mean(all_rewards[:-50])))

    # # TODO: Visualize your rewards.
    # visualize_data(all_rewards)


if __name__ == '__main__':
    main()
