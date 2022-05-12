import os
import sys
from pylab import *
import numpy as np
import tensorflow as tf
from deepq_model import DeepQNetwork
from agent import Agent
from game import SnakeGameAI, Direction, Point
from helper import plot
import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def main():

    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    # load model from file
    if len(sys.argv) > 1 and sys.argv[1].lower() == "file":
        try:
            agent.model.load_weights('./checkpoints/' + sys.argv[2])
            f = open('./checkpoints/history-' + sys.argv[2], 'rb')
            plot_scores, plot_mean_scores, total_score, record, agent.n_games, agent.epsilon = pickle.load(f)
            f.close()
            print(f'Loading checkpoint from {sys.argv[2]}')
        except:
            pass


    while True:
        state_old = agent.get_state(game)

        final_move = agent.get_action(state_old)

        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

            if agent.n_games % 20 == 0:
                game.play_next()
                if len(sys.argv) > 1 and sys.argv[1].lower() == "file":
                    print(f'Saving checkpoint for {sys.argv[2]}')
                    agent.save_model('./checkpoints/' + sys.argv[2])
                    f = open('./checkpoints/history-' + sys.argv[2], 'wb')
                    pickle.dump((plot_scores, plot_mean_scores, total_score, 
                                record, agent.n_games, agent.epsilon), f)
                    f.close()
                    


if __name__ == '__main__':
    main()
