import tensorflow as tf
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
# from deepq_model import Linear_QNet, QTrainer
from deepq_model import DeepQNetwork
# from helper import plot

MAX_MEM = 10_000
BSIZE = 1000

class Agent:
    def __init__(self):
        self.n_games = 0
        self.e = 0 
        self.g = 0.9
        self.memory = deque(maxlen=MAX_MEM) 
        
        self.sz = 11
        self.num_ac = 3
        self.model = DeepQNetwork(self.sz, self.num_ac, self.g)

    def get_state(self, game):
        head = game.snake[0]
        head_location = [Point(head.x + 20, head.y), Point(head.x - 20, head.y), Point(head.x, head.y - 20), Point(head.x, head.y + 20)]
        snake_direction = [game.direction == Direction.L , game.direction == Direction.R, game.direction == Direction.UP, game.direction == Direction.D]
        return self.make_states(head_location, snake_direction, game)

    def make_states(self, points, directions, game):
        l, r, u, d = directions
        res = np.array([any([x and game.is_collision(y) for x,y in zip([r,l,u,d], points)]),
                        any([x and game.is_collision(y) for x,y in zip([u,d,l,r], points)]),  
                        any([x and game.is_collision(y) for x,y in zip([d,u,r,l], points)]),
                        *directions,
                        game.food.x < game.head.x, 
                        game.food.x > game.head.x,  
                        game.food.y < game.head.y,  
                        game.food.y > game.head.y], dtype=int)
        return res

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) 

    def train_long_memory(self):
        if len(self.memory) > BSIZE:
            mini_sample = random.sample(self.memory, BSIZE) 
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.train_step(self.model, states, actions, rewards, next_states, dones)


    def train_short_memory(self, state, action, reward, next_state, done):
        self.train_step(self.model, state, action, reward, next_state, done)

    def train_step(self, model, state, action, reward, next_state, done):
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        next_state = tf.convert_to_tensor(next_state, dtype=tf.float32)
        action = tf.convert_to_tensor(action, dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)

    
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


    def get_action(self, state):
        self.e = 90 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 250) < self.e:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), axis=0)
            prediction = self.model(state0)
            move = tf.math.argmax(tf.squeeze(prediction))
            final_move[move] = 1

        return final_move

    def save_model(self, path):
        self.model.save_weights(path)
