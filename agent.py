import torch
import random
import numpy as np
from collections import deque
from game import SnakeGame, Direction, Point
from model import QNetwork, Trainer
from utils import plot

MAX_MEMORY_SIZE = 100_000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001

class SnakeAgent:

    def __init__(self):
        self.game_count = 0
        self.epsilon = 0
        self.discount_factor = 0.9
        self.memory = deque(maxlen=MAX_MEMORY_SIZE)
        self.q_network = QNetwork(11, 256, 3)
        self.trainer = Trainer(self.q_network, lr=LEARNING_RATE, gamma=self.discount_factor)

    def get_state(self, game):
        head_position = game.snake[0]
        left_point = Point(head_position.x - 20, head_position.y)
        right_point = Point(head_position.x + 20, head_position.y)
        up_point = Point(head_position.x, head_position.y - 20)
        down_point = Point(head_position.x, head_position.y + 20)

        direction_left = game.direction == Direction.LEFT
        direction_right = game.direction == Direction.RIGHT
        direction_up = game.direction == Direction.UP
        direction_down = game.direction == Direction.DOWN

        state_vector = [
            (direction_right and game.is_collision(right_point)) or 
            (direction_left and game.is_collision(left_point)) or 
            (direction_up and game.is_collision(up_point)) or 
            (direction_down and game.is_collision(down_point)),
            (direction_up and game.is_collision(right_point)) or 
            (direction_down and game.is_collision(left_point)) or 
            (direction_left and game.is_collision(up_point)) or 
            (direction_right and game.is_collision(down_point)),
            (direction_down and game.is_collision(right_point)) or 
            (direction_up and game.is_collision(left_point)) or 
            (direction_right and game.is_collision(up_point)) or 
            (direction_left and game.is_collision(down_point)),
            direction_left,
            direction_right,
            direction_up,
            direction_down,
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y
        ]

        return np.array(state_vector, dtype=int)

    def remember_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        mini_batch = random.sample(self.memory, min(len(self.memory), BATCH_SIZE))
        states, actions, rewards, next_states, dones = zip(*mini_batch)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def choose_action(self, state):
        self.epsilon = 80 - self.game_count
        action_vector = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            random_action = random.randint(0, 2)
            action_vector[random_action] = 1
        else:
            state_tensor = torch.tensor(state, dtype=torch.float)
            predicted_q_values = self.q_network(state_tensor)
            best_action = torch.argmax(predicted_q_values).item()
            action_vector[best_action] = 1

        return action_vector


def train_snake_agent():
    score_history = []
    mean_score_history = []
    total_score = 0
    highest_score = 0
    agent = SnakeAgent()
    game_instance = SnakeGame()
    
    while True:
        current_state = agent.get_state(game_instance)
        action_to_take = agent.choose_action(current_state)

        reward, game_over, current_score = game_instance.play_step(action_to_take)
        new_state = agent.get_state(game_instance)

        agent.train_short_memory(current_state, action_to_take, reward, new_state, game_over)
        agent.remember_experience(current_state, action_to_take, reward, new_state, game_over)

        if game_over:
            game_instance.reset()
            agent.game_count += 1
            agent.train_long_memory()

            if current_score > highest_score:
                highest_score = current_score
                agent.q_network.save_model()

            print(f'Game {agent.game_count} | Score: {current_score} | Record: {highest_score}')

            score_history.append(current_score)
            total_score += current_score
            mean_score = total_score / agent.game_count
            mean_score_history.append(mean_score)
            plot(score_history, mean_score_history)


if __name__ == '__main__':
    train_snake_agent()
