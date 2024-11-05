Agent

- game
- model

Training:

- state = get_state(game)
- action = get_move(state):
  - model.predict()
- reward, game_over, score = game.play_step(action)
- new_state = get_state(game)
- remember
- model.train()

Game(PyGame)

- play_step(action)
  - reward, game_over, score

Model(PyTorch)

Linear_QNet(DQN)

- model.predict(state)
  - action

Reward:

- eat food: +10
- game over: -10
- else: 0

Action:

[1, 0, 0] -> straight (stay in current direction)
[0, 1, 0] -> right turn (depends on the current direction)
[0, 0, 1] -> left turn

Deep Q Learning

Q value stands for "quality of action"

0. Init Q Value (=init model)
1. Choose action (model.predict(state)) or random move
2. Perform action
3. Measure reward
4. Update Q value (+train model)
