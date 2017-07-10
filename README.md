# RL A3C Pytorch

This repository includes my implementation of Asynchronous Advantage Actor-Critic (A3C) in Pytorch an algorithm from Google Deep Mind's paper "Asynchronous Methods for Deep Reinforcement Learning".

## Requirements

- Python 3+
- OpenAI Gym 
- Pytorch

## A3C GRU
![pong](https://user-images.githubusercontent.com/18737539/27256296-286fdc44-53ce-11e7-8445-e14a173af00d.gif)
![breakout](https://user-images.githubusercontent.com/18737539/27256294-2627d298-53ce-11e7-8dd4-7997ae5386ae.gif)

I implemented an A3C model, using GRU's rather than LSTM's, and trained it on two atari 2600 environments, that are PongDeterministic-v4 and BreakoutDeterministic-v4 provided in the Openai Gym. So far my model currently has completed the game of Pong and has an average score of 329.5 in Breakout. Saved models in Pong-v4 and Breakout-v4 folder. Trained models may not run properly if you have older version gym and v3 atari. To make sure they run properly u need to keep gym version <= 0.9.1 and atari-py version <= 0.1.1.

You can use RMSprop and Adam for sharing statistics between the networks.

- Adam optimizer was used for sharing weights in the saved .pkl files.

## Training
Limit number of worker threads to number of cpu cores available as too many threads (e.g. more than one thread per cpu core available) will actually result in decrease of training speed and effectiveness.

To train agent in PongDeterministic-v4 environment with 4 different worker threads:

```
python main.py --env-name PongDeterministic-v4 --num-processes 4
```

or to train agent in BreakoutDeterministic-v4 environment:

```
python main.py --env-name BreakoutDeterministic-v4 --num-processes 4
```

Pong will approximately takes 40 minutes to finish if 4 worker threads are used.

Breakout, for me, took more than 10 hours to reach a score of 300 with 4 workers and less than 5 hours using 8 worker threads.

The test.py file will save weights with score more than or equal to 300 while the training will still continue.

## Test Run on Gym
To run a 50 episode gym evaluation with trained model

```
python gym_test.py --env-name PongDeterministic-v4 --num-episodes 50
```
or

```
python gym_test.py --env-name BreakoutDeterministic-v4 --num-episodes 50
```

## References

- https://arxiv.org/pdf/1602.01783.pdf
- https://github.com/ikostrikov/pytorch-a3c
- https://github.com/dgriff777/rl_a3c_pytorch
