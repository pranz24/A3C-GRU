from __future__ import print_function
import argparse
import os
import torch
import torch.multiprocessing as mp
from environment import create_atari_env
from A3C_model import A3C
from train import train
from test import test
import shared_optim

# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training Parameters
parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--tau', type=float, default=1.00, metavar='T',
                    help='parameter for GAE (default: 1.00)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--num-processes', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 4')
parser.add_argument('--num-steps', type=int, default=20, metavar='NS',
                    help='number of forward steps in A3C (default: 20)')
parser.add_argument('--max-episode-length', type=int, default=10000, metavar='M',
                    help='maximum length of an episode (default: 10000)')
parser.add_argument('--env-name', default='BreakoutDeterministic-v4', metavar='ENV',
                    help='environment to train on (default: BreakoutDeterministic-v4)')
parser.add_argument('--optimizer', default='Adam', metavar='OPT',
                    help='shares optimizer choice of Adam or RMSprop')

if __name__ == '__main__':
    #Number of thread per cpu cores
    os.environ['OMP_NUM_THREADS'] = '1'
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    env = create_atari_env(args.env_name)
    shared_model = A3C(
        env.observation_space.shape[0], env.action_space)
    #Too load shared weights from saved pkl file use
    #shared_model.load_state_dict(torch.load('./Breakout-v4/A3C(shared-Pong-1).pkl')) For Pong
    #shared_model.load_state_dict(torch.load('./Breakout-v4/A3C(shared-Breakout-1).pkl')) For Breakout
    shared_model.share_memory()

    if args.optimizer:
        if args.optimizer == 'RMSprop':
            optimizer = shared_optim.SharedRMSprop(shared_model.parameters(), lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = shared_optim.SharedAdam(shared_model.parameters(), lr=args.lr)
        optimizer.share_memory()
    else:
        optimizer = None


    processes = []
    #Test Run
    p = mp.Process(target=test, args=(args.num_processes, args, shared_model))
    p.start()
    processes.append(p)
    #Run as many incarnation of the network for a given enviroment
    for rank in range(0, args.num_processes):
        p = mp.Process(target=train, args=(rank, args, shared_model, optimizer))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()