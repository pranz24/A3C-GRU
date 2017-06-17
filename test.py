import time
from collections import deque
import torch
import torch.nn.functional as F
from environment import create_atari_env
from A3C_model import A3C
from torch.autograd import Variable


def test(rank, args, shared_model):
    torch.manual_seed(args.seed + rank)

    env = create_atari_env(args.env_name)
    env.seed(args.seed + rank)

    model = A3C(env.observation_space.shape[0], env.action_space)
    model.eval()

    state = env.reset()
    state = torch.from_numpy(state)
    reward_sum = 0
    done = True

    start_time = time.time()
    # a quick hack to prevent the agent from stucking
    actions = deque(maxlen=100)
    episode_length = 0
    while True:
        episode_length += 1
        # Sync with the shared model
        if done:
            model.load_state_dict(shared_model.state_dict())
            hx = Variable(torch.zeros(1, 256), volatile=True)
        else:
            hx = Variable(hx.data, volatile=True)

        value, logit, hx = model(
            (Variable(state.unsqueeze(0), volatile=True), hx))
        prob = F.softmax(logit)
        action = prob.max(1)[1].data.numpy()

        state, reward, done, _ = env.step(action[0, 0])
        done = done or episode_length >= args.max_episode_length
        reward_sum += reward

        # a quick hack to prevent the agent from stucking
        actions.append(action[0, 0])
        if actions.count(actions[0]) == actions.maxlen:
            done = True

        if done:
            print("Time {}, episode reward {}, episode length {}".format(
                time.strftime("%Hh %Mm %Ss",
                              time.gmtime(time.time() - start_time)),
                reward_sum, episode_length))
            #Save Shared Weights and Weights when certain scores are achieved
            if reward_sum >= 20 and args.env_name == "PongDeterministic-v4":
                print("Finished")
                torch.save(model.state_dict(), ('./A3C(Pong-1).pkl'))
                torch.save(shared_model.state_dict(), ('./A3C(Shared-Pong-1).pkl'))
                break
            elif reward_sum >= 300:
                print("Finished")
                torch.save(model.state_dict(),('./A3C(Breakout).pkl'))
                torch.save(shared_model.state_dict(),('./A3C(Shared-Breakout).pkl'))
            reward_sum = 0
            episode_length = 0
            actions.clear()
            state = env.reset()
            #Rest
            time.sleep(60)
        state = torch.from_numpy(state)

