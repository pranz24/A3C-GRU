import torch
import torch.nn.functional as F
import torch.optim as optim
from environment import create_atari_env
from A3C_model import A3C
from torch.autograd import Variable



def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def train(rank, args, shared_model, optimizer=None):
    torch.manual_seed(args.seed + rank)

    env = create_atari_env(args.env_name)
    env.seed(args.seed + rank)

    model = A3C(env.observation_space.shape[0], env.action_space)

    if optimizer is args.optimizer:
        optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)
    else:
        optimizer = optim.RMSprop(shared_model.parameters(), lr=args.lr)
    model.train()
    state = env.reset()
    state = torch.from_numpy(state)
    done = True
    episode_length = 0
    while True:
        episode_length += 1
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())
        if done:
            hx = Variable(torch.zeros(1, 256))
        else:
            hx = Variable(hx.data)

        values = []
        log_probs = []
        rewards = []
        entropies = []
        #Take Action
        for step in range(args.num_steps):
            value, logit, hx = model(
                (Variable(state.unsqueeze(0)), hx))
            prob = F.softmax(logit)
            log_prob = F.log_softmax(logit)
            entropy = -(log_prob * prob).sum(1)
            entropies.append(entropy)

            action = prob.multinomial().data
            log_prob = log_prob.gather(1, Variable(action))
            #Compute States and Rewards
            state, reward, done, _ = env.step(action.numpy())
            done = done or episode_length >= args.max_episode_length
            #Clip Rewards from -1 to +1
            reward = max(min(reward, 1), -1)
            if done:
                episode_length = 0
                state = env.reset()
            #Append rewards, value functions, advantage and state
            state = torch.from_numpy(state)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)
            env.render()
            if done:
                break
        R = torch.zeros(1, 1)
        if not done:
            value, _, _ = model((Variable(state.unsqueeze(0)), hx))
            R = value.data
        #Calculating Gradients
        values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        R = Variable(R)
        GAE = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            #Discounted Sum of Future Rewards + reward for the given state
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)
            # Generalized Advantage Estimataion(GAE)
            delta_t = rewards[i] + args.gamma * \
                values[i + 1].data - values[i].data
            GAE = GAE * args.gamma * args.tau + delta_t
            policy_loss = policy_loss - \
                log_probs[i] * Variable(GAE) - 0.01 * entropies[i]
        optimizer.zero_grad()
        (policy_loss + 0.5 * value_loss).backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 40)
        ensure_shared_grads(model, shared_model)
        optimizer.step()
    env.close()