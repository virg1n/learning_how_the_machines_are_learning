import numpy as np
import random
import time
import copy
from collections import deque


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from checkers import Checkers

MAX_JUMPS=4


class QNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state, action):
        """
        state:  tensor of shape [B, state_dim]
        action: tensor of shape [B, action_dim]
        returns: Q-values of shape [B]
        """
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x).squeeze(1)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class EpsilonScheduler:
    def __init__(self, epsilon_start=1.0, epsilon_end=0.05, decay_steps=20000):
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay_steps = decay_steps
        self.step = 0

    def get_epsilon(self):
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                  max(0, (1 - self.step / self.decay_steps))
        return epsilon

    def update(self):
        self.step += 1

def encode_actions(fr, seq, max_jumps=MAX_JUMPS):
    encoded = [0] * 10
    encoded[0] = fr[0]/8
    encoded[1] = fr[1]/8

    for i in range(min(len(seq), 8)):
        encoded[2 + i] = seq[i]/8

    return torch.tensor(encoded, dtype=torch.float32)

def encode_states(board):
    ch_black_pawn  = (board == -1).astype(np.float32)
    ch_white_pawn  = (board == 1).astype(np.float32)
    ch_black_king  = (board == -2).astype(np.float32)
    ch_white_king  = (board == 2).astype(np.float32)
    state = np.stack([ch_black_pawn, ch_white_pawn, ch_black_king, ch_white_king], axis=0)
    return torch.tensor(state).view(-1)  # shape [400]


def main():
    
    epsilon_scheduler = EpsilonScheduler()

    learning_rate = 0.005
    discount_factor = 0.95
    num_of_episodes = 10000
    
    Q = QNet(400, 2 + 2 * MAX_JUMPS)
    target_Q = copy.deepcopy(Q)
    target_update_freq = 70
    optimizer = optim.Adam(Q.parameters(), lr=learning_rate)

    def e_greedy(state_batch, encoded_actions):
        epsilon = epsilon_scheduler.get_epsilon()
        if random.random() < epsilon:
            return random.randrange(encoded_actions.size(0))
        qs = Q(state_batch, encoded_actions)
        return qs.argmax().item()
    
    def maxByQ(state, possible_actions):
        all_qs = Q(state, possible_actions) #[K]
        return possible_actions[all_qs.argmax()]
    
    def updateQ(q_pred, q_target):
        if q_target.dim() == 0:
            q_target = q_target.unsqueeze(0)
        loss = F.mse_loss(q_pred, q_target.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    buffer = ReplayBuffer(100_000)
    step = 0

    for episod in range(num_of_episodes):
        turn = 1
        gain = 0

        game = Checkers()
        if episod % 300 == 0:
            print(episod)
        
        for _ in range(1000):
                epsilon_scheduler.update()
                possible_actions = game.GetPossibleMoves(turn)
                if not possible_actions:
                    break

                encoded_actions, action_mapping = [], []
                for action in possible_actions:
                    for fr, seqs in action.items():
                        for seq in seqs:
                            encoded_actions.append(encode_actions(list(fr), seq)) #[K, 10]
                            action_mapping.append( (fr, seq) )

                encoded_actions = torch.stack(encoded_actions, dim=0)

                state = game.getBoard()
                encoded_board = encode_states(state) #[400]
                batch_states = encoded_board.unsqueeze(0).repeat(encoded_actions.size(0), 1) #[K, 400]

                i = e_greedy(batch_states, encoded_actions)
                chosen_encoding = encoded_actions[i]
                fr, seq = action_mapping[i]
                q_pred = Q(encoded_board.unsqueeze(0), chosen_encoding.unsqueeze(0))  # [1]

                gain = game.takeMove(fr[0], fr[1], seq)
                terminal = game.isEnd()

                if terminal:
                    # assign final reward (e.g. +1 for win, -1 for loss)
                    winner = game.whoWon()
                    print("win" if winner == turn else "lose")
                    final_reward = +1 if winner == turn else -1
                    q_target = gain + final_reward
                    updateQ(q_pred, torch.tensor([q_target], dtype=torch.float32))
                    break

                # random move from opponent
                if game.playRandomMove(turn = -turn) == -1: # opponent lost
                    print("win")
                    q_target = gain + 1
                    updateQ(q_pred, torch.tensor([q_target], dtype=torch.float32))
                    break

                if game.isEnd():
                    winner = game.whoWon()
                    print("win" if winner == turn else "lose")
                    final_reward = +1 if winner == turn else -1
                    q_target = gain + final_reward
                    updateQ(q_pred, torch.tensor([q_target], dtype=torch.float32))
                    break

                # Update q_pred
                encoded_board_new = encode_states(game.getBoard()) #[400]
            
                possible_actions_new = game.GetPossibleMoves(turn)
                if not possible_actions_new:
                    q_target = -1
                    updateQ(q_pred, torch.tensor([q_target], dtype=torch.float32))
                    break

                encoded_actions_new = []
                for action in possible_actions_new:
                    for a, b in action.items():
                        for seq in b:
                            encoded_actions_new.append(encode_actions(list(a), seq)) #[K, 10]

                encoded_actions_new = torch.stack(encoded_actions_new, dim=0)
                batch_states_new = encoded_board_new.unsqueeze(0).repeat(encoded_actions_new.size(0), 1) #[K, 400]
                q_target = gain + discount_factor * target_Q(batch_states_new, encoded_actions_new).max()
                updateQ(q_pred, q_target)

                buffer.push(encoded_board, chosen_encoding, gain, game.getBoard(), bool(terminal))
                step += 1
                if step % target_update_freq == 0:
                    target_Q.load_state_dict(Q.state_dict())

                # Memory buffer
                buffer_batch_size = 32
                if len(buffer) >= buffer_batch_size:
                    batch = buffer.sample(buffer_batch_size)

                    states, actions, rewards, next_states, dones = zip(*batch)
                    states = torch.stack(states) # [B, 400]
                    actions = torch.stack(actions) # [B, 10]
                    rewards = torch.tensor(rewards) # [B]
                    next_states = next_states #[10, 10]
                    dones = torch.tensor(dones) # [B] (bool/int)

                    q_pred = Q(states, actions) #[B]
                    q_target = torch.zeros_like(q_pred)

                    for sample in range(buffer_batch_size):
                        
                        buffer_game = Checkers()
                        if dones[sample]:
                            q_target[sample] = rewards[sample]
                        else:
                            buffer_game.setBoard(next_states[sample])

                            possible_actions = buffer_game.GetPossibleMoves(turn)
                            if not possible_actions:
                                break

                            encoded_actions, action_mapping = [], []
                            for action in possible_actions:
                                for fr, seqs in action.items():
                                    for seq in seqs:
                                        encoded_actions.append(encode_actions(list(fr), seq))
                                        action_mapping.append( (fr, seq) )

                            encoded_actions = torch.stack(encoded_actions, dim=0)
                            encoded_board = encode_states(next_states[sample])

                            batch_states = encoded_board.unsqueeze(0).repeat(encoded_actions.size(0), 1)  # Shape: [K, 256]
                            q_boorstrap = target_Q(batch_states, encoded_actions)
                            q_target[sample] = rewards[sample] + discount_factor * q_boorstrap.max()

                    loss = F.mse_loss(q_pred, q_target.detach())
                    optimizer.zero_grad(); 
                    loss.backward(); 
                    optimizer.step()

                         
if __name__ == "__main__":
    main()