import numpy as np
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from checkers import Checkers

import torch
import torch.nn as nn
import torch.nn.functional as F

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

class EpsilonScheduler:
    def __init__(self, epsilon_start=1.0, epsilon_end=0.05, decay_steps=10000):
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
    ch_white_pawn  = (board == +1).astype(np.float32)
    ch_black_king  = (board == -2).astype(np.float32)
    ch_white_king  = (board == +2).astype(np.float32)
    state = np.stack([ch_black_pawn, ch_white_pawn, ch_black_king, ch_white_king], axis=0)
    return torch.tensor(state).view(-1)  # shape [400]


def main():
    
    epsilon_scheduler = EpsilonScheduler()

    learning_rate = 0.1
    discount_factor = 0.95
    num_of_episodes = 10000
    
    Q = QNet(400, 2 + 2 * MAX_JUMPS)
    optimizer = optim.Adam(Q.parameters(), lr=learning_rate)

    def e_greedy(state, possible_actions):
        epsilon = epsilon_scheduler.get_epsilon()
        if random.random() < epsilon:
            return random.choice(possible_actions)
        return maxByQ(state, possible_actions)
    
    def maxByQ(state, possible_actions):
        all_qs = Q(state, possible_actions) #[K]
        return possible_actions[all_qs.argmax()]
    
    def updateQ(q_pred, q_target):
        loss = F.mse_loss(q_pred, q_target.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for episod in range(num_of_episodes):
        turn = 1
        gain = 0

        game = Checkers()

        epsilon_scheduler.update()
        while True:
                possible_actions = game.GetPossibleMoves(turn)
                encoded_actions = []
                for action in possible_actions:
                    for a, b in action.items():
                        for seq in b:
                            encoded_actions.append(encode_actions(list(a), seq)) #[K, 10]

                encoded_actions = torch.stack(encoded_actions, dim=0)
                state = game.getBoard()
                encoded_board = encode_states(state) #[400]
                batch_states = encoded_board.unsqueeze(0).repeat(len(possible_actions), 1) #[K, 400]

                move = e_greedy(batch_states, encoded_actions)
                q_pred = Q(encoded_board.unsqueeze(0), move.unsqueeze(0)) #[1]

                if not possible_actions:
                    print(f"{turn} is lost")
                    q_target = -1
                    updateQ(q_pred, q_target)
                    break

                gain = game.takeMove(list(move.keys())[0][0], list(move.keys())[0][1], random.choice(list(move.values())[0]))

                terminal = game.isEnd()
                if terminal:
                    gain += terminal
                    q_target = gain
                    updateQ(q_pred, q_target)
                    break
                
                # Update q_pred
                encoded_board_new = encode_states(game.getBoard()) #[400]
            
                possible_actions_new = game.GetPossibleMoves(turn)
                encoded_actions_new = []
                for action in possible_actions_new:
                    for a, b in action.items():
                        for seq in b:
                            encoded_actions_new.append(encode_actions(list(a), seq)) #[K, 10]

                batch_states_new = encoded_board_new.unsqueeze(0).repeat(len(possible_actions_new), 1) #[K, 400]


                q_target = gain + discount_factor * maxByQ(batch_states_new, encoded_actions_new)
                updateQ(q_pred, q_target)

                turn = -1 * turn


if __name__ == "__main__":
    main()