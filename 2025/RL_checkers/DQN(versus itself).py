import numpy as np
import random
import copy
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from checkers import Checkers

MAX_JUMPS = 4
REWARD_FOR_WIN = 10

learning_rate = 1e-4
discount_factor = 0.95
num_of_episodes = 190000
target_update_freq = 500

torch.autograd.set_detect_anomaly(False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

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

    def push(self, state, action, reward, next_state, done, turn):
        self.buffer.append((state, action, reward, next_state, done, turn))

    def sample(self, batch_size, turn):
        """
        batch_size: number of samples to return
        turn: which player's moves to return
        returns: [(state, action, reward, next_state, done, turn), ...] of shape [batch_size]
        """
        # Filter buffer for experiences with the requested turn
        filtered = [exp for exp in self.buffer if exp[-1] == turn]
        if len(filtered) < batch_size:
            raise ValueError(f"Not enough samples for turn={turn} (requested {batch_size}, available {len(filtered)})")
        return random.sample(filtered, batch_size)

    def __len__(self):
        return len(self.buffer)


class EpsilonScheduler:
    def __init__(self, epsilon_start=1.0, epsilon_end=0.05, decay_steps=45000):
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
    encoded = [0] * (2 + 2 * max_jumps)
    encoded[0] = fr[0]/8
    encoded[1] = fr[1]/8

    for i in range(min(len(seq), 8)):
        encoded[2 + i] = seq[i]/8

    return torch.tensor(encoded, dtype=torch.float32)

def encode_list_of_actions(possible_actions):
    encoded_actions_new = []
    for action in possible_actions:
        for a, b in action.items():
            for seq in b:
                encoded_actions_new.append(encode_actions(list(a), seq).to(device)) #[K, 10]

    return torch.stack(encoded_actions_new, dim=0).to(device)

def encode_states(board):
    ch_black_pawn = (board == -1).astype(np.float32)
    ch_white_pawn = (board == 1).astype(np.float32)
    ch_black_king = (board == -2).astype(np.float32)
    ch_white_king = (board == 2).astype(np.float32)
    state = np.stack([ch_black_pawn, ch_white_pawn, ch_black_king, ch_white_king], axis=0)
    return torch.tensor(state, dtype=torch.float32, device=device).view(-1)  # shape [400]

def playEGreedyAction(game, turn, epsilon_scheduler, Q):
    poss_actions = game.GetPossibleMoves(turn)
    if not poss_actions:
        return None, None, None, None, None

    encoded_actions, action_mapping = [], []
    for action in poss_actions:
        for fr, seqs in action.items():
            for seq in seqs:
                encoded_actions.append(encode_actions(list(fr), seq).to(device)) #[K, 10]
                action_mapping.append( (fr, seq) )

    encoded_actions = torch.stack(encoded_actions, dim=0).to(device)

    state = game.getBoard()
    encoded_board = encode_states(state) #[400]
    batch_states = encoded_board.unsqueeze(0).repeat(encoded_actions.size(0), 1).to(device).contiguous() #[K, 400]

    i = e_greedy(batch_states, encoded_actions, epsilon_scheduler, Q)
    chosen_encoding = encoded_actions[i]
    fr, seq = action_mapping[i]

    q_pred = Q(encoded_board.unsqueeze(0).to(device), chosen_encoding.unsqueeze(0))

    return fr, seq, q_pred, encoded_board, chosen_encoding

def e_greedy(state_batch, encoded_actions, epsilon_scheduler, Q, temperature=1.0, device=device):
    epsilon = epsilon_scheduler.get_epsilon()
    K = encoded_actions.size(0)
    if random.random() < epsilon:
        return random.randrange(K)

    qs = Q(state_batch, encoded_actions).to(device)

    qs = qs.view(-1)
    probs = torch.softmax(qs / temperature, dim=0)
    idx = torch.multinomial(probs, num_samples=1)

    return idx.item()

def update_loss(q_pred, q_target):
    q_target = q_target.detach()
    if q_target.dim() == 0:
        q_target = q_target.unsqueeze(0)
    return F.mse_loss(q_pred, q_target)

def update_optimizer(loss, optimizer):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def main():
    epsilon_scheduler = EpsilonScheduler()

    Q_white = QNet(400, 2 + 2 * MAX_JUMPS).to(device)
    target_Q_white = copy.deepcopy(Q_white).to(device)

    Q_black = QNet(400, 2 + 2 * MAX_JUMPS).to(device)
    target_Q_black = copy.deepcopy(Q_black).to(device)

    optimizer_white = optim.Adam(Q_white.parameters(), lr=learning_rate)
    optimizer_black = optim.Adam(Q_black.parameters(), lr=learning_rate)

    buffer = ReplayBuffer(100_000)
    step = 0

    for episod in range(num_of_episodes):
        turn = 1
        game = Checkers()

        if episod % 300 == 0:
            print(episod)

        # first step for white
        gain_white, gain_black = 0, 0

        fr_white, seq_white, q_pred_white, encoded_board_white, chosen_encoding_white = playEGreedyAction(game, turn, epsilon_scheduler=epsilon_scheduler, Q=Q_white)
        gain_white += game.takeMove(fr_white[0], fr_white[1], seq_white)
        terminal_white = game.isEnd()

        # first step for black
        fr_black, seq_black, q_pred_black, encoded_board_black, chosen_encoding_black = playEGreedyAction(game, -turn, epsilon_scheduler=epsilon_scheduler, Q=Q_black)
        gain_black += game.takeMove(fr_black[0], fr_black[1], seq_black)
        terminal_black = game.isEnd()

        
        for _ in range(1000): # max 1000 steps per game
            epsilon_scheduler.update()

            # if black captured
            if gain_black > 0:
                gain_white -= (gain_black)

            
            # Update Q for white
            next_encoded_board_white = encode_states(game.getBoard()).to(device) #[400]
        
            next_possible_actions_white = game.GetPossibleMoves(turn) 
            if not next_possible_actions_white: # No moves left
                q_target_white = -10
                q_target_black = 10
                buffer.push(encoded_board_white, chosen_encoding_white, (gain_white + q_target_white), game.getBoard(), True, turn=1)

                loss_white = update_loss(q_pred_white, torch.tensor([q_target_white], dtype=torch.float32).to(device))
                loss_black = update_loss(q_pred_black, torch.tensor([q_target_black], dtype=torch.float32).to(device))

                # update white first
                update_optimizer(loss_white, optimizer_white)
                # optimizer_white.zero_grad()
                # loss_white.backward()
                # optimizer_white.step()
                
                # update black
                update_optimizer(loss_black, optimizer_black)
                # optimizer_black.zero_grad()
                # loss_black.backward()
                # optimizer_black.step()
                break

            
            encoded_actions_new = encode_list_of_actions(next_possible_actions_white)
            batch_states = next_encoded_board_white.unsqueeze(0).repeat(encoded_actions_new.size(0), 1).to(device).contiguous() #[K, 400]
            with torch.no_grad():
                max_q_white = target_Q_white(batch_states, encoded_actions_new).to(device).max()
            q_target_white = gain_white + discount_factor * max_q_white
            loss_white = update_loss(q_pred_white, q_target_white)
            
            # update
            update_optimizer(loss_white, optimizer_white)

            # save to memory buffer
            buffer.push(encoded_board_white, chosen_encoding_white, gain_white, game.getBoard(), bool(terminal_white), turn=1)

            # make next move for white
            fr_white, seq_white, q_pred_white, encoded_board_white, chosen_encoding_white = playEGreedyAction(game, turn, epsilon_scheduler=epsilon_scheduler, Q=Q_white)
            gain_white = game.takeMove(fr_white[0], fr_white[1], seq_white)
            terminal_white = game.isEnd()

            # if black was captured
            if gain_white > 0:
                gain_black -= (gain_white)

            if terminal_white:
                winner = game.whoWon()
                final_reward = +1 if winner == turn else -1
                final_reward *= 10
                
                buffer.push(encoded_board_white, chosen_encoding_white, (gain_white + final_reward), game.getBoard(), True, turn=1)
                
                loss_white = update_loss(q_pred_white, torch.tensor([gain_white + final_reward], dtype=torch.float32).to(device))
                loss_black = update_loss(q_pred_black, torch.tensor([gain_black - final_reward], dtype=torch.float32).to(device))

                update_optimizer(loss_white, optimizer_white)
                update_optimizer(loss_black, optimizer_black)

                break

            # -----------------
            # black's turn
            # -----------------
            turn = -turn

            next_encoded_board_black = encode_states(game.getBoard()).to(device) #[400]
        
            next_possible_actions_black = game.GetPossibleMoves(turn)
            if not next_possible_actions_black: # No moves left
                q_target_white = 10
                q_target_black = -10

                buffer.push(encoded_board_black, chosen_encoding_black, (gain_black + q_target_black), game.getBoard(), True, turn=-1)

                winner = game.whoWon()
                loss_white = update_loss(q_pred_white, torch.tensor([q_target_white], dtype=torch.float32).to(device))
                loss_black = update_loss(q_pred_black, torch.tensor([q_target_black], dtype=torch.float32).to(device))

                update_optimizer(loss_white, optimizer_white)
                update_optimizer(loss_black, optimizer_black)

                break

            encoded_actions_new = encode_list_of_actions(next_possible_actions_black)
            batch_states = next_encoded_board_black.unsqueeze(0).repeat(encoded_actions_new.size(0), 1).to(device).contiguous() #[K, 400]
            with torch.no_grad():
                max_q_black = target_Q_black(batch_states, encoded_actions_new).to(device).max()
            q_target_black = gain_black + discount_factor * max_q_black
            
            loss_black = update_loss(q_pred_black, q_target_black)


            update_optimizer(loss_black, optimizer_black)

            buffer.push(encoded_board_black, chosen_encoding_black, gain_black, game.getBoard(), bool(terminal_black), turn=-1)

            # make next action for black
            fr_black, seq_black, q_pred_black, encoded_board_black, chosen_encoding_black = playEGreedyAction(game, turn, epsilon_scheduler=epsilon_scheduler, Q=Q_black)
            gain_black = game.takeMove(fr_black[0], fr_black[1], seq_black)
            terminal_black = game.isEnd()

            if terminal_black:
                winner = game.whoWon()
                final_reward = +1 if winner == turn else -1
                final_reward *= 10

                buffer.push(encoded_board_black, chosen_encoding_black, (gain_black + final_reward), game.getBoard(), True, turn=-1)

                loss_white = update_loss(q_pred_white, torch.tensor([gain_white - final_reward], dtype=torch.float32).to(device))
                loss_black = update_loss(q_pred_black, torch.tensor([gain_black + final_reward], dtype=torch.float32).to(device))
                
                update_optimizer(loss_white, optimizer_white)
                update_optimizer(loss_black, optimizer_black)

                break
            turn = -turn

            step += 1
            if step % target_update_freq == 0:
                target_Q_white.load_state_dict(Q_white.state_dict())
                target_Q_black.load_state_dict(Q_black.state_dict())

        # Memory buffer for specific player
        buffer_batch_size = 128
        if len(buffer) >= buffer_batch_size * 6:
            buffer_turn = random.choice([-1, 1])
            batch = buffer.sample(buffer_batch_size, turn=buffer_turn)

            states, actions, rewards, next_states, dones, _ = zip(*batch)
            states = torch.stack(states).to(device) # [B, 400]
            actions = torch.stack(actions).to(device) # [B, 10]
            rewards = torch.tensor(rewards).to(device) # [B]
            # next_states #[10, 10]
            dones = torch.tensor(dones).to(device) # [B] (bool/int)

            if buffer_turn == 1:
                q_pred = Q_white(states, actions) #[B]
                q_target = torch.zeros_like(q_pred).to(device)
                target_Q_buffer = target_Q_white
            else:
                q_pred = Q_black(states, actions).to(device) #[B]
                q_target = torch.zeros_like(q_pred).to(device)
                target_Q_buffer = target_Q_black
            
            for sample in range(buffer_batch_size):
                buffer_game = Checkers()
                if dones[sample]:
                    q_target[sample] = rewards[sample]
                else:
                    buffer_game.setBoard(next_states[sample])

                    possible_actions = buffer_game.GetPossibleMoves(buffer_turn)
                    if not possible_actions:
                        q_boorstrap = -10.0
                    else:
                        encoded_actions, action_mapping = [], []
                        for action in possible_actions:
                            for fr, seqs in action.items():
                                for seq in seqs:
                                    encoded_actions.append(encode_actions(list(fr), seq))
                                    action_mapping.append( (fr, seq) )
    
                        encoded_actions = torch.stack(encoded_actions, dim=0).to(device)
                        encoded_board = encode_states(next_states[sample]).to(device)
    
                        batch_states = encoded_board.unsqueeze(0).repeat(encoded_actions.size(0), 1).to(device)  # Shape: [K, 256]
    
                        q_boorstrap = target_Q_buffer(batch_states, encoded_actions).to(device).max()
                q_target[sample] = rewards[sample] + discount_factor * q_boorstrap
        
            loss = F.mse_loss(q_pred, q_target.detach())
            print(f"episod: {episod} loss: {loss.item()}")

            if buffer_turn == 1: # white
                update_optimizer(loss, optimizer_white)
            else: # black
                update_optimizer(loss, optimizer_black)

        if (episod % 30000 == 0 or episod == num_of_episodes - 1):
            save_path_white = f"qnet_weights_white_2_{episod}.pth"
            save_path_black = f"qnet_weights_black_2_{episod}.pth"
            torch.save(Q_white.state_dict(), save_path_white)
            torch.save(Q_black.state_dict(), save_path_black)
            print(f"Saved Q-network weights to {save_path_white}")
                         
if __name__ == "__main__":
    main()
    