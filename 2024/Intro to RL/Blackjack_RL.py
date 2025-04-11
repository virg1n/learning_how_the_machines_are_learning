# # Current sum: 12 - 21
# # Dealers card: 1 of these (ace - 10)
# # Do I Have a useable ace? (1, 0)

# # Action: stick
#     # +1 if your sum > dealer's sum / dealer's sum > 21
#     # 0 if your sum == dealer's sum
#     # -1 if your sum < dealer's sum

# # Action: draw
#     # -1 if your sum > 21
#     # 0 otherwise

# # Monte-carlo Learning

import copy
import random

init_deck = [
    [2, 2, 2, 2],
    [3, 3, 3, 3],
    [4, 4, 4, 4],
    [5, 5, 5, 5],
    [6, 6, 6, 6],
    [7, 7, 7, 7],
    [8, 8, 8, 8],
    [9, 9, 9, 9],
    [10, 10, 10, 10],  # Tens
    [10, 10, 10, 10],  # Jacks
    [10, 10, 10, 10],  # Queens
    [10, 10, 10, 10],  # Kings
    [11, 11, 11, 11]   # Aces (initially 11; can be adjusted to 1)
]


policy = [[[True if random.randint(0,1) == 1 else False for num in range(22)] for i in range(12)] for _ in range (2)]
# v_p = [[[0 for num in range(22)] for i in range(12)] for ace in range (2)]
returns_suma = [[[[0 for a in range(2)] for your_sum in range(22)] for dealers in range(12)] for ace in range(2)]
returns_count = [[[[0 for a in range(2)] for your_sum in range(22)] for dealers in range(12)] for ace in range(2)]
Q = [[[[0 for a in range(2)] for your_sum in range(22)] for dealers in range(12)] for ace in range(2)]



def sim_action(act, deck, your_sum, your_aces, dealers_sum, dealers_aces):
        if act:
            your_sum, your_aces = add_card(your_sum, your_aces, draw_card(deck))
            if your_sum > 21:
                return -1, your_sum, your_aces, dealers_sum, dealers_aces  # player busts
            return 0, your_sum, your_aces, dealers_sum, dealers_aces
        else:
            while dealers_sum < 17:
                dealers_sum, dealers_aces = add_card(dealers_sum, dealers_aces, draw_card(deck))
            if dealers_sum > 21: 
                return 1, your_sum, your_aces, dealers_sum, dealers_aces
            elif dealers_sum > your_sum:
                return -1, your_sum, your_aces, dealers_sum, dealers_aces
            elif dealers_sum == your_sum:
                return 0, your_sum, your_aces, dealers_sum, dealers_aces
            else:
                return 1, your_sum, your_aces, dealers_sum, dealers_aces


# def update_vp(r, your_aces, dealers_sum, your_sum):
#     # print((your_aces > 0), dealers_sum, your_sum)
#     v_p[(your_aces > 0)][dealers_sum][your_sum] += r

# def update_vp(r, your_aces, dealers_sum, your_sum):
#     ace_flag = (your_aces > 0)
#     returns_suma[ace_flag][dealers_sum][your_sum] += r
#     returns_count[ace_flag][dealers_sum][your_sum] += 1
#     v_p[ace_flag][dealers_sum][your_sum] = returns_suma[ace_flag][dealers_sum][your_sum] / returns_count[ace_flag][dealers_sum][your_sum]
    

def update_q(r, your_aces, dealers_sum, your_sum, action):
    ace_flag = (your_aces > 0)
    returns_suma[ace_flag][dealers_sum][your_sum][action] += r
    returns_count[ace_flag][dealers_sum][your_sum][action] += 1
    Q[ace_flag][dealers_sum][your_sum][action] = (returns_suma[ace_flag][dealers_sum][your_sum][action] /
                                                  returns_count[ace_flag][dealers_sum][your_sum][action])


# def play(policy, deck, your_sum, your_aces, dealers_sum, dealers_aces):
    
#     action = policy[(your_aces>0)][dealers_sum][your_sum]
#     states = [your_aces, dealers_sum, your_sum]
#     if action:
#         r, your_sum, your_aces, dealers_sum, dealers_aces = sim_action(action, deck, your_sum, your_aces, dealers_sum, dealers_aces)
#         if r == -1:
#             update_vp(r, states[0], states[1], states[2])
#             return r
#         else:
#             r = play(policy, deck, your_sum, your_aces, dealers_sum, dealers_aces)
#             update_vp(r, states[0], states[1], states[2])
#             return r
            
#     else:
#         r, your_sum, your_aces, _, dealers_aces = sim_action(action, deck, your_sum, your_aces, dealers_sum, dealers_aces)
#         update_vp(r, your_aces, dealers_sum, your_sum)
#         return r

def play(policy, deck, your_sum, your_aces, dealers_sum, dealers_aces):
    state = (your_aces, dealers_sum, your_sum)
    action = policy[(your_aces > 0)][dealers_sum][your_sum]
    a = 1 if action else 0
    if action:  
        r, your_sum, your_aces, dealers_sum, dealers_aces = sim_action(action, deck, your_sum, your_aces, dealers_sum, dealers_aces)
        if r == -1:  
            update_q(r, state[0], state[1], state[2], a)
            return r
        else:
            r = play(policy, deck, your_sum, your_aces, dealers_sum, dealers_aces)
            update_q(r, state[0], state[1], state[2], a)
            return r
    else:  
        r, your_sum, your_aces, _, dealers_aces = sim_action(action, deck, your_sum, your_aces, dealers_sum, dealers_aces)
        update_q(r, state[0], state[1], state[2], a)
        return r


def draw_card(deck):
    non_empty_ranks = [i for i, group in enumerate(deck) if group]
    random_rank = random.choice(non_empty_ranks)
    random_suit = random.randint(0, len(deck[random_rank]) - 1)
    card = deck[random_rank].pop(random_suit)
    return card

def add_card(current_sum, aces, card):
    current_sum += card
    if card == 11:
        aces += 1

    while current_sum > 21 and aces > 0:
        current_sum -= 10
        aces -= 1
    return current_sum, aces

for upd_pol_int in range(1000):
    for i in range(4000):
        deck = copy.deepcopy(init_deck)
        # print(deck)
        
        your_sum, your_aces = 0, 0
        your_sum, your_aces = add_card(your_sum, your_aces, draw_card(deck))
        your_sum, your_aces = add_card(your_sum, your_aces, draw_card(deck))
        
        dealers_sum, dealers_aces = 0, 0
        dealers_sum, dealers_aces = add_card(dealers_sum, dealers_aces, draw_card(deck))

        play(policy, deck, your_sum, your_aces, dealers_sum, dealers_aces)

    
    # for i in range(len(policy)):
    #     for j in range(len(policy[i])):
    #         for k in range(len(policy[i][j])):
    #             if v_p[i][j][k] > 0:
    #                 policy[i][j][k] = True
    #             elif v_p[i][j][k] < 0:
    #                 policy[i][j][k] = False    
    for i in range(len(policy)):
        for j in range(len(policy[i])):
            for k in range(len(policy[i][j])):
                # Compare Q-values for hit (index 1) vs. stick (index 0)
                if Q[i][j][k][1] >= Q[i][j][k][0]:
                    policy[i][j][k] = True   # choose hit
                else:
                    policy[i][j][k] = False  # choose stick

    # Q = [[[[0 for a in range(2)] for num in range(22)] for i in range(12)] for _ in range(2)]
    # returns_count = [[[[0 for a in range(2)] for num in range(22)] for i in range(12)] for _ in range(2)]
    # returns_suma = [[[[0 for a in range(2)] for num in range(22)] for i in range(12)] for _ in range(2)]

    # v_p = [[[0 for num in range(22)] for i in range(12)] for ace in range (2)]
print(policy)

print("Final improved policy (usable ace = 0):")
for dealer in range(2, 12):
    policy_line = []
    for player in range(0, 22):
        action = policy[0][dealer][player]
        policy_line.append("H" if action else "S")
    print("Dealer {}: {}".format(dealer, " ".join(policy_line)))

print("\nFinal improved policy (usable ace = 1):")
for dealer in range(2, 12):
    policy_line = []
    for player in range(0, 22):
        action = policy[1][dealer][player]
        policy_line.append("H" if action else "S")
    print("Dealer {}: {}".format(dealer, " ".join(policy_line)))
