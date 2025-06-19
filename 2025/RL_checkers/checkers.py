import numpy as np
import random
import time

EAT_REWARD_MULT = 0.2

class Checkers:
    def __init__(self):
        self.black = np.zeros((10, 10))
        self.white = np.zeros((10, 10))
        for row in range(1, 4):
            for col in range(row % 2 + 1, 9, 2):
                self.black[row][col] = 1

        for row in range(6, 9):
            for col in range(row % 2 + 1, 9, 2):
                self.white[row][col] = 1
        
        board = np.where(self.black, -1, 0)
        self.board = np.where(self.white, 1, board)
    

    def show(self):
        print(self.board)

    def getBoard(self):
        return self.board.copy()
    
    def setBoard(self, board):
        self.board = board


    def promotePawn(self, y, x):
        # Promote to king when reaching the far edge
        val = self.board[y][x]
        if val == 1 and y == 1:
            self.board[y][x] = 2
        elif val == -1 and y == 8:
            self.board[y][x] = -2

    def movePawn(self, y_prev, x_prev, y, x):
        turn = self.board[y_prev][x_prev] 
        if turn == 0:
            return "In (x, y) nothing"

        self.board[y_prev][x_prev] = 0
        self.board[y][x] = turn
        self.promotePawn(y, x)


    def takeMove(self, y_prev, x_prev, new_coords):
        self.show()
        print()
        turn = self.board[y_prev][x_prev]
        y_dest, x_dest = new_coords[0], new_coords[1]

        if abs(turn) == 2:  # king piece
            # Determine if path crosses exactly one enemy piece (a capture) or none (slide)
            dy = y_dest - y_prev
            dx = x_dest - x_prev
            if abs(dy) != abs(dx):
                return "Illegal king move (not diagonal)"
            step_y = 1 if dy > 0 else -1
            step_x = 1 if dx > 0 else -1
            ny, nx = y_prev + step_y, x_prev + step_x
            enemies = []
            while (ny, nx) != (y_dest, x_dest):
                if self.board[ny][nx] != 0:
                    if self.board[ny][nx] * turn < 0:
                        enemies.append((ny, nx))
                    else:
                        return "Illegal: own piece blocks king path"
                ny += step_y
                nx += step_x
            if len(enemies) == 0:
                self.movePawn(y_prev, x_prev, y_dest, x_dest)
                return -0.005 #reward for doing nothing
            elif len(enemies) == 1:
                return self.takeEatKing(y_prev, x_prev, new_coords) * EAT_REWARD_MULT #reward for eating
            else:
                return "Illegal: king cannot jump over multiple enemies in one segment"

        if abs(y_prev - y_dest) == 1:
            self.movePawn(y_prev, x_prev, y_dest, x_dest)
            return -0.005 #reward for doing nothing
        else:
            return self.takeEat(y_prev, x_prev, new_coords) * EAT_REWARD_MULT #reward for eating


    def takeEat(self, y_prev, x_prev, new_coords, seq=1):
        y, x = new_coords[0], new_coords[1]
        self.movePawn(y_prev, x_prev, y, x)

        mid_y, mid_x = (y_prev + y)//2, (x_prev + x)//2
        self.board[mid_y][mid_x] = 0

        if len(new_coords) > 2:
            seq = self.takeEat(y, x, new_coords[2:], seq=seq+1)
        return seq

    def takeEatKing(self, y_prev, x_prev, new_coords, seq=1):
        y, x = new_coords[0], new_coords[1]
        turn = self.board[y_prev][x_prev]
        dy = y - y_prev
        dx = x - x_prev
        step_y = 1 if dy > 0 else -1
        step_x = 1 if dx > 0 else -1
        # find the opponent piece along the path
        ny, nx = y_prev + step_y, x_prev + step_x
        cap_y = cap_x = None

        while (ny, nx) != (y, x):
            if self.board[ny][nx] * turn < 0:
                cap_y, cap_x = ny, nx
                break
            ny += step_y
            nx += step_x

        # perform move and capture
        self.board[y_prev][x_prev] = 0
        self.board[cap_y][cap_x] = 0
        self.board[y][x] = turn
        # continue multi-capture if present
        if len(new_coords) > 2:
            seq = self.takeEatKing(y, x, new_coords[2:], seq = seq + 1)
        return seq

    def GetPossibleMovesForKing(self, y, x):
        turn = self.board[y][x]
        moves = self.CanEatForKing(y, x, turn)
        canEat = True
        if not moves:
            canEat = False

            for dy, dx in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                ny, nx = y + dy, x + dx
                while InRange(nx, ny) and self.board[ny][nx] == 0:
                    moves.append([ny, nx])
                    ny += dy
                    nx += dx
        return moves, canEat

    def CanEatForKing(self, y, x, turn):
        simple = turn // abs(turn)
        other_vals = [-simple, -2 * simple]  # opponent pawn and king
        moves = []
        # For each diagonal direction, look for an opponent piece with empty landings beyond
        for dy, dx in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            ny, nx = y + dy, x + dx
            # slide to find first non-empty square
            while InRange(nx, ny) and self.board[ny][nx] == 0:
                ny += dy
                nx += dx

            if InRange(nx, ny) and self.board[ny][nx] in other_vals:
                after_y, after_x = ny + dy, nx + dx

                while InRange(after_x, after_y) and self.board[after_y][after_x] == 0:

                    other = self.board[ny][nx]
                    self.board[y][x] = 0
                    self.board[ny][nx] = 0
                    self.board[after_y][after_x] = turn

                    next_moves = self.CanEatForKing(after_y, after_x, turn)
                    if next_moves:
                        for nm in next_moves:
                            moves.append([after_y, after_x] + nm)
                    else:
                        moves.append([after_y, after_x])

                    self.board[y][x] = turn
                    self.board[ny][nx] = other
                    self.board[after_y][after_x] = 0

                    after_y += dy
                    after_x += dx
        return moves


    def GetPossibleMovesForPawn(self, y, x):
        turn = self.board[y][x]

        if turn == 0:
            return "In (x, y) nothing"
        
        elif abs(turn) == 2:
            return self.GetPossibleMovesForKing(y, x)

        if turn >= 1:
            symb = -1
        else:
            symb = 1
        
        moves = self.CanEatForPawn(y, x, turn)
        canEat = True
        if not moves:
            
            canEat = False

            if InRange(x+1, y+symb) and self.board[y+symb][x+1] == 0:
                moves.append([y+symb, x+1])

            if InRange(x-1, y+symb) and self.board[y+symb][x-1] == 0:
                moves.append([y+symb, x-1])

        return moves, canEat

    def CanEatForPawn(self, y, x, turn):

        other_turn = -1 * turn
        moves = []

        # top-left
        if InRange(x-1, y-1) and (self.board[y-1][x-1] == other_turn or self.board[y-1][x-1] == other_turn * 2):
            if InRange(x-2, y-2) and self.board[y-2][x-2] == 0:
                # simulate capture
                other_pawn = self.board[y-1][x-1]
                self.board[y][x] = 0
                self.board[y-1][x-1] = 0
                self.board[y-2][x-2] = turn

                next_moves = self.CanEatForPawn(y-2, x-2, turn)
                if next_moves:
                    for nm in next_moves:
                        moves.append([y-2, x-2] + nm)
                else:
                    moves.append([y-2, x-2])

                # restore
                self.board[y][x] = turn
                self.board[y-1][x-1] = other_pawn
                self.board[y-2][x-2] = 0

        # top-right
        if InRange(x+1, y-1) and (self.board[y-1][x+1] == other_turn or self.board[y-1][x+1] == other_turn * 2):
            if InRange(x+2, y-2) and self.board[y-2][x+2] == 0:
                other_pawn = self.board[y-1][x+1]
                self.board[y][x] = 0
                self.board[y-1][x+1] = 0
                self.board[y-2][x+2] = turn

                next_moves = self.CanEatForPawn(y-2, x+2, turn)
                if next_moves:
                    for nm in next_moves:
                        moves.append([y-2, x+2] + nm)
                else:
                    moves.append([y-2, x+2])

                self.board[y][x] = turn
                self.board[y-1][x+1] = other_pawn
                self.board[y-2][x+2] = 0

        # bottom-left
        if InRange(x-1, y+1) and (self.board[y+1][x-1] == other_turn or self.board[y+1][x-1] == other_turn * 2):
            if InRange(x-2, y+2) and self.board[y+2][x-2] == 0:
                other_pawn = self.board[y+1][x-1]
                self.board[y][x] = 0
                self.board[y+1][x-1] = 0
                self.board[y+2][x-2] = turn

                next_moves = self.CanEatForPawn(y+2, x-2, turn)
                if next_moves:
                    for nm in next_moves:
                        moves.append([y+2, x-2] + nm)
                else:
                    moves.append([y+2, x-2])

                self.board[y][x] = turn
                self.board[y+1][x-1] = other_pawn
                self.board[y+2][x-2] = 0

        # bottom-right
        if InRange(x+1, y+1) and (self.board[y+1][x+1] == other_turn or self.board[y+1][x+1] == other_turn * 2):
            if InRange(x+2, y+2) and self.board[y+2][x+2] == 0:
                other_pawn = self.board[y+1][x+1]
                self.board[y][x] = 0
                self.board[y+1][x+1] = 0
                self.board[y+2][x+2] = turn

                next_moves = self.CanEatForPawn(y+2, x+2, turn)
                if next_moves:
                    for nm in next_moves:
                        moves.append([y+2, x+2] + nm)
                else:
                    moves.append([y+2, x+2])

                self.board[y][x] = turn
                self.board[y+1][x+1] = other_pawn
                self.board[y+2][x+2] = 0

        return moves


    def GetPossibleMoves(self, turn=None, eatable=False):
        if turn is None:
            turn = 1

        moves = []
        for i in range(10):
            for j in range(10):
                if self.board[i][j] * turn > 0:
                    move, canEat = self.GetPossibleMovesForPawn(i, j) #[[], []] or [[]]
                    if move is not None:
                        if eatable:
                            if canEat:
                                moves.append({(i, j): move})
                        else: #not eatable
                            if canEat:
                                return self.GetPossibleMoves(turn, eatable=True)
                            if move:
                                moves.append({(i, j): move})

        return moves


    def playRandomMove(self, turn=None):
        if turn is None:
            turn = 1
        moves = self.GetPossibleMoves(turn)
        # print(moves)
        if not moves:
            # print(f"{turn} is lost")
            return -1
        
        move = random.choice(moves)
        self.takeMove(list(move.keys())[0][0], list(move.keys())[0][1], random.choice(list(move.values())[0]))
        # self.show()
        # print()


    def isEnd(self):
        black = white = 0
        for i in range(10):
            for j in range(10):
                val = self.board[i][j]
                if val > 0:
                    white = 1
                elif val < 0:
                    black = 1

                if white and black:
                    return False
        return True
    
    def whoWon(self):
        for i in range(10):
            for j in range(10):
                val = self.board[i][j]
                if val > 0:
                    return 1
                elif val < 0:
                    return -1

        return 0


        
def InRange(x, y):
    return 1 <= x <= 8 and 1 <= y <= 8


def main():
    for _ in range (1):
        check = Checkers()
        turn = 1

        while True:
            time.sleep(0.2)
            if check.playRandomMove(turn = turn) == -1:
                break

            turn = -1 * turn

            if check.isEnd():
                break

def test():
    check = Checkers()
    check.movePawn(3, 4, 4, 3)
    check.movePawn(6, 3, 5, 2)
    check.movePawn(6, 5, 5, 4)
    check.movePawn(6, 1, 5, 4)

    check.movePawn(6, 7, 5, 6)
    check.movePawn(7, 8, 5, 8)
    check.movePawn(8, 7, 7, 6)

    check.movePawn(7, 2, 3, 6)
    check.movePawn(2, 5, 0, 0)

    check.movePawn(5, 2, 1, 2)
    check.movePawn(5, 6, 1, 2)
    check.show()
    print(check.GetPossibleMovesForPawn(1, 2))

if __name__ == "__main__":
    main()
    # test()


# 0, 2, 4, 6
# 1 0 1 0 1 0 1 0
# 0 1 0 1 0 1 0 1
# 1 0 1 0 1 0 1 0
# 0 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0 0
# 0 1 0 1 0 1 0 1
# 1 0 1 0 1 0 1 0
# 0 1 0 1 0 1 0 1


# black = [1 0 1 0 1 0 1 0, 
# 0 1 0 1 0 1 0 1,
# 1 0 1 0 1 0 1 0,
# 0, 0, 0..., 0, 0]



