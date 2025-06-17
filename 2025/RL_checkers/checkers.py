import numpy as np
import random
import time

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

    def movePawn(self, y_prev, x_prev, y, x):
        turn = self.board[y_prev][x_prev] 
        if turn == 0:
            return "In (x, y) nothing"

        self.board[y_prev][x_prev] = 0
        self.board[y][x] = turn

    def takeMove(self, y_prev, x_prev, new_coords):
        if abs(y_prev-new_coords[0]) == 1:
            self.movePawn(y_prev, x_prev, new_coords[0], new_coords[1])
        else:
            self.takeEat(y_prev, x_prev, new_coords)


    def takeEat(self, y_prev, x_prev, new_coords):
        y, x = new_coords[0], new_coords[1]
        self.movePawn(y_prev, x_prev, y, x)

        mid_y, mid_x = (y_prev + y)//2, (x_prev + x)//2
        self.board[mid_y][mid_x] = 0

        if len(new_coords) > 2:
            self.takeEat(y, x, new_coords[2:])


    def GetPossibleMovesForPawn(self, y, x):
        turn = self.board[y][x]

        if turn == 0:
            return "In (x, y) nothing"

        if turn == 1:
            symb = -1
        else:
            symb = 1
        
        moves = self.CanEatForPawn(y, x, turn)
        canEat = True
        if not moves:
            if (y == 1 and turn == 1) or (y == 8 and turn == -1):
                return None, False  # todo: implement king moves
            
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
                if self.board[i][j] == turn:
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
        if not moves:
            print(f"{turn} is lost")
            return 0
        
        move = random.choice(moves)
        self.takeMove(list(move.keys())[0][0], list(move.keys())[0][1], random.choice(list(move.values())[0]))
        self.show()
        print()


    def isEnd(self):
        black, white = 0, 0
        for i in range(10):
            for j in range(10):
                if self.board[i][j] == 1:
                    white = 1
                elif self.board[i][j] == -1:
                    black = 1
                if white == 1 and black == 1:
                    return False
        return True

        
def InRange(x, y):
    return 1 <= x <= 8 and 1 <= y <= 8


def main():
    
    for _ in range (1):
        check = Checkers()
        turn = 1

        while True:
            time.sleep(0.5)
            if check.playRandomMove(turn = turn) == 0:
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
    check.show()
    print(check.GetPossibleMovesForPawn(4, 3))

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



